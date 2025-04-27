import os
import time
import torch
import sys
import torch.nn.functional as F
import torch.nn as nn
import scipy.sparse as sp
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.layers import *

# Evidence conversion functions
def relu_evidence(y):
    return F.relu(y)

def exp_evidence(y):
    return torch.exp(torch.clamp(y, -10, 10))

def softplus_evidence(y):
    return F.softplus(y)

def beta_evidence(y):
    """Convert network outputs to evidence scores"""
    return F.relu(y)

def beta_edl_criterion(B_alpha, B_beta, targets):
    """
    Beta distribution-based evidential learning loss
    
    Args:
        B_alpha: Alpha parameters [batch_size, num_classes]
        B_beta: Beta parameters [batch_size, num_classes]
        targets: Binary target labels [batch_size, num_classes]
        
    Returns:
        Loss value (scalar)
    """
    # Ensure all tensors are on the same device
    device = B_alpha.device
    targets = targets.to(device)
    
    # Calculate loss using digamma functions
    edl_loss = torch.mean(targets * (torch.digamma(B_alpha + B_beta) - torch.digamma(B_alpha)) + 
                         (1 - targets) * (torch.digamma(B_alpha + B_beta) - torch.digamma(B_beta)))
    return edl_loss

def beta_kl_divergence(alpha, beta, num_classes, device=None):
    """
    KL divergence for Beta distribution against Beta(1,1) prior
    
    Args:
        alpha: Alpha parameters
        beta: Beta parameters
        num_classes: Number of classes
        device: Computation device
        
    Returns:
        KL divergence values
    """
    # Use tensor's device if not specified
    if device is None:
        device = alpha.device
    
    # Create uniform prior Beta(1,1)
    alpha_prior = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    beta_prior = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    
    # Calculate KL divergence terms
    kl = torch.lgamma(alpha + beta) - torch.lgamma(alpha) - torch.lgamma(beta)
    kl += torch.lgamma(alpha_prior) + torch.lgamma(beta_prior) - torch.lgamma(alpha_prior + beta_prior)
    kl += (alpha - alpha_prior) * torch.digamma(alpha)
    kl += (beta - beta_prior) * torch.digamma(beta)
    kl -= (alpha + beta - alpha_prior - beta_prior) * torch.digamma(alpha + beta)
    
    return kl

def beta_edl_loss(outputs_alpha, outputs_beta, targets, epoch_num, num_classes, annealing_step, p, device=None):
    """
    Combined loss function for Beta distribution with uncertainty regularization
    
    Args:
        outputs_alpha: Alpha parameters from the model
        outputs_beta: Beta parameters from the model
        targets: Binary target labels
        epoch_num: Current epoch number for annealing
        num_classes: Number of classes
        annealing_step: Number of epochs for annealing
        p: Sample weights/probabilities
        device: Computation device
        
    Returns:
        Total loss value (scalar)
    """
    # Use tensor's device if not specified
    if device is None:
        device = outputs_alpha.device
    
    # Ensure all tensors are on the same device
    targets = targets.to(device)
    p = p.to(device)
    
    # Get evidence scores (alpha and beta directly from outputs)
    alpha = outputs_alpha
    beta = outputs_beta
    
    # Compute expected probability for loss calculation
    S = alpha + beta
    
    # Main data fitting term - careful with dimensions
    main_term = targets * (torch.digamma(S) - torch.digamma(alpha)) + \
                (1 - targets) * (torch.digamma(S) - torch.digamma(beta))
    
    # Apply weight p to each sample (maintaining dimensions)
    A = p.unsqueeze(1) * main_term
    
    # Annealing coefficient for KL divergence
    annealing_coef = min(1.0, epoch_num / annealing_step)
    
    # KL regularization term - modified alpha and beta for KL
    kl_alpha = (alpha - 1) * (1 - targets) + 1
    kl_beta = (beta - 1) * targets + 1
    
    # Calculate KL divergence for uncertainty
    kl_div = annealing_coef * beta_kl_divergence(kl_alpha, kl_beta, num_classes, device)
    
    # Apply weight p to kl_div as well
    weighted_kl = p.unsqueeze(1) * kl_div
    
    # Combine losses, maintaining dimensions until final reduction
    total_loss = A + weighted_kl
    
    # Final reduction to scalar
    return torch.mean(total_loss)

class BetaReconstructionLoss(nn.Module):
    """
    Reconstruction loss for Beta-based MPERL
    Computes weighted loss across all Markov process steps
    """
    def __init__(self, num_classes, annealing_step=10, device=None):
        super().__init__()
        self.num_classes = num_classes
        self.annealing_step = annealing_step
        
        # Use specified device or detect automatically
        if device is None:
            device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.device = device
        
        # Move to device
        self.to(device)
    
    def forward(self, p, alphas, betas, targets, epoch_num):
        """
        Args:
            p: Step probabilities [num_steps, batch_size]
            alphas: Alpha parameters [num_steps, batch_size, num_classes]
            betas: Beta parameters [num_steps, batch_size, num_classes]
            targets: Ground truth labels [batch_size, num_classes]
            epoch_num: Current epoch number for annealing
        """
        # Ensure inputs are on the correct device
        device = self.device
        targets = targets.to(device)
        
        total_loss = 0.0
        
        # Iterate over all steps in the Markov process
        for n in range(p.shape[0]):
            # Calculate EDL loss for this step
            step_loss = beta_edl_loss(
                alphas[n], betas[n], targets, epoch_num, 
                self.num_classes, self.annealing_step, p[n], device
            )
            
            # Add to total loss
            total_loss = total_loss + step_loss
            
        return total_loss

class BetaMPERLModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_bases, num_rel, num_layer, dropout, max_steps=3, featureless=True, device=None, seed=0):
        """
        Enhanced BetaMPERLModel with improved Markov process and evidence handling
        Args:
            input_size: Dimension of input features
            hidden_size: Dimension of hidden layers
            output_size: Number of output classes
            num_bases: Number of bases for relation weights
            num_rel: Number of relation types in the graph
            num_layer: Number of GCN layers
            dropout: Dropout rate
            max_steps: Maximum number of Markov steps
            featureless: Whether to use node features or not
            device: Computation device (auto-detected if None)
            seed: Random seed
        """
        torch.manual_seed(seed)
        super(BetaMPERLModel, self).__init__()
        
        # Auto-detect device if not specified
        if device is None:
            device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.max_steps = max_steps
        self.num_layer = num_layer
        self.dropout = dropout
        self.device = device
        
        # Improved lambda network with more capacity
        self.lambda_net = nn.Sequential(
            nn.Linear(3, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        ).to(device)
        
        # Layer normalization for better stability
        self.layer_norms = nn.ModuleList()
        for i in range(num_layer):
            if i < num_layer - 1:
                self.layer_norms.append(nn.LayerNorm(hidden_size).to(device))
            else:
                self.layer_norms.append(nn.LayerNorm(output_size).to(device))
        
        # GCN layers
        self.layers = nn.ModuleList()
        for i in range(self.num_layer):
            if i == 0:
                self.layers.append(BetaMPERLGraphConvLayer(
                    input_size, hidden_size, num_bases, num_rel, bias=True, device=device))
            else:
                if i == self.num_layer-1:
                    self.layers.append(BetaMPERLGraphConvLayer(
                        hidden_size, output_size, num_bases, num_rel, bias=True, device=device))
                else:
                    self.layers.append(BetaMPERLGraphConvLayer(
                        hidden_size, hidden_size, num_bases, num_rel, bias=True, device=device))
        
        # Evidence scale parameter (learned)
        self.evidence_scale = nn.Parameter(torch.ones(1, device=device) * 0.5)
        
        self.relu = nn.ReLU()
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters for better training"""
        # Initialize lambda network with small weights
        for module in self.lambda_net.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        
        # Initialize evidence scale
        nn.init.constant_(self.evidence_scale, 0.5)
    
    def calculate_lambda(self, alpha, beta):
        """
        Calculate halting probability based on uncertainty and confidence signals
        from the Beta distribution parameters
        """
        alpha = torch.clamp(alpha, min=1.1) # new 
        beta = torch.clamp(beta, min=1.1) # new
        # Calculate uncertainty using Beta distribution variance
        S = alpha + beta
        #uncertainty = beta / (S * (S + 1.0))
        uncertainty = beta / (torch.clamp(S * (S + 1.0), min=1e-5)) # new
        
        # Calculate confidence signal (normalized difference between alpha and beta)
        # confidence = torch.abs(alpha - beta) / S
        confidence = torch.abs(alpha - beta) / torch.clamp(S, min=1e-5) # new
        
        # Combine signals into a feature vector for the lambda network
        halt_signal = torch.cat([
            torch.mean(uncertainty, dim=1, keepdim=True),  # Average uncertainty
            torch.mean(confidence, dim=1, keepdim=True),   # Average confidence
            torch.max(uncertainty, dim=1, keepdim=True)[0]  # Max uncertainty
        ], dim=1)
        
        # Process through the lambda network to get halting probability
        return self.lambda_net(halt_signal)
    
# Replace inplace ReLU operations with non-inplace versions
    def process_gcn_layers(self, A, h):
        """Process input through GCN layers with residual connections"""
        current_h = h
        
        for i, (layer, norm) in enumerate(zip(self.layers, self.layer_norms)):
            alpha, beta = layer(A, current_h, i)
            
            # Replace inplace operations
            alpha = torch.clamp(alpha, min=1.1)  # Instead of alpha.clamp_(min=1.1)
            beta = torch.clamp(beta, min=1.1)    # Instead of beta.clamp_(min=1.1)
            
            if norm is not None:
                alpha = norm(alpha)
                beta = norm(beta)
            
            if i < self.num_layer - 1:
                current_h = alpha / (alpha + beta)
                current_h = F.relu(current_h)  # Non-inplace
                current_h = F.dropout(current_h, self.dropout, training=self.training)
        
        return alpha, beta
    def aggregate_evidence(self, alphas, betas, ps):
        """
        Aggregate evidence across Markov steps using halting probabilities as weights
        """
        # Get dimensions
        num_steps, batch_size, num_classes = alphas.shape
        
        # Initialize aggregated parameters
        agg_alpha = torch.zeros(batch_size, num_classes, device=self.device)
        agg_beta = torch.zeros(batch_size, num_classes, device=self.device)
        
        # Compute normalized weights directly from ps
        total_ps = ps.sum(dim=0, keepdim=True) + 1e-8  # Avoid division by zero
        
        # Loop through steps and add weighted evidence
        for step in range(num_steps):
            step_weight = (ps[step] / total_ps.squeeze()).view(batch_size, 1)
            agg_alpha += alphas[step] * step_weight
            agg_beta += betas[step] * step_weight
        
        # Safety checks
        agg_alpha = torch.clamp(agg_alpha, min=1.1)
        agg_beta = torch.clamp(agg_beta, min=1.1)
        
        return agg_alpha, agg_beta
    
    def forward(self, A, X):
        """
        Forward pass with Markov process for adaptive computation
        """
        batch_size = A[0].shape[0]
        
        # Start with actual features or create a featureless placeholder
        if X is not None:
            h = X.to(self.device)
        else:
            h = torch.ones(batch_size, self.input_size, device=self.device)
        
        # Process through GCN layers to get initial alpha and beta
        alpha, beta = self.process_gcn_layers(A, h)
        
        # Initialize Markov process variables
        p = []  # halting probabilities
        alphas = []  # alpha parameters
        betas = []  # beta parameters
        lambdas = []  # lambda values
        
        # Unhalted probability starts at 1
        un_halted_prob = torch.ones(batch_size, device=self.device)
        
        # Track halted samples
        halted = torch.zeros(batch_size, device=self.device)
        
        # Markov process iterations
        for n in range(1, self.max_steps + 1):
            # Set halting probability to 1 for the last step
            if n == self.max_steps:
                lambda_n = torch.ones(batch_size, device=self.device)
            else:
                # Calculate lambda using our improved method
                lambda_n = self.calculate_lambda(alpha, beta).squeeze(-1)
                lambda_n = torch.clamp(lambda_n, min=0.0, max=1.0)
            
            # Compute step probability
            p_n = un_halted_prob * lambda_n
            un_halted_prob = un_halted_prob * (1 - lambda_n)

            # Collect values for each step with safety checks
            p.append(p_n)
            alpha_safe = torch.clamp(alpha, min=1.1)
            beta_safe = torch.clamp(beta, min=1.1)
            alphas.append(alpha_safe)
            betas.append(beta_safe)
            lambdas.append(lambda_n)

            # For inference: probabilistic halting
            if not self.training:
                halt = torch.bernoulli(lambda_n) * (1 - halted)
                halted = halted + halt
                if halted.sum() == batch_size:
                    break
            
            # Stop if almost all probability mass has halted
            if not self.training and un_halted_prob.max() < 1e-4:
                break

            # Process next step
            alpha, beta = self.process_gcn_layers(A, h)

        # Stack all outputs
        alphas_stacked = torch.stack(alphas)
        betas_stacked = torch.stack(betas)
        
        # Ensure all values are valid
        alphas_stacked = torch.clamp(alphas_stacked, min=1.1)
        betas_stacked = torch.clamp(betas_stacked, min=1.1)
        
        return alphas_stacked, betas_stacked, torch.stack(p), torch.stack(lambdas)

# Replace the test code at the bottom of models.py with this:

if __name__ == "__main__":
    # First test the loss functions (existing test code)
    print("----- Testing Loss Functions -----")
    import time
    
    # Auto-detect device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Test the Beta EDL criterion
    batch_size = 4
    num_classes = 3
    
    # Create sample alpha and beta parameters (all > 1)
    alpha = torch.tensor([
        [2.0, 1.5, 3.0], 
        [1.5, 4.0, 1.2],
        [2.5, 1.8, 1.3],
        [1.8, 2.1, 3.5]
    ], requires_grad=True, device=device)
    
    beta = torch.tensor([
        [1.5, 3.0, 1.2], 
        [3.5, 1.2, 4.0],
        [1.2, 1.5, 3.8],
        [2.2, 1.9, 1.6]
    ], requires_grad=True, device=device)
    
    # Binary target labels
    targets = torch.tensor([
        [1, 0, 1], 
        [0, 1, 0],
        [1, 0, 0],
        [0, 1, 1]
    ], dtype=torch.float, device=device)
    
    print("Running on device:", device)
    
    # Test individual components
    start_time = time.time()
    loss_criterion = beta_edl_criterion(alpha, beta, targets)
    elapsed = time.time() - start_time
    print(f"Basic EDL Loss: {loss_criterion.item():.4f} (computed in {elapsed*1000:.2f} ms)")
    
    # Test KL divergence
    start_time = time.time()
    kl_div = beta_kl_divergence(alpha, beta, num_classes)
    elapsed = time.time() - start_time
    print(f"KL Divergence shape: {kl_div.shape}")
    print(f"KL Divergence mean: {kl_div.mean().item():.4f} (computed in {elapsed*1000:.2f} ms)")
    
    # Test with probability weighting
    p = torch.tensor([0.7, 0.8, 0.9, 0.6], device=device)  # Step probabilities
    
    start_time = time.time()
    full_loss = beta_edl_loss(alpha, beta, targets, 5, num_classes, 10, p)
    elapsed = time.time() - start_time
    print(f"Full Beta EDL Loss: {full_loss.item():.4f} (computed in {elapsed*1000:.2f} ms)")
    
    # Test backward pass
    start_time = time.time()
    full_loss.backward()
    elapsed = time.time() - start_time
    print(f"Backward pass completed in {elapsed*1000:.2f} ms")
    print("Alpha gradients exist:", alpha.grad is not None)
    print("Beta gradients exist:", beta.grad is not None)
    
    print("\nTest completed successfully!")

    # Test BetaReconstructionLoss
    print("\n----- Testing BetaReconstructionLoss -----")
    
    # Create sample data for multiple steps
    num_steps = 3
    batch_size = 4
    num_classes = 3
    
    # Create tensors with requires_grad=True
    alphas = []
    betas = []
    
    for s in range(num_steps):
        # Create different values for each step
        alpha_step = torch.ones(batch_size, num_classes, device=device) * (2.0 + s * 0.2)
        beta_step = torch.ones(batch_size, num_classes, device=device) * (1.5 + s * 0.1)
        alphas.append(alpha_step)
        betas.append(beta_step)
    
    # Stack and enable gradient tracking
    alphas = torch.stack(alphas, dim=0).requires_grad_(True)
    betas = torch.stack(betas, dim=0).requires_grad_(True)
    
    # Step probabilities (needs gradient too)
    p = torch.tensor([
        [0.2, 0.3, 0.1, 0.4],  # Step 1
        [0.5, 0.4, 0.6, 0.3],  # Step 2
        [0.3, 0.3, 0.3, 0.3],  # Step 3
    ], requires_grad=True, device=device)
    
    # Binary target labels
    targets = torch.tensor([
        [1, 0, 1],
        [0, 1, 0],
        [1, 0, 0],
        [0, 1, 1]
    ], dtype=torch.float, device=device)
    
    # Initialize the reconstruction loss
    rec_loss = BetaReconstructionLoss(num_classes=num_classes, annealing_step=10, device=device)
    
    # Test the loss
    start_time = time.time()
    loss_value = rec_loss(p, alphas, betas, targets, epoch_num=5)
    elapsed = time.time() - start_time
    
    print(f"Reconstruction Loss: {loss_value.item():.4f} (computed in {elapsed*1000:.2f} ms)")
    
    # Test backward pass
    start_time = time.time()
    loss_value.backward()
    elapsed = time.time() - start_time
    print(f"Backward pass completed in {elapsed*1000:.2f} ms")
    print("Alphas gradient shape:", alphas.grad.shape)
    print("Betas gradient shape:", betas.grad.shape)
    
    print("BetaReconstructionLoss test completed successfully!")
    



    
    # Then test the BetaMPERLModel
    print("\n----- Testing BetaMPERLModel -----")
    
    # Auto-detect device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Synthetic data parameters
    n_nodes = 10
    input_size = 16
    hidden_size = 32
    output_size = 5
    num_bases = 2
    num_rel = 3
    num_layer = 2
    dropout = 0.1
    max_steps = 3
    
    # Create synthetic sparse adjacency matrices (fixed version)
    A = []
    for _ in range(num_rel):
        # Create a dense matrix first, then convert to sparse
        dense_matrix = (np.random.rand(n_nodes, n_nodes) > 0.7).astype(np.float32)
        A.append(sp.csr_matrix(dense_matrix))
    
    # Create features
    X = torch.rand(n_nodes, input_size, device=device)
    
    # Initialize the model
    model = BetaMPERLModel(
        input_size, hidden_size, output_size, num_bases, num_rel, 
        num_layer, dropout, max_steps, featureless=False, device=device
    )
    
    print("\nModel architecture:")
    print(model)
    
    # Test forward pass
    print("\nRunning forward pass...")
    start_time = time.time()
    alphas, betas, ps, lambdas = model(A, X)
    elapsed = time.time() - start_time
    print(f"Forward pass completed in {elapsed*1000:.2f} ms")
    
    # Verify outputs
    print("\nOutput shapes:")
    print("alphas:", alphas.shape)  # [num_steps, batch_size, num_classes]
    print("betas:", betas.shape)
    print("ps:", ps.shape)  # [num_steps, batch_size]
    print("lambdas:", lambdas.shape)
    
    # Check for NaN values
    print("\nNaN checks:")
    print("alphas has NaN:", torch.isnan(alphas).any().item())
    print("betas has NaN:", torch.isnan(betas).any().item())
    print("ps has NaN:", torch.isnan(ps).any().item())
    print("lambdas has NaN:", torch.isnan(lambdas).any().item())
    
    # Value ranges
    print("\nValue ranges:")
    print("alphas min/max:", alphas.min().item(), alphas.max().item())
    print("betas min/max:", betas.min().item(), betas.max().item())
    print("ps min/max:", ps.min().item(), ps.max().item())
    print("lambdas min/max:", lambdas.min().item(), lambdas.max().item())
    
    # Compute expected probabilities
    expected_probs = alphas / (alphas + betas)
    print("\nExpected probabilities range:", expected_probs.min().item(), "to", expected_probs.max().item())
    
    # Basic validity checks
    assert torch.all(alphas > 1), "All alpha values should be > 1"
    assert torch.all(betas > 1), "All beta values should be > 1"
    assert torch.all(expected_probs >= 0) and torch.all(expected_probs <= 1), "Probabilities should be in [0,1]"
    assert torch.all(ps >= 0) and torch.all(ps <= 1), "Halting probabilities should be in [0,1]"
    
    print("\nAll BetaMPERLModel tests passed!")