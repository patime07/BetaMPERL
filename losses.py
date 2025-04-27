import torch
import torch.nn.functional as F
#for testing purposes I hide the next line

import numpy as np
import torch.nn as nn

#these functions convert network outputs into evidence scores using different activation functions.
def relu_evidence(y):
    return F.relu(y)
def exp_evidence(y):
    return torch.exp(torch.clamp(y, -10, 10))
def softplus_evidence(y):
    return F.softplus(y)


#Fatima Zahra Iguenfer Feb 27 2025
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
    """
    # Calculate loss using digamma functions
    edl_loss = torch.mean(targets * (torch.digamma(B_alpha + B_beta) - torch.digamma(B_alpha)) + 
                         (1 - targets) * (torch.digamma(B_alpha + B_beta) - torch.digamma(B_beta)))
    return edl_loss

def beta_kl_divergence(alpha, beta, num_classes, device=None):
    """
    KL divergence for Beta distribution against Beta(1,1) prior
    """
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
    """
    if device is None:
        device = outputs_alpha.device
    
    # Get evidence scores
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
    
    # KL regularization term
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

# Test code the beta edl loss functions
if __name__ == "__main__":
    # Test the Beta EDL criterion
    batch_size = 4
    num_classes = 3
    
    # Create sample alpha and beta parameters (all > 1)
    alpha = torch.tensor([[2.0, 1.5, 3.0], 
                          [1.5, 4.0, 1.2],
                          [2.5, 1.8, 1.3],
                          [1.8, 2.1, 3.5]], requires_grad=True)
    
    beta = torch.tensor([[1.5, 3.0, 1.2], 
                         [3.5, 1.2, 4.0],
                         [1.2, 1.5, 3.8],
                         [2.2, 1.9, 1.6]], requires_grad=True)
    
    # Binary target labels
    targets = torch.tensor([[1, 0, 1], 
                           [0, 1, 0],
                           [1, 0, 0],
                           [0, 1, 1]], dtype=torch.float)
    
    # Test individual components
    loss_criterion = beta_edl_criterion(alpha, beta, targets)
    print("Basic EDL Loss:", loss_criterion.item())
    
    # Test KL divergence
    kl_div = beta_kl_divergence(alpha, beta, num_classes)
    print("KL Divergence shape:", kl_div.shape)
    print("KL Divergence mean:", kl_div.mean().item())
    
    # Test with probability weighting
    p = torch.tensor([0.7, 0.8, 0.9, 0.6])  # Step probabilities
    full_loss = beta_edl_loss(alpha, beta, targets, 5, num_classes, 10, p)
    print("Full Beta EDL Loss:", full_loss.item())
    
    # Test backward pass
    full_loss.backward()
    print("Alpha gradients exist:", alpha.grad is not None)
    print("Beta gradients exist:", beta.grad is not None)
    
    print("Test completed successfully!")

# New Beta Loss
class BetaReconstructionLoss(nn.Module):
    """
    Reconstruction loss for Beta-based MPERL
    Computes weighted loss across all Markov process steps
    """
    def __init__(self, num_classes, annealing_step=10, device=None):
        super().__init__()
        self.num_classes = num_classes
        self.annealing_step = annealing_step
        self.device = device
    
    def forward(self, p, alphas, betas, targets, epoch_num):
        """
        Args:
            p: Step probabilities [num_steps, batch_size]
            alphas: Alpha parameters [num_steps, batch_size, num_classes]
            betas: Beta parameters [num_steps, batch_size, num_classes]
            targets: Ground truth labels [batch_size, num_classes]
            epoch_num: Current epoch number for annealing
        """
        total_loss = 0.0
        
        # Iterate over all steps in the Markov process
        for n in range(p.shape[0]):
            # Calculate EDL loss for this step
            step_loss = beta_edl_loss(
                alphas[n], betas[n], targets, epoch_num, 
                self.num_classes, self.annealing_step, p[n], self.device
            )
            
            # Add to total loss
            total_loss = total_loss + step_loss
            
        return total_loss


# Test beta reconstruction loss
if __name__ == "__main__":
    # ... (previous test code)
    
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
        alpha_step = torch.ones(batch_size, num_classes) * (2.0 + s * 0.2)
        beta_step = torch.ones(batch_size, num_classes) * (1.5 + s * 0.1)
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
    ], requires_grad=True)
    
    # Binary target labels
    targets = torch.tensor([
        [1, 0, 1],
        [0, 1, 0],
        [1, 0, 0],
        [0, 1, 1]
    ], dtype=torch.float)
    
    # Initialize the reconstruction loss
    rec_loss = BetaReconstructionLoss(num_classes=num_classes, annealing_step=10)
    
    # Test the loss
    loss_value = rec_loss(p, alphas, betas, targets, epoch_num=5)
    
    print(f"Reconstruction Loss: {loss_value.item()}")
    
    # Test backward pass
    loss_value.backward()
    print("Alphas gradient shape:", alphas.grad.shape)
    print("Betas gradient shape:", betas.grad.shape)
    
    print("BetaReconstructionLoss test completed successfully!")