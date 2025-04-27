import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.nn.modules.module import Module
import numpy as np
import scipy.sparse as sp


def csr2tensor(A):
    """Convert CSR sparse matrix to PyTorch sparse tensor (always on CPU)"""
    coo = A.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))
    
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape

    # Always create sparse tensors on CPU (MPS doesn't support them)
    return torch.sparse_coo_tensor(i, v, torch.Size(shape)).coalesce()


class BetaMPERLGraphConvLayer(nn.Module):
    def __init__(self, input_size, output_size, num_bases, num_rel, bias=True, device=None):
        super(BetaMPERLGraphConvLayer, self).__init__()
        
        # Auto-detect device (prefer MPS if available)
        if device is None:
            device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.device = device
        self.use_sparse_on_device = device.type not in ['mps']  # True if we can use sparse operations directly on device

        self.input_size = input_size
        self.output_size = output_size
        self.num_bases = num_bases
        self.num_rel = num_rel

        # R-GCN weights for alpha parameters
        if num_bases > 0:
            self.w_bases_alpha = Parameter(torch.FloatTensor(num_bases, input_size, output_size))
            self.w_rel_alpha = Parameter(torch.FloatTensor(num_rel, num_bases))
        else:
            self.w_alpha = Parameter(torch.FloatTensor(num_rel, input_size, output_size))
            
        # R-GCN weights for beta parameters
        if num_bases > 0:
            self.w_bases_beta = Parameter(torch.FloatTensor(num_bases, input_size, output_size))
            self.w_rel_beta = Parameter(torch.FloatTensor(num_rel, num_bases))
        else:
            self.w_beta = Parameter(torch.FloatTensor(num_rel, input_size, output_size))
            
        # Class-specific scale
        self.class_scale = Parameter(torch.ones(output_size))
        
        # Optional biases
        if bias:
            self.bias_alpha = Parameter(torch.FloatTensor(output_size))
            self.bias_beta = Parameter(torch.FloatTensor(output_size))
        else:
            self.register_parameter('bias_alpha', None)
            self.register_parameter('bias_beta', None)

        self.reset_parameters()
        
        # Move all parameters to the specified device
        self.to(device)

    def reset_parameters(self):
        """Initialize parameters with appropriate distributions"""
        if self.num_bases > 0:
            nn.init.xavier_uniform_(self.w_bases_alpha.data, gain=0.8)
            nn.init.xavier_uniform_(self.w_rel_alpha.data, gain=0.8)
            nn.init.xavier_uniform_(self.w_bases_beta.data, gain=1.2)
            nn.init.xavier_uniform_(self.w_rel_beta.data, gain=1.2)
        else:
            nn.init.xavier_uniform_(self.w_alpha.data, gain=0.8)
            nn.init.xavier_uniform_(self.w_beta.data, gain=1.2)

        nn.init.normal_(self.class_scale.data, mean=1.0, std=0.1)

        if self.bias_alpha is not None:
            nn.init.constant_(self.bias_alpha.data, 0.2)
        if self.bias_beta is not None:
            nn.init.constant_(self.bias_beta.data, 0.3)

    def _compute_weights(self):
        """Compute relation-specific weights using basis decomposition"""
        device = self.device
        
        if self.num_bases > 0:
            w_alpha = torch.einsum('rb, bio -> rio', self.w_rel_alpha, self.w_bases_alpha)
            w_beta = torch.einsum('rb, bio -> rio', self.w_rel_beta, self.w_bases_beta)
        else:
            w_alpha = self.w_alpha
            w_beta = self.w_beta
            
        return w_alpha.to(device), w_beta.to(device)

    def _process_adjacency_matrices(self, A, X):
        """Process adjacency matrices and prepare for message passing"""
        device = self.device
        cpu_device = torch.device("cpu")
        
        # Safe handling of X
        if X is not None:
            X_safe = torch.nan_to_num(X, nan=0.0)
        else:
            return [torch.zeros(0) for _ in range(self.num_rel)]
            
        supports = []
        
        # Choose computation path based on device capabilities
        if self.use_sparse_on_device:
            # Can use sparse operations directly on device
            for i in range(self.num_rel):
                # Normalize adjacency matrix
                row_sum = A[i].sum(axis=1).A1
                norm_factor = 1.0 / (row_sum + 1e-8)
                A_norm = A[i].multiply(norm_factor[:, None])
                
                # Convert to sparse tensor directly on device
                A_tensor = csr2tensor(A_norm).to(device)
                
                # Compute sparse matrix multiplication
                support = torch.sparse.mm(A_tensor, X_safe)
                supports.append(support)
        else:
            # Need to use CPU for sparse operations
            X_safe_cpu = X_safe.to(cpu_device)
            
            # Process all adjacency matrices in one batch to reduce transfers
            for i in range(self.num_rel):
                # Normalize adjacency matrix
                row_sum = A[i].sum(axis=1).A1
                norm_factor = 1.0 / (row_sum + 1e-8)
                A_norm = A[i].multiply(norm_factor[:, None])
                
                # Convert to sparse tensor on CPU
                A_tensor = csr2tensor(A_norm)
                
                # Perform sparse multiplication on CPU
                support_cpu = torch.sparse.mm(A_tensor, X_safe_cpu)
                
                # Transfer result back to device
                supports.append(support_cpu.to(device))
                
        return supports

    def forward(self, A, X, l=0):
        """
        Forward pass of the layer
        
        Args:
            A: List of adjacency matrices for each relation
            X: Node feature matrix
            l: Layer index (not used in current implementation)
            
        Returns:
            alpha, beta: Parameters for Beta distributions
        """
        device = self.device
        
        # Handle None inputs and move to correct device
        if X is not None:
            X = X.to(device)
        
        # Compute weights
        w_alpha, w_beta = self._compute_weights()
        
        # Process adjacency matrices and get supports
        supports = self._process_adjacency_matrices(A, X)
        
        if not supports[0].numel():
            # Handle empty supports case (no neighbors)
            batch_size = A[0].shape[0]
            return (torch.ones(batch_size, self.output_size, device=device) * 1.01, 
                    torch.ones(batch_size, self.output_size, device=device) * 1.01)
        
        # Concatenate all supports
        tmp = torch.cat(supports, dim=1).to(device)
        
        # Reshape weights for matrix multiplication
        weights_alpha = w_alpha.reshape(-1, self.output_size)
        weights_beta = w_beta.reshape(-1, self.output_size)

        # Replace inplace operations
        out_alpha = F.relu(torch.mm(tmp.float(), weights_alpha))  # Non-inplace
        out_beta = F.relu(torch.mm(tmp.float(), weights_beta))    # Non-inplace
        
        # Replace inplace additions
        if self.bias_alpha is not None:
            out_alpha = out_alpha + self.bias_alpha.unsqueeze(0)  # Non-inplace
        if self.bias_beta is not None:
            out_beta = out_beta + self.bias_beta.unsqueeze(0)     # Non-inplace
        
        alpha = 1.01 + F.softplus(out_alpha)  # Non-inplace
        beta = 1.01 + F.softplus(out_beta)    # Non-inplace

        return alpha, beta


# Test code (when run as main script)
if __name__ == "__main__":
    import numpy as np
    import scipy.sparse as sp
    import time

    # Automatically choose device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    n_nodes = 5
    input_size = 10
    output_size = 8
    num_bases = 2
    num_rel = 3

    # Create synthetic sparse adjacency matrices
    A = [sp.csr_matrix(np.random.rand(n_nodes, n_nodes)) for _ in range(num_rel)]
    X = torch.rand(n_nodes, input_size)

    # Create layer
    layer = BetaMPERLGraphConvLayer(input_size, output_size, num_bases, num_rel, bias=True, device=device)
    
    # Test forward pass
    start_time = time.time()
    alpha, beta = layer(A, X)
    elapsed = time.time() - start_time
    
    print(f"Forward pass took {elapsed*1000:.2f} ms")
    print("Alpha:", alpha)
    print("Beta:", beta)

    p = alpha / (alpha + beta)
    print("Expected Probability (p):", p)

    assert torch.all(p >= 0) and torch.all(p <= 1), "Expected probability p must lie in [0, 1]."
    print("Test passed")