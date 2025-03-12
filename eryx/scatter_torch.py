"""
PyTorch implementation of structure factor calculations for diffuse scattering.

This module contains PyTorch versions of the structure factor calculations defined
in eryx/scatter.py. All implementations maintain the same API as the NumPy versions
but use PyTorch tensors and operations to enable gradient flow.

References:
    - Original NumPy implementation in eryx/scatter.py
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, List, Optional, Union

def compute_form_factors(q_grid: torch.Tensor, ff_a: torch.Tensor, 
                         ff_b: torch.Tensor, ff_c: torch.Tensor) -> torch.Tensor:
    """
    Calculate atomic form factors at the input q-vectors using PyTorch.
    
    Args:
        q_grid: PyTorch tensor of shape (n_points, 3) containing q-vectors in Angstrom
        ff_a: PyTorch tensor of shape (n_atoms, 4) with a coefficients of atomic form factors
        ff_b: PyTorch tensor of shape (n_atoms, 4) with b coefficients of atomic form factors
        ff_c: PyTorch tensor of shape (n_atoms,) with c coefficients of atomic form factors
        
    Returns:
        PyTorch tensor of shape (n_points, n_atoms) with atomic form factors
        
    References:
        - Original implementation: eryx/scatter.py:compute_form_factors
    """
    # Calculate Q = ||q||^2 / (4*pi)^2
    Q = torch.square(torch.norm(q_grid, dim=1) / (4 * torch.pi))
    
    # Reshape Q for broadcasting with ff_a and ff_b
    # Q shape: (n_points, 1, 1)
    Q = Q.view(-1, 1, 1)
    
    # Compute the exponential terms for each atom and coefficient
    # ff_a shape: (n_atoms, 4, 1) for broadcasting
    # ff_b shape: (n_atoms, 4, 1) for broadcasting
    exp_terms = ff_a.unsqueeze(-1) * torch.exp(-1 * ff_b.unsqueeze(-1) * Q.transpose(0, 2))
    
    # Sum over coefficients (dim=1) and add ff_c
    # Result shape before adding ff_c: (n_atoms, n_points)
    fj = torch.sum(exp_terms, dim=1) + ff_c.unsqueeze(-1)
    
    # Transpose to get shape (n_points, n_atoms)
    return fj.transpose(0, 1)

def structure_factors_batch(q_grid: torch.Tensor, xyz: torch.Tensor, 
                           ff_a: torch.Tensor, ff_b: torch.Tensor, ff_c: torch.Tensor, 
                           U: Optional[torch.Tensor] = None,
                           compute_qF: bool = False, 
                           project_on_components: Optional[torch.Tensor] = None,
                           sum_over_atoms: bool = True) -> torch.Tensor:
    # Determine the primary dtype to use (prefer float64 if any input is float64)
    dtype = torch.float32
    for tensor in [q_grid, xyz, ff_a, ff_b, ff_c]:
        if tensor.dtype == torch.float64:
            dtype = torch.float64
            break
    
    # Ensure all input tensors have consistent dtype
    q_grid = q_grid.to(dtype=dtype)
    xyz = xyz.to(dtype=dtype)
    ff_a = ff_a.to(dtype=dtype)
    ff_b = ff_b.to(dtype=dtype)
    ff_c = ff_c.to(dtype=dtype)
    
    if U is not None:
        U = U.to(dtype=dtype)
    
    if project_on_components is not None:
        project_on_components = project_on_components.to(dtype=dtype)
    """
    Compute structure factors for an atomic model at the given q-vectors using PyTorch.
    
    Args:
        q_grid: PyTorch tensor of shape (n_points, 3) with q-vectors in Angstrom
        xyz: PyTorch tensor of shape (n_atoms, 3) with atomic positions in Angstrom
        ff_a: PyTorch tensor of shape (n_atoms, 4) with a coefficients
        ff_b: PyTorch tensor of shape (n_atoms, 4) with b coefficients 
        ff_c: PyTorch tensor of shape (n_atoms,) with c coefficients
        U: Optional PyTorch tensor of shape (n_atoms,) with isotropic displacement parameters
        compute_qF: If True, return structure factors times q-vectors
        project_on_components: Optional projection matrix
        sum_over_atoms: If True, sum over atoms; otherwise return per-atom values
        
    Returns:
        PyTorch tensor containing structure factors
        
    References:
        - Original implementation: eryx/scatter.py:structure_factors_batch
    """
    # Handle None for U by creating zero tensor
    if U is None:
        U = torch.zeros(xyz.shape[0], device=xyz.device)
    
    # Compute form factors
    fj = compute_form_factors(q_grid, ff_a, ff_b, ff_c)
    
    # Calculate q magnitudes
    qmags = torch.norm(q_grid, dim=1)
    
    # Calculate Debye-Waller factors
    # qUq = |q|^2 * U
    qUq = torch.square(qmags.unsqueeze(1)) * U
    
    # Import complex operations from torch_utils
    from eryx.torch_utils import ComplexTensorOps
    
    # Calculate phases (q·r)
    # q_grid: (n_points, 3), xyz.T: (3, n_atoms) -> phases: (n_points, n_atoms)
    phases = torch.matmul(q_grid, xyz.T)
    
    # Compute complex exponentials e^(iq·r)
    real_part, imag_part = ComplexTensorOps.complex_exp(phases)
    
    # Apply Debye-Waller factor
    dwf = torch.exp(-0.5 * qUq)
    
    # Combine form factors, phase factors, and DW factors
    # Form structure factors: f_j * e^(iq·r) * e^(-0.5*qUq)
    A_real = fj * real_part * dwf
    A_imag = fj * imag_part * dwf
    
    # Handle compute_qF option: multiply by q-vectors
    if compute_qF:
        # Reshape for broadcasting: (n_points, n_atoms, 1)
        A_real = A_real.unsqueeze(-1)
        A_imag = A_imag.unsqueeze(-1)
        
        # Broadcast with q: (n_points, 1, 3) -> (n_points, n_atoms, 3)
        q_expanded = q_grid.unsqueeze(1)
        
        # Multiply complex structure factors by q-vectors
        A_real_q = A_real * q_expanded
        A_imag_q = A_imag * q_expanded
        
        # Reshape to (n_points, n_atoms*3)
        A_real = A_real_q.reshape(A_real_q.shape[0], -1)
        A_imag = A_imag_q.reshape(A_imag_q.shape[0], -1)
    
    # Handle optional projection onto components
    if project_on_components is not None:
        # Matricial product: (n_points, n_atoms) × (n_atoms, n_components)
        # Ensure consistent dtype before matrix multiplication
        A_real = torch.matmul(A_real, project_on_components.to(dtype=A_real.dtype))
        A_imag = torch.matmul(A_imag, project_on_components.to(dtype=A_imag.dtype))
    
    # Handle atom summation option
    if sum_over_atoms:
        A_real = torch.sum(A_real, dim=1)
        A_imag = torch.sum(A_imag, dim=1)
    
    # Return complex structure factors
    # For PyTorch, we'll use a tensor with an extra dimension for real/imag parts
    return torch.complex(A_real, A_imag)

def structure_factors(q_grid: torch.Tensor, xyz: torch.Tensor, 
                     ff_a: torch.Tensor, ff_b: torch.Tensor, ff_c: torch.Tensor, 
                     U: Optional[torch.Tensor] = None,
                     batch_size: int = 100000, n_processes: int = 1,
                     compute_qF: bool = False, 
                     project_on_components: Optional[torch.Tensor] = None,
                     sum_over_atoms: bool = True) -> torch.Tensor:
    """
    Batched version of structure factor calculation using PyTorch.
    
    Args:
        q_grid: PyTorch tensor of shape (n_points, 3) with q-vectors in Angstrom
        xyz: PyTorch tensor of shape (n_atoms, 3) with atomic positions in Angstrom
        ff_a: PyTorch tensor of shape (n_atoms, 4) with a coefficients
        ff_b: PyTorch tensor of shape (n_atoms, 4) with b coefficients 
        ff_c: PyTorch tensor of shape (n_atoms,) with c coefficients
        U: Optional PyTorch tensor of shape (n_atoms,) with isotropic displacement parameters
        batch_size: Number of q-vectors to evaluate per batch
        n_processes: Number of processes (ignored in PyTorch implementation)
        compute_qF: If True, return structure factors times q-vectors
        project_on_components: Optional projection matrix
        sum_over_atoms: If True, sum over atoms; otherwise return per-atom values
        
    Returns:
        PyTorch tensor containing structure factors
        
    References:
        - Original implementation: eryx/scatter.py:structure_factors
    """
    # Calculate number of batches
    n_batches = (q_grid.shape[0] + batch_size - 1) // batch_size  # Ceiling division
    
    # If only one batch is needed, simply call the batch function
    if n_batches <= 1:
        return structure_factors_batch(
            q_grid, xyz, ff_a, ff_b, ff_c, U=U,
            compute_qF=compute_qF, project_on_components=project_on_components,
            sum_over_atoms=sum_over_atoms
        )
    
    # Process each batch
    batch_results = []
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, q_grid.shape[0])
        
        # Extract batch of q-vectors
        q_batch = q_grid[start_idx:end_idx]
        
        # Calculate structure factors for this batch
        batch_result = structure_factors_batch(
            q_batch, xyz, ff_a, ff_b, ff_c, U=U,
            compute_qF=compute_qF, project_on_components=project_on_components,
            sum_over_atoms=sum_over_atoms
        )
        
        # Collect batch result
        batch_results.append(batch_result)
    
    # Concatenate batch results
    return torch.cat(batch_results, dim=0)
