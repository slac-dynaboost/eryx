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
    
    This optimized version efficiently handles fully collapsed tensor format
    where q_grid has shape [n_points, 3] with n_points = h_dim * k_dim * l_dim.
    
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
    # Ensure high precision
    q_grid = q_grid.to(dtype=torch.float64)
    ff_a = ff_a.to(dtype=torch.float64)
    ff_b = ff_b.to(dtype=torch.float64)
    ff_c = ff_c.to(dtype=torch.float64)
    # Compute q^2 / (16π^2) for each q-vector
    # Shape: [n_points, 1]
    q_squared = torch.sum(q_grid * q_grid, dim=1, keepdim=True) / (16 * torch.pi * torch.pi)
    
    # Prepare form factor arrays
    # ff_a, ff_b: [n_atoms, 4] → [1, n_atoms, 4]
    # q_squared: [n_points, 1] → [n_points, 1, 1]
    ff_a_expanded = ff_a.unsqueeze(0)
    ff_b_expanded = ff_b.unsqueeze(0)
    q_squared_expanded = q_squared.unsqueeze(2)
    
    # Compute Gaussian form factor components: a_i * exp(-b_i * s^2)
    # Broadcasting: [n_points, 1, 1] * [1, n_atoms, 4] = [n_points, n_atoms, 4]
    # The exponential is applied to all elements simultaneously
    exponents = -ff_b_expanded * q_squared_expanded
    
    # Clamp extreme values to avoid numerical issues
    exponents = torch.clamp(exponents, min=-50.0, max=50.0)
    
    # Compute the exponential terms
    exp_terms = torch.exp(exponents)
    
    # Multiply by a coefficients and sum over the 4 Gaussian components
    # [n_points, n_atoms, 4] × [1, n_atoms, 4] = [n_points, n_atoms, 4]
    gaussian_terms = ff_a_expanded * exp_terms
    
    # Sum over the last dimension to get [n_points, n_atoms]
    ff_gauss = torch.sum(gaussian_terms, dim=2)
    
    # Add the constant term c
    # [n_points, n_atoms] + [1, n_atoms] = [n_points, n_atoms]
    form_factors = ff_gauss + ff_c.unsqueeze(0)
    
    return form_factors

def structure_factors_batch(q_grid: torch.Tensor, xyz: torch.Tensor, 
                           ff_a: torch.Tensor, ff_b: torch.Tensor, ff_c: torch.Tensor, 
                           U: Optional[torch.Tensor] = None,
                           compute_qF: bool = False, 
                           project_on_components: Optional[torch.Tensor] = None,
                           sum_over_atoms: bool = True) -> torch.Tensor:
    """
    Compute structure factors for an atomic model at the given q-vectors using PyTorch.
    
    This optimized version efficiently handles fully collapsed tensor format
    where q_grid has shape [n_points, 3] with n_points = h_dim * k_dim * l_dim.
    
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
    # Use high precision
    dtype = torch.float64
    complex_dtype = torch.complex128
    
    # Ensure all input tensors have consistent high precision
    q_grid = q_grid.to(dtype=dtype)
    xyz = xyz.to(dtype=dtype)
    ff_a = ff_a.to(dtype=dtype)
    ff_b = ff_b.to(dtype=dtype)
    ff_c = ff_c.to(dtype=dtype)
    
    if U is not None:
        U = U.to(dtype=dtype)
    
    if project_on_components is not None:
        project_on_components = project_on_components.to(dtype=dtype)
        
    # Import ComplexTensorOps for optimized complex operations
    from eryx.torch_utils import ComplexTensorOps
    
    # Get dimensions
    n_points = q_grid.shape[0]
    n_atoms = xyz.shape[0]
    
    # Handle None for U by creating zero tensor
    if U is None:
        U = torch.zeros(n_atoms, device=xyz.device, dtype=dtype)
    
    # Compute form factors efficiently
    fj = compute_form_factors(q_grid, ff_a, ff_b, ff_c)  # [n_points, n_atoms]
    
    # Compute q·r for all q-vectors and all atoms efficiently
    # q_grid: [n_points, 3], xyz: [n_atoms, 3]
    # Result: [n_points, n_atoms]
    q_dot_r = torch.matmul(q_grid, xyz.transpose(0, 1))
    
    # Compute complex exponentials e^(i*q·r) efficiently
    real_part, imag_part = ComplexTensorOps.complex_exp(q_dot_r)
    
    # Calculate Debye-Waller factors
    # qUq = |q|^2 * U
    q_squared = torch.sum(q_grid * q_grid, dim=1, keepdim=True)  # [n_points, 1]
    dwf = torch.exp(-0.5 * q_squared * U.unsqueeze(0))
    
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
    
    # Return complex structure factors with high precision
    # For PyTorch, we'll use a tensor with an extra dimension for real/imag parts
    return torch.complex(A_real, A_imag).to(dtype=torch.complex128)

def structure_factors(q_grid: torch.Tensor, xyz: torch.Tensor, 
                     ff_a: torch.Tensor, ff_b: torch.Tensor, ff_c: torch.Tensor, 
                     U: Optional[torch.Tensor] = None,
                     n_processes: int = 1,
                     compute_qF: bool = False, 
                     project_on_components: Optional[torch.Tensor] = None,
                     sum_over_atoms: bool = True) -> torch.Tensor:
    """
    Calculate structure factors for a set of q-vectors.
    
    This function processes all q-vectors in a single operation using the fully collapsed format
    where q_grid has shape [n_points, 3] with n_points = h_dim * k_dim * l_dim.
    
    Args:
        q_grid: Q-vector tensor with shape [n_points, 3]
        xyz: Atomic coordinates with shape [n_atoms, 3]
        ff_a, ff_b: Form factor coefficients with shape [n_atoms, 4]
        ff_c: Form factor coefficient with shape [n_atoms]
        U: Atomic displacement parameters with shape [n_atoms] (optional)
        n_processes: Number of processes for parallel computation (ignored for PyTorch)
        compute_qF: If True, compute q-weighted structure factors
        project_on_components: Optional projection matrix with shape [n_atoms*3, n_dof]
        sum_over_atoms: If True, sum over atoms to get total structure factor
        
    Returns:
        Structure factors tensor with shape [n_points] or [n_points, n_dof]
    """
    # Ensure all inputs are PyTorch tensors on the same device
    device = q_grid.device
    # Use high precision for all inputs
    dtype = torch.float64
    
    if not isinstance(xyz, torch.Tensor):
        xyz = torch.tensor(xyz, dtype=dtype, device=device)
    if not isinstance(ff_a, torch.Tensor):
        ff_a = torch.tensor(ff_a, dtype=dtype, device=device)
    if not isinstance(ff_b, torch.Tensor):
        ff_b = torch.tensor(ff_b, dtype=dtype, device=device)
    if not isinstance(ff_c, torch.Tensor):
        ff_c = torch.tensor(ff_c, dtype=dtype, device=device)
    if U is not None and not isinstance(U, torch.Tensor):
        U = torch.tensor(U, dtype=dtype, device=device)
    if project_on_components is not None and not isinstance(project_on_components, torch.Tensor):
        project_on_components = torch.tensor(project_on_components, dtype=dtype, device=device)
    
    # Ensure all tensor inputs are high precision
    q_grid = q_grid.to(dtype=dtype)
    xyz = xyz.to(dtype=dtype)
    ff_a = ff_a.to(dtype=dtype)
    ff_b = ff_b.to(dtype=dtype)
    ff_c = ff_c.to(dtype=dtype)
    if U is not None:
        U = U.to(dtype=dtype)
    if project_on_components is not None:
        project_on_components = project_on_components.to(dtype=dtype)
    
    # Process all q-vectors in a single operation
    return structure_factors_batch(
        q_grid, xyz, ff_a, ff_b, ff_c, U,
        compute_qF, project_on_components, sum_over_atoms
    )
