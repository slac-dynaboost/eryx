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
        PyTorch tensor of shape (n_points, n_atoms) with atomic form factors (float64)
    """
    # Use high precision consistently
    real_dtype = torch.float64
    q_grid = q_grid.to(dtype=real_dtype)
    ff_a = ff_a.to(dtype=real_dtype)
    ff_b = ff_b.to(dtype=real_dtype)
    ff_c = ff_c.to(dtype=real_dtype)

    # Compute q^2 / (16π^2) for each q-vector
    q_squared = torch.sum(q_grid * q_grid, dim=1, keepdim=True) / (16 * torch.pi * torch.pi) # float64

    # Prepare form factor arrays
    ff_a_expanded = ff_a.unsqueeze(0) # [1, n_atoms, 4], float64
    ff_b_expanded = ff_b.unsqueeze(0) # [1, n_atoms, 4], float64
    q_squared_expanded = q_squared.unsqueeze(2) # [n_points, 1, 1], float64

    # Compute Gaussian form factor components
    exponents = -ff_b_expanded * q_squared_expanded # [n_points, n_atoms, 4], float64
    exponents = torch.clamp(exponents, min=-50.0, max=50.0)
    exp_terms = torch.exp(exponents) # [n_points, n_atoms, 4], float64
    gaussian_terms = ff_a_expanded * exp_terms # [n_points, n_atoms, 4], float64

    # Sum over the last dimension
    ff_gauss = torch.sum(gaussian_terms, dim=2) # [n_points, n_atoms], float64

    # Add the constant term c
    form_factors = ff_gauss + ff_c.unsqueeze(0) # [n_points, n_atoms], float64

    # Return with high precision
    assert form_factors.dtype == real_dtype, f"compute_form_factors output dtype is {form_factors.dtype}, expected {real_dtype}"
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
    where q_grid has shape [n_points, 3]. It rigorously enforces float64/complex128 precision.

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
        PyTorch tensor containing structure factors (complex128)
    """
    # Define high precision dtypes
    real_dtype = torch.float64
    complex_dtype = torch.complex128
    device = q_grid.device # Assume all inputs should be on the same device as q_grid

    # --- Input Casting and Validation ---
    q_grid = q_grid.to(device=device, dtype=real_dtype)
    xyz = xyz.to(device=device, dtype=real_dtype)
    ff_a = ff_a.to(device=device, dtype=real_dtype)
    ff_b = ff_b.to(device=device, dtype=real_dtype)
    ff_c = ff_c.to(device=device, dtype=real_dtype)

    if U is not None:
        U = U.to(device=device, dtype=real_dtype)
    else:
        # Create zero tensor if U is None, ensuring correct dtype and device
        U = torch.zeros(xyz.shape[0], device=device, dtype=real_dtype)

    if project_on_components is not None:
        project_on_components = project_on_components.to(device=device, dtype=real_dtype)

    # Import ComplexTensorOps for optimized complex operations
    from eryx.torch_utils import ComplexTensorOps

    # Get dimensions
    n_points = q_grid.shape[0]
    n_atoms = xyz.shape[0]

    # --- Core Calculation with High Precision ---

    # Compute form factors efficiently - expect float64
    fj = compute_form_factors(q_grid, ff_a, ff_b, ff_c)
    assert fj.dtype == real_dtype, f"fj dtype mismatch after compute_form_factors: {fj.dtype}"

    # Compute q·r for all q-vectors and all atoms efficiently
    q_dot_r = torch.matmul(q_grid, xyz.transpose(0, 1)) # float64
    assert q_dot_r.dtype == real_dtype, f"q_dot_r dtype mismatch: {q_dot_r.dtype}"

    # Compute complex exponentials e^(i*q·r) efficiently
    real_part, imag_part = ComplexTensorOps.complex_exp(q_dot_r) # float64, float64
    assert real_part.dtype == real_dtype and imag_part.dtype == real_dtype, f"Complex exp parts dtype mismatch: {real_part.dtype}, {imag_part.dtype}"

    # Calculate Debye-Waller factors
    q_squared = torch.sum(q_grid * q_grid, dim=1, keepdim=True)  # [n_points, 1], float64
    assert q_squared.dtype == real_dtype, f"q_squared dtype mismatch: {q_squared.dtype}"

    U_expanded = U.unsqueeze(0) # [1, n_atoms], float64
    assert U_expanded.dtype == real_dtype, f"U_expanded dtype mismatch: {U_expanded.dtype}"

    dwf_exponent = -0.5 * q_squared * U_expanded # [n_points, n_atoms], float64
    assert dwf_exponent.dtype == real_dtype, f"dwf_exponent dtype mismatch: {dwf_exponent.dtype}"

    dwf = torch.exp(dwf_exponent) # [n_points, n_atoms], float64
    assert dwf.dtype == real_dtype, f"dwf dtype mismatch: {dwf.dtype}"

    # Combine form factors, phase factors, and DW factors
    A_real = (fj * real_part * dwf) # float64
    A_imag = (fj * imag_part * dwf) # float64
    assert A_real.dtype == real_dtype and A_imag.dtype == real_dtype, f"Initial A_real/A_imag dtype mismatch: {A_real.dtype}, {A_imag.dtype}"

    # Handle compute_qF option: multiply by q-vectors
    if compute_qF:
        A_real = A_real.unsqueeze(-1) # [n_points, n_atoms, 1], float64
        A_imag = A_imag.unsqueeze(-1) # [n_points, n_atoms, 1], float64

        q_expanded = q_grid.unsqueeze(1) # [n_points, 1, 3], float64

        # Perform multiplication
        A_real_q = A_real * q_expanded # [n_points, n_atoms, 3], float64
        A_imag_q = A_imag * q_expanded # [n_points, n_atoms, 3], float64

        # Reshape to (n_points, n_atoms*3)
        A_real = A_real_q.reshape(n_points, -1) # float64
        A_imag = A_imag_q.reshape(n_points, -1) # float64
        assert A_real.dtype == real_dtype and A_imag.dtype == real_dtype, f"qF A_real/A_imag dtype mismatch: {A_real.dtype}, {A_imag.dtype}"

    # Handle optional projection onto components
    if project_on_components is not None:
        expected_proj_dim = n_atoms * 3 if compute_qF else n_atoms
        if project_on_components.shape[0] != expected_proj_dim:
             raise ValueError(f"Projection matrix shape mismatch: Expected ({expected_proj_dim}, N), got {project_on_components.shape}")
        assert project_on_components.dtype == real_dtype, f"Projection matrix dtype mismatch: {project_on_components.dtype}"

        A_real = torch.matmul(A_real, project_on_components) # float64
        A_imag = torch.matmul(A_imag, project_on_components) # float64
        assert A_real.dtype == real_dtype and A_imag.dtype == real_dtype, f"Projected A_real/A_imag dtype mismatch: {A_real.dtype}, {A_imag.dtype}"

    # Handle atom summation option
    if sum_over_atoms:
        A_real = torch.sum(A_real, dim=1) # float64
        A_imag = torch.sum(A_imag, dim=1) # float64
        assert A_real.dtype == real_dtype and A_imag.dtype == real_dtype, f"Summed A_real/A_imag dtype mismatch: {A_real.dtype}, {A_imag.dtype}"

    # Final check before creating complex tensor
    assert A_real.dtype == real_dtype, f"Final A_real dtype is {A_real.dtype}, expected {real_dtype}"
    assert A_imag.dtype == real_dtype, f"Final A_imag dtype is {A_imag.dtype}, expected {real_dtype}"

    # Return complex structure factors with high precision (complex128)
    final_complex = torch.complex(A_real, A_imag)
    assert final_complex.dtype == complex_dtype, f"Final structure factor dtype is {final_complex.dtype}, expected {complex_dtype}"

    # Ensure requires_grad is set if any input required grad
    if (q_grid.requires_grad or xyz.requires_grad or
        (U is not None and U.requires_grad) or
        (project_on_components is not None and project_on_components.requires_grad)):
        final_complex.requires_grad_(True)

    return final_complex

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
    where q_grid has shape [n_points, 3]. It ensures high precision (float64/complex128) is used internally.

    Args:
        q_grid: Q-vector tensor with shape [n_points, 3]
        xyz: Atomic coordinates with shape [n_atoms, 3]
        ff_a, ff_b: Form factor coefficients with shape [n_atoms, 4]
        ff_c: Form factor coefficient with shape [n_atoms]
        U: Atomic displacement parameters with shape [n_atoms] (optional)
        n_processes: Number of processes for parallel computation (ignored for PyTorch)
        compute_qF: If True, compute q-weighted structure factors
        project_on_components: Optional projection matrix with shape [n_atoms*3 or n_atoms, n_dof]
        sum_over_atoms: If True, sum over atoms to get total structure factor

    Returns:
        Structure factors tensor (complex128) with shape [n_points] or [n_points, n_dof]
    """
    # Determine device from q_grid
    device = q_grid.device
    # Set high precision dtypes
    real_dtype = torch.float64
    complex_dtype = torch.complex128

    # --- Convert/Ensure Inputs are Tensors with Correct Dtype/Device ---
    # Use a helper function for clarity
    def _ensure_tensor(data, target_dtype):
        if not isinstance(data, torch.Tensor):
            return torch.tensor(data, dtype=target_dtype, device=device)
        return data.to(dtype=target_dtype, device=device)

    q_grid = _ensure_tensor(q_grid, real_dtype)
    xyz = _ensure_tensor(xyz, real_dtype)
    ff_a = _ensure_tensor(ff_a, real_dtype)
    ff_b = _ensure_tensor(ff_b, real_dtype)
    ff_c = _ensure_tensor(ff_c, real_dtype)

    if U is not None:
        U = _ensure_tensor(U, real_dtype)
    # No need for else, structure_factors_batch handles None U

    if project_on_components is not None:
        project_on_components = _ensure_tensor(project_on_components, real_dtype)

    # Process all q-vectors in a single batch operation
    result = structure_factors_batch(
        q_grid, xyz, ff_a, ff_b, ff_c, U,
        compute_qF, project_on_components, sum_over_atoms
    )
    assert result.dtype == complex_dtype, f"Structure factors wrapper output dtype is {result.dtype}, expected {complex_dtype}"
    return result
