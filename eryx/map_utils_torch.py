"""
PyTorch implementation of map utility functions for diffuse scattering.

This module contains PyTorch versions of the map utility functions defined
in eryx/map_utils.py. All implementations maintain the same API as the NumPy versions
but use PyTorch tensors and operations to enable gradient flow.

References:
    - Original NumPy implementation in eryx/map_utils.py
"""

import numpy as np
import torch
import gemmi  # Keep gemmi imports for crystallographic data
from typing import Tuple, List, Dict, Optional, Union, Any

def generate_grid(A_inv: torch.Tensor, hsampling: Tuple[float, float, float], 
                 ksampling: Tuple[float, float, float], lsampling: Tuple[float, float, float], 
                 return_hkl: bool = False) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
    """
    Generate a grid of q-vectors based on the desired extents and spacing in hkl space.
    
    Args:
        A_inv: PyTorch tensor of shape (3, 3) with fractional cell orthogonalization matrix
        hsampling: Tuple (hmin, hmax, oversampling) for h dimension
        ksampling: Tuple (kmin, kmax, oversampling) for k dimension
        lsampling: Tuple (lmin, lmax, oversampling) for l dimension
        return_hkl: If True, return hkl indices rather than q-vectors
        
    Returns:
        Tuple containing:
            - PyTorch tensor of shape (n_points, 3) with q-vectors or hkl indices
            - Tuple with shape of 3D map
            
    References:
        - Original implementation: eryx/map_utils.py:generate_grid
    """
    # Calculate steps for each dimension
    hsteps = int(hsampling[2] * (hsampling[1] - hsampling[0]) + 1)
    ksteps = int(ksampling[2] * (ksampling[1] - ksampling[0]) + 1)
    lsteps = int(lsampling[2] * (lsampling[1] - lsampling[0]) + 1)
    
    # Create linspace for each dimension
    h_grid = torch.linspace(hsampling[0], hsampling[1], hsteps, device=A_inv.device)
    k_grid = torch.linspace(ksampling[0], ksampling[1], ksteps, device=A_inv.device)
    l_grid = torch.linspace(lsampling[0], lsampling[1], lsteps, device=A_inv.device)
    
    # Create meshgrid - using indexing='ij' to match NumPy's default behavior
    h_mesh, k_mesh, l_mesh = torch.meshgrid(h_grid, k_grid, l_grid, indexing='ij')
    
    # Get map shape
    map_shape = (h_mesh.size(0), k_mesh.size(1), l_mesh.size(2))
    
    # Reshape and reorder dimensions to match NumPy version
    hkl_grid = torch.stack([h_mesh.flatten(), k_mesh.flatten(), l_mesh.flatten()], dim=1)
    
    if return_hkl:
        return hkl_grid, map_shape
    else:
        # Calculate q_grid using matrix multiplication: q_grid = 2π * A_inv^T * hkl_grid^T
        q_grid = 2 * torch.pi * torch.matmul(A_inv.T, hkl_grid.T).T
        return q_grid, map_shape

def get_symmetry_equivalents(hkl_grid: torch.Tensor, sym_ops: Dict[int, torch.Tensor]) -> torch.Tensor:
    """
    Get symmetry equivalent Miller indices of input hkl_grid.
    
    Args:
        hkl_grid: PyTorch tensor of shape (n_points, 3) with hkl indices
        sym_ops: Dictionary mapping integer keys to rotation matrices as PyTorch tensors
        
    Returns:
        PyTorch tensor of shape (n_asu, n_points, 3) with stacked hkl indices
        
    References:
        - Original implementation: eryx/map_utils.py:get_symmetry_equivalents
    """
    # Initialize list to collect rotated hkl indices
    hkl_grid_rotated_list = []
    
    # Apply each symmetry operation
    for i, rot in sym_ops.items():
        # Apply rotation matrix to hkl_grid
        hkl_grid_rot = torch.matmul(hkl_grid, rot)
        hkl_grid_rotated_list.append(hkl_grid_rot)
    
    # Stack all rotated grids
    hkl_grid_sym = torch.stack(hkl_grid_rotated_list, dim=0)
    
    # Shape should be (n_asu, n_points, 3)
    return hkl_grid_sym

def get_ravel_indices(hkl_grid_sym: torch.Tensor, 
                    sampling: Tuple[float, float, float]) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
    """
    Map 3D hkl indices to corresponding 1D indices after raveling.
    
    Args:
        hkl_grid_sym: PyTorch tensor of shape (n_asu, n_points, 3) with symmetry equivalents
        sampling: Tuple with sampling rates along (h, k, l)
        
    Returns:
        Tuple containing:
            - PyTorch tensor of shape (n_asu, n_points) with raveled indices
            - Tuple with shape of expanded/raveled map
            
    References:
        - Original implementation: eryx/map_utils.py:get_ravel_indices
    """
    # Reshape for processing
    n_asu, n_points, _ = hkl_grid_sym.shape
    hkl_grid_stacked = hkl_grid_sym.reshape(-1, 3)
    
    # Convert to integer indices with scaling
    sampling_tensor = torch.tensor(sampling, device=hkl_grid_sym.device)
    hkl_grid_int = torch.round(hkl_grid_stacked * sampling_tensor).long()
    
    # Find bounds and calculate map shape
    lbounds, _ = torch.min(hkl_grid_int, dim=0)
    ubounds, _ = torch.max(hkl_grid_int, dim=0)
    map_shape_ravel = tuple((ubounds - lbounds + 1).tolist())
    
    # Reshape back to original dimensions
    hkl_grid_int = hkl_grid_int.reshape(n_asu, n_points, 3)
    
    # Initialize output tensor
    ravel = torch.zeros((n_asu, n_points), dtype=torch.long, device=hkl_grid_sym.device)
    
    # Implement ravel_multi_index equivalent
    for i in range(n_asu):
        # Shift indices to start from 0
        shifted = hkl_grid_int[i] - lbounds
        
        # Calculate raveled indices
        # Formula: index = x * (dim_y * dim_z) + y * dim_z + z
        strides = torch.tensor([map_shape_ravel[1] * map_shape_ravel[2], 
                               map_shape_ravel[2], 
                               1], device=hkl_grid_sym.device)
        
        ravel[i] = torch.sum(shifted * strides, dim=1)
    
    return ravel, map_shape_ravel

def cos_sq(angles: torch.Tensor) -> torch.Tensor:
    """
    Compute cosine squared of input angles in radians.
    
    Args:
        angles: PyTorch tensor with angles in radians
        
    Returns:
        PyTorch tensor with cos^2 of angles
    """
    return torch.square(torch.cos(angles))

def sin_sq(angles: torch.Tensor) -> torch.Tensor:
    """
    Compute sine squared of input angles in radians.
    
    Args:
        angles: PyTorch tensor with angles in radians
        
    Returns:
        PyTorch tensor with sin^2 of angles
    """
    return torch.square(torch.sin(angles))

def compute_resolution(cell: torch.Tensor, hkl: torch.Tensor) -> torch.Tensor:
    """
    Compute reflections' resolution in 1/Angstrom using PyTorch.
    
    Args:
        cell: PyTorch tensor of shape (6,) with unit cell parameters
        hkl: PyTorch tensor of shape (n_refl, 3) with Miller indices
        
    Returns:
        PyTorch tensor of shape (n_refl,) with resolution in Angstrom
        
    References:
        - Original implementation: eryx/map_utils.py:compute_resolution
        
    This function works with both grid-based and arbitrary vector lists.
    hkl can be any tensor of shape (..., 3) where the last dimension
    contains the Miller indices.
    """
    # Extract cell parameters
    a, b, c = cell[0], cell[1], cell[2]
    alpha, beta, gamma = torch.deg2rad(cell[3]), torch.deg2rad(cell[4]), torch.deg2rad(cell[5])
    
    # Extract Miller indices
    h, k, l = hkl[:, 0], hkl[:, 1], hkl[:, 2]
    
    # Calculate terms for the formula
    pf = 1.0 - cos_sq(alpha) - cos_sq(beta) - cos_sq(gamma) + 2.0 * torch.cos(alpha) * torch.cos(beta) * torch.cos(gamma)
    
    n1 = torch.square(h) * sin_sq(alpha) / torch.square(a) + \
         torch.square(k) * sin_sq(beta) / torch.square(b) + \
         torch.square(l) * sin_sq(gamma) / torch.square(c)
    
    n2a = 2.0 * k * l * (torch.cos(beta) * torch.cos(gamma) - torch.cos(alpha)) / (b * c)
    n2b = 2.0 * l * h * (torch.cos(gamma) * torch.cos(alpha) - torch.cos(beta)) / (c * a)
    n2c = 2.0 * h * k * (torch.cos(alpha) * torch.cos(beta) - torch.cos(gamma)) / (a * b)
    
    # Calculate resolution with safe division
    denominator = (n1 + n2a + n2b + n2c) / pf
    
    # Handle potential divide by zero
    safe_denominator = torch.where(denominator > 0, denominator, torch.ones_like(denominator) * 1e-10)
    resolution = 1.0 / torch.sqrt(safe_denominator)
    
    # Set resolution to infinity where denominator is zero or negative
    resolution = torch.where(denominator > 0, resolution, float('inf') * torch.ones_like(resolution))
    
    return resolution

def get_resolution_mask(cell: torch.Tensor, hkl_grid: torch.Tensor, 
                       res_limit: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a boolean mask for resolution limits.
    
    Args:
        cell: PyTorch tensor of shape (6,) with cell parameters
        hkl_grid: PyTorch tensor of shape (n_points, 3) with hkl indices
        res_limit: High resolution limit in Angstrom
        
    Returns:
        Tuple containing:
            - PyTorch tensor of shape (n_points,) with boolean mask
            - PyTorch tensor of shape (n_points,) with resolution map
            
    References:
        - Original implementation: eryx/map_utils.py:get_resolution_mask
    """
    # Compute resolution map for each grid point
    res_map = compute_resolution(cell, hkl_grid)
    
    # Create boolean mask by comparing to resolution limit
    # Points with resolution > res_limit are kept (True)
    res_mask = res_map > res_limit
    
    return res_mask, res_map

def get_dq_map(A_inv: torch.Tensor, hkl_grid: torch.Tensor) -> torch.Tensor:
    """
    Compute distance to nearest Bragg peak for grid points.
    
    Args:
        A_inv: PyTorch tensor of shape (3, 3) with cell orthogonalization matrix
        hkl_grid: PyTorch tensor of shape (n_points, 3) with hkl indices
        
    Returns:
        PyTorch tensor of shape (n_points,) with distances
        
    References:
        - Original implementation: eryx/map_utils.py:get_dq_map
    """
    # Find closest integral hkl points using torch.round
    hkl_closest = torch.round(hkl_grid)
    
    # Convert to q-vectors
    q_closest = 2 * torch.pi * torch.matmul(A_inv.T, hkl_closest.T).T
    q_grid = 2 * torch.pi * torch.matmul(A_inv.T, hkl_grid.T).T
    
    # Compute distances using torch.norm
    dq = torch.norm(torch.abs(q_closest - q_grid), dim=1)
    
    # Round to specified precision (8 decimal places)
    # PyTorch doesn't have a direct equivalent to np.around with decimals
    # We can multiply by 10^8, round, then divide by 10^8
    scale = 1e8
    dq = torch.round(dq * scale) / scale
    
    return dq

def get_centered_sampling(map_shape: Tuple[int, int, int], 
                         sampling: Tuple[float, float, float]) -> List[Tuple[float, float, float]]:
    """
    Get sampling tuples for map centered about the origin.
    
    Args:
        map_shape: Tuple with map dimensions
        sampling: Tuple with fractional sampling rates
        
    Returns:
        List of sampling tuples for h, k, l dimensions
        
    References:
        - Original implementation: eryx/map_utils.py:get_centered_sampling
    """
    # Calculate extent for each dimension
    # This is a pure calculation that doesn't need tensors
    extents = [((map_shape[i] - 1) / sampling[i] / 2.0) for i in range(3)]
    
    # Create tuples for min, max, sampling rate
    return [(-extents[i], extents[i], sampling[i]) for i in range(3)]

def resize_map(new_map: torch.Tensor, 
              old_sampling: List[Tuple[float, float, float]], 
              new_sampling: List[Tuple[float, float, float]]) -> torch.Tensor:
    """
    Resize map if symmetrization changed dimensions.
    
    Args:
        new_map: PyTorch tensor of shape (dim_h, dim_k, dim_l) with map data
        old_sampling: List of (min, max, rate) tuples for original grid
        new_sampling: List of (min, max, rate) tuples for new map
        
    Returns:
        PyTorch tensor with resized map
        
    References:
        - Original implementation: eryx/map_utils.py:resize_map
    """
    # Define tolerance
    tol = 1e-6
    
    # Check sampling differences and crop if needed
    resized_map = new_map
    
    # Check and crop h dimension
    if abs(new_sampling[0][1] - old_sampling[0][1]) > tol:
        excise = int(torch.round(torch.tensor(2 * (new_sampling[0][1] - old_sampling[0][1]))))
        resized_map = resized_map[excise:-excise, :, :]
    
    # Check and crop k dimension
    if abs(new_sampling[1][1] - old_sampling[1][1]) > tol:
        excise = int(torch.round(torch.tensor(2 * (new_sampling[1][1] - old_sampling[1][1]))))
        resized_map = resized_map[:, excise:-excise, :]
    
    # Check and crop l dimension
    if abs(new_sampling[2][1] - old_sampling[2][1]) > tol:
        excise = int(torch.round(torch.tensor(2 * (new_sampling[2][1] - old_sampling[2][1]))))
        resized_map = resized_map[:, :, excise:-excise]
    
    return resized_map
