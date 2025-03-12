"""
Mock data generators for testing PyTorch implementations.

This module provides utilities for generating mock data for testing
PyTorch implementations of diffuse scattering calculations.
"""

import torch
import numpy as np
from typing import Tuple, Optional, List, Dict, Any


def create_mock_kvectors(shape: Tuple[int, int, int, int] = (2, 2, 2, 3), 
                        device: Optional[torch.device] = None,
                        requires_grad: bool = True) -> torch.Tensor:
    """
    Create mock k-vectors tensor for testing.
    
    Args:
        shape: Shape of the k-vectors tensor (h_dim, k_dim, l_dim, 3)
        device: PyTorch device to place tensor on
        requires_grad: Whether tensor requires gradients
        
    Returns:
        PyTorch tensor with mock k-vectors
    """
    # Create tensor with systematic values
    h_dim, k_dim, l_dim, _ = shape
    
    # Create a grid of values that varies systematically
    kvectors = torch.zeros(shape, device=device)
    
    for h in range(h_dim):
        for k in range(k_dim):
            for l in range(l_dim):
                # Create values that depend on indices for testing gradient flow
                kvectors[h, k, l, 0] = (h + 1) * 0.1  # h component
                kvectors[h, k, l, 1] = (k + 1) * 0.1  # k component
                kvectors[h, k, l, 2] = (l + 1) * 0.1  # l component
    
    # Set requires_grad if needed
    if requires_grad:
        kvectors.requires_grad_(True)
        
    return kvectors


def create_mock_hessian(n_asu: int = 2, 
                       n_atoms: int = 3, 
                       n_cell: int = 8, 
                       device: Optional[torch.device] = None,
                       requires_grad: bool = True) -> torch.Tensor:
    """
    Create mock hessian tensor for testing.
    
    Args:
        n_asu: Number of asymmetric units
        n_atoms: Number of atoms per ASU
        n_cell: Number of unit cells
        device: PyTorch device to place tensor on
        requires_grad: Whether tensor requires gradients
        
    Returns:
        PyTorch tensor with mock hessian
    """
    # Create complex tensor with systematic values
    # Shape: (n_asu, n_atoms, n_cell, n_asu, n_atoms)
    hessian_shape = (n_asu, n_atoms, n_cell, n_asu, n_atoms)
    
    # Create real and imaginary parts separately
    real_part = torch.zeros(hessian_shape, device=device)
    imag_part = torch.zeros(hessian_shape, device=device)
    
    # Fill with systematic values
    for i_asu in range(n_asu):
        for i_atom in range(n_atoms):
            for i_cell in range(n_cell):
                for j_asu in range(n_asu):
                    for j_atom in range(n_atoms):
                        # Diagonal elements (i==j) are positive, off-diagonal are negative
                        if i_asu == j_asu and i_atom == j_atom and i_cell == 0:
                            real_part[i_asu, i_atom, i_cell, j_asu, j_atom] = 2.0
                        else:
                            # Create values that depend on indices
                            real_part[i_asu, i_atom, i_cell, j_asu, j_atom] = -0.1 * (
                                (i_asu + 1) * (i_atom + 1) * (j_asu + 1) * (j_atom + 1)
                            ) / (i_cell + 1)
                            
                        # Add some imaginary component for cells != 0
                        if i_cell != 0:
                            imag_part[i_asu, i_atom, i_cell, j_asu, j_atom] = 0.05 * i_cell
    
    # Combine real and imaginary parts
    hessian = torch.complex(real_part, imag_part)
    
    # Set requires_grad if needed
    if requires_grad:
        hessian.requires_grad_(True)
        
    return hessian


def create_mock_eigendecomposition(n_modes: int = 6, 
                                 n_nan: int = 2, 
                                 device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create mock eigendecomposition results (values and vectors) for testing.
    
    Args:
        n_modes: Number of modes (eigenvalues)
        n_nan: Number of NaN values to include
        device: PyTorch device to place tensors on
        
    Returns:
        Tuple of (eigenvalues, eigenvectors) as PyTorch tensors
    """
    # Create eigenvalues with some NaN values
    eigenvalues = torch.linspace(0.1, 1.0, n_modes, device=device)
    
    # Set some values to NaN
    if n_nan > 0:
        nan_indices = torch.randperm(n_modes)[:n_nan]
        eigenvalues[nan_indices] = float('nan')
    
    # Create corresponding eigenvectors (orthogonal)
    eigenvectors = torch.eye(n_modes, device=device)
    
    # Add some off-diagonal elements to make it more realistic
    noise = torch.randn(n_modes, n_modes, device=device) * 0.01
    eigenvectors = eigenvectors + noise
    
    # Ensure orthogonality by using QR decomposition
    eigenvectors, _ = torch.linalg.qr(eigenvectors)
    
    # Convert to complex for compatibility with some functions
    eigenvectors = torch.complex(eigenvectors, torch.zeros_like(eigenvectors))
    
    return eigenvalues, eigenvectors


def create_mock_model_dict(n_asu: int = 2, 
                          n_atoms: int = 4, 
                          device: Optional[torch.device] = None) -> Dict[str, Any]:
    """
    Create a mock model dictionary for testing.
    
    Args:
        n_asu: Number of asymmetric units
        n_atoms: Number of atoms per ASU
        device: PyTorch device to place tensors on
        
    Returns:
        Dictionary with mock model attributes
    """
    # Create mock model dictionary with key attributes
    model_dict = {}
    
    # Create xyz coordinates
    model_dict['xyz'] = [torch.rand((n_atoms, 3), device=device) for _ in range(n_asu)]
    
    # Create form factors
    model_dict['ff_a'] = [torch.rand((n_atoms, 4), device=device) for _ in range(n_asu)]
    model_dict['ff_b'] = [torch.rand((n_atoms, 4), device=device) for _ in range(n_asu)]
    model_dict['ff_c'] = [torch.rand((n_atoms), device=device) for _ in range(n_asu)]
    
    # Create ADPs
    model_dict['adp'] = [torch.ones(n_atoms, device=device)]
    
    # Create cell parameters
    model_dict['cell'] = torch.tensor([10.0, 10.0, 10.0, 90.0, 90.0, 90.0], device=device)
    
    # Create A_inv matrix
    model_dict['A_inv'] = torch.eye(3, device=device)
    
    return model_dict


def create_mock_crystal_dict(n_asu: int = 2, 
                            n_atoms: int = 4, 
                            n_cell: int = 8, 
                            device: Optional[torch.device] = None) -> Dict[str, Any]:
    """
    Create a mock crystal dictionary for testing.
    
    Args:
        n_asu: Number of asymmetric units
        n_atoms: Number of atoms per ASU
        n_cell: Number of unit cells
        device: PyTorch device to place tensors on
        
    Returns:
        Dictionary with mock crystal attributes
    """
    # Create mock crystal dictionary with key attributes
    crystal_dict = {}
    
    # Set dimensions
    crystal_dict['n_asu'] = n_asu
    crystal_dict['n_atoms_per_asu'] = n_atoms
    crystal_dict['n_cell'] = n_cell
    
    # Create mock methods
    def mock_hkl_to_id(hkl):
        # Simple mock implementation
        if isinstance(hkl, list) and len(hkl) == 3:
            return sum(hkl) % n_cell
        return 0
    
    def mock_id_to_hkl(cell_id):
        # Simple mock implementation
        return [cell_id % 3, (cell_id // 3) % 3, cell_id // 9]
    
    def mock_get_unitcell_origin(unit_cell=None):
        # Simple mock implementation
        if unit_cell is None:
            return torch.zeros(3, device=device)
        return torch.tensor([float(unit_cell[0]), float(unit_cell[1]), float(unit_cell[2])], 
                           device=device)
    
    def mock_get_asu_xyz(asu_id=0, unit_cell=None):
        # Simple mock implementation
        return torch.rand((n_atoms, 3), device=device) + asu_id
    
    # Add methods to dictionary
    crystal_dict['hkl_to_id'] = mock_hkl_to_id
    crystal_dict['id_to_hkl'] = mock_id_to_hkl
    crystal_dict['get_unitcell_origin'] = mock_get_unitcell_origin
    crystal_dict['get_asu_xyz'] = mock_get_asu_xyz
    
    return crystal_dict
