"""
PyTorch implementation of PDB handling and Gaussian Network Model.

This module contains PyTorch versions of the PDB handling and GNM classes defined in
eryx/pdb.py. All implementations maintain the same API as the NumPy versions
but use PyTorch tensors and operations to enable gradient flow.

References:
    - Original NumPy implementation in eryx/pdb.py
"""

import torch
import numpy as np
from typing import Optional, Tuple, List, Dict, Union, Any

class GaussianNetworkModel:
    """
    PyTorch implementation of the Gaussian Network Model.
    
    This class provides methods for computing the Hessian matrix, dynamical matrix K,
    and its inverse for elastic network models using PyTorch tensors.
    
    References:
        - Original NumPy implementation in eryx/pdb.py:GaussianNetworkModel
    """
    
    def compute_hessian(self) -> torch.Tensor:
        """
        For a pair of atoms the Hessian in a GNM is defined as:
        1. i not j and dij =< cutoff: -gamma_ij
        2. i not j and dij > cutoff: 0
        3. i=j: -sum_{j not i} hessian_ij
        
        Returns:
            hessian: torch.Tensor of shape (n_asu, n_atoms_per_asu, n_cell, n_asu, n_atoms_per_asu)
                with dtype torch.complex64
        """
        hessian = torch.zeros((self.n_asu, self.n_atoms_per_asu,
                              self.n_cell, self.n_asu, self.n_atoms_per_asu),
                             dtype=torch.complex64, device=self.device)
        hessian_diagonal = torch.zeros((self.n_asu, self.n_atoms_per_asu),
                                      dtype=torch.complex64, device=self.device)

        # off-diagonal
        for i_asu in range(self.n_asu):
            for i_cell in range(self.n_cell):
                for j_asu in range(self.n_asu):
                    for i_at in range(self.n_atoms_per_asu):
                        iat_neighbors = self.asu_neighbors[i_asu][i_cell][j_asu][i_at]
                        if len(iat_neighbors) > 0:
                            gamma = self.gamma[i_cell, i_asu, j_asu].to(torch.complex64)
                            for j_at in iat_neighbors:
                                hessian[i_asu, i_at, i_cell, j_asu, j_at] = -gamma
                            hessian_diagonal[i_asu, i_at] -= gamma * len(iat_neighbors)

        # diagonal (also correct for over-counted self term)
        for i_asu in range(self.n_asu):
            for i_at in range(self.n_atoms_per_asu):
                gamma_self = self.gamma[self.id_cell_ref, i_asu, i_asu].to(torch.complex64)
                hessian[i_asu, i_at, self.id_cell_ref, i_asu, i_at] = -hessian_diagonal[i_asu, i_at] - gamma_self

        return hessian

    def compute_K(self, hessian: torch.Tensor, kvec_batch: torch.Tensor) -> torch.Tensor:
        """
        Compute K matrices for a batch of k-vectors.
        
        Args:
            hessian: Hessian tensor from compute_hessian() with shape [n_asu, n_atoms_per_asu, n_cell, n_asu, n_atoms_per_asu]
            kvec_batch: Batch of k-vectors with shape [batch_size, 3]
            
        Returns:
            Kmat_batch: Batch of K matrices with shape [batch_size, n_asu, n_atoms_per_asu, n_asu, n_atoms_per_asu]
            
        Note:
            This method efficiently processes multiple k-vectors at once.
        """
        # Get batch size
        batch_size = kvec_batch.shape[0]
        
        # Start with reference cell contributions (which don't depend on k-vector)
        # Create a proper copy by using repeat instead of expand to avoid overlapping memory
        ref_cell_hessian = hessian[:, :, self.id_cell_ref, :, :].clone()
        Kmat_batch = ref_cell_hessian.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)
        
        # Handle complex data type for batch
        if hessian.dtype != torch.complex64:
            Kmat_batch = Kmat_batch.to(torch.complex64)
        
        # For each non-reference cell:
        for j_cell in range(self.n_cell):
            if j_cell == self.id_cell_ref:
                continue
                
            # Get unit cell origin coordinates
            r_cell = self.crystal.get_unitcell_origin(self.crystal.id_to_hkl(j_cell))
            
            # Calculate phase for all k-vectors in batch: batch_phases has shape [batch_size]
            # Sum along last dimension (3) to get dot product of each k-vector with r_cell
            batch_phases = torch.sum(kvec_batch * r_cell, dim=1)
            
            # Use ComplexTensorOps from torch_utils or compute complex exponentials directly
            try:
                from eryx.torch_utils import ComplexTensorOps
                real_part, imag_part = ComplexTensorOps.complex_exp(batch_phases)
            except ImportError:
                # Fallback to direct computation
                real_part = torch.cos(batch_phases)
                imag_part = torch.sin(batch_phases)
            
            # Create complex exponentials for all phases: eikr_batch has shape [batch_size]
            eikr_batch = torch.complex(real_part, imag_part)
            
            # Add contribution from this cell for all k-vectors
            # Need to expand hessian block and phase terms to align dimensions
            for i_asu in range(self.n_asu):
                for j_asu in range(self.n_asu):
                    # Get hessian block for this cell and ASU combo
                    block = hessian[i_asu, :, j_cell, j_asu, :]
                    
                    # Apply phase factors to hessian blocks and accumulate
                    # Need to reshape eikr_batch to allow broadcasting
                    # [batch_size] -> [batch_size, 1, 1] for proper broadcasting
                    eikr_reshaped = eikr_batch.view(batch_size, 1, 1)
                    
                    # Update the batch with this contribution
                    Kmat_batch[:, i_asu, :, j_asu, :] += block * eikr_reshaped
        
        return Kmat_batch

    def _batched_pinv(self, batch_matrices: torch.Tensor, rcond: float = 1e-10) -> torch.Tensor:
        """
        Compute the pseudo-inverse for a batch of matrices using SVD.
        
        Args:
            batch_matrices: Tensor of shape [batch_size, n, m] containing matrices to invert
            rcond: Cutoff for small singular values (relative to largest singular value)
            
        Returns:
            Tensor of shape [batch_size, m, n] containing pseudo-inverses
        """
        # Get batch dimensions and matrix shape
        batch_size, n, m = batch_matrices.shape
        
        # Compute batched SVD
        U, S, Vh = torch.linalg.svd(batch_matrices, full_matrices=False)
        
        # Apply rcond cutoff relative to largest singular value in each matrix
        S_max, _ = torch.max(S, dim=1, keepdim=True)
        S_cutoff = rcond * S_max
        S_valid = S > S_cutoff.expand_as(S)
        
        # Create batched reciprocal of singular values with zeros for invalid entries
        S_pinv = torch.zeros_like(S)
        S_pinv[S_valid] = 1.0 / S[S_valid]
        
        # Create diagonal matrices from singular values for matrix multiplication
        min_dim = min(m, n)
        
        # Process entire batch at once using vectorized operations
        # Create a batch of diagonal matrices using the reciprocal singular values
        S_pinv_values = S_pinv[:, :min_dim].to(dtype=batch_matrices.dtype)
        
        # Create empty matrices for the batch
        S_pinv_matrix = torch.zeros(batch_size, m, n, device=self.device, dtype=batch_matrices.dtype)
        
        # Create indices for batch diagonal assignment
        batch_indices = torch.arange(batch_size, device=self.device).repeat_interleave(min_dim)
        row_indices = torch.arange(min_dim, device=self.device).repeat(batch_size)
        col_indices = row_indices.clone()
        
        # Flatten S_pinv_values for assignment
        flat_values = S_pinv_values.reshape(-1)
        
        # Set all diagonal elements at once
        S_pinv_matrix[batch_indices, row_indices, col_indices] = flat_values
        
        # Compute the pseudo-inverse using U, S_pinv, and Vh
        # pinv(A) = V * S_pinv * U^H
        return torch.bmm(Vh.transpose(1, 2).conj(), torch.bmm(S_pinv_matrix, U.transpose(1, 2).conj()))
    
    def compute_Kinv(self, hessian: torch.Tensor, kvec_batch: torch.Tensor, 
                    reshape: bool = True) -> torch.Tensor:
        """
        Compute the inverse of K(kvec) for a batch of k-vectors.
        
        Args:
            hessian: Hessian tensor from compute_hessian() with shape 
                   [n_asu, n_atoms_per_asu, n_cell, n_asu, n_atoms_per_asu]
            kvec_batch: Batch of k-vectors with shape [batch_size, 3]
            reshape: Whether to reshape the output to match input dimensions
            
        Returns:
            Kinv_batch: Batch of inverse K matrices with shape
                    [batch_size, n_asu*n_atoms_per_asu, n_asu*n_atoms_per_asu] if reshape=False,
                    or [batch_size, n_asu, n_atoms_per_asu, n_asu, n_atoms_per_asu] if reshape=True
            
        Note:
            This method efficiently processes multiple k-vectors at once using batched operations.
        """
        # Compute K matrices for the batch
        Kmat_batch = self.compute_K(hessian, kvec_batch)
        
        # Get shape information
        batch_size = kvec_batch.shape[0]
        
        # Get other dimensions from the Kmat shape
        n_asu = Kmat_batch.shape[1]
        n_atoms = Kmat_batch.shape[2]
        
        # Reshape each K matrix to 2D for pseudo-inverse computation
        # From [batch_size, n_asu, n_atoms, n_asu, n_atoms] 
        # to [batch_size, n_asu*n_atoms, n_asu*n_atoms]
        Kmat_batch_2d = Kmat_batch.reshape(batch_size, n_asu * n_atoms, n_asu * n_atoms)
        
        # Apply regularization for numerical stability
        eps = 1e-10
        identity = torch.eye(
            Kmat_batch_2d.shape[1], 
            device=Kmat_batch_2d.device, 
            dtype=Kmat_batch_2d.dtype
        ).unsqueeze(0).expand(batch_size, -1, -1)
        
        Kmat_batch_2d_reg = Kmat_batch_2d + eps * identity
        
        # Compute pseudo-inverse using batched operation
        try:
            Kinv_batch = self._batched_pinv(Kmat_batch_2d_reg, rcond=eps)
        except RuntimeError as e:
            # Fallback to direct pinv if batched version fails
            print(f"WARNING: Batched pseudo-inverse failed with error: {e}")
            print(f"Falling back to direct pinv for all matrices at once.")
            
            # Log more diagnostic information
            print(f"Matrix shape: {Kmat_batch_2d_reg.shape}, dtype: {Kmat_batch_2d_reg.dtype}")
            
            # Process all matrices at once using direct pinv
            try:
                # Process all matrices at once with direct pinv
                Kinv_batch = torch.linalg.pinv(Kmat_batch_2d_reg, rcond=eps)
            except Exception as direct_e:
                print(f"WARNING: Direct pinv failed: {direct_e}")
                print("Using identity matrices as last resort")
                # Last resort fallback - use identity matrices
                Kinv_batch = torch.eye(
                    Kmat_batch_2d_reg.shape[1],
                    device=Kmat_batch_2d_reg.device,
                    dtype=Kmat_batch_2d_reg.dtype
                ).unsqueeze(0).expand(batch_size, -1, -1)
        
        # Reshape if requested
        if reshape:
            # From [batch_size, n_asu*n_atoms, n_asu*n_atoms] 
            # to [batch_size, n_asu, n_atoms, n_asu, n_atoms]
            Kinv_batch = Kinv_batch.reshape(batch_size, n_asu, n_atoms, n_asu, n_atoms)
        
        return Kinv_batch

def sym_str_as_matrix(sym_str: str) -> torch.Tensor:
    """
    Convert a symmetry operation from string to matrix format,
    only retaining the rotational component.
    
    Args:
        sym_str: Symmetry operation in string format, e.g. '-Y,X-Y,Z+1/3'
    
    Returns:
        sym_matrix: Rotation portion of symmetry operation in matrix format
    """
    sym_matrix = torch.zeros((3, 3))
    for i, item in enumerate(sym_str.split(",")):
        if '-X' in item:
            sym_matrix[i, 0] = -1
        if 'X' in item and '-X' not in item:
            sym_matrix[i, 0] = 1
        if 'Y' in item and '-Y' not in item:
            sym_matrix[i, 1] = 1
        if '-Y' in item:
            sym_matrix[i, 1] = -1
        if 'Z' in item and '-Z' not in item:
            sym_matrix[i, 2] = 1
        if '-Z' in item:
            sym_matrix[i, 2] = -1
    return sym_matrix

class AtomicModel:
    """
    PyTorch implementation of the AtomicModel class.
    
    This class handles atomic coordinates, form factors, and related data
    using PyTorch tensors for gradient-based calculations.
    """
    
    def _get_xyz_asus(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        Apply symmetry operations to get xyz coordinates of all 
        asymmetric units and pack them into the unit cell.
        
        Args:
            xyz: Atomic coordinates for a single asymmetric unit of shape (n_atoms, 3)
            
        Returns:
            xyz_asus: Atomic coordinates for all ASUs in the unit cell of shape (n_asu, n_atoms, 3)
        """
        xyz_asus = []
        for i, op in self.transformations.items():
            rot, trans = op[:3, :3], op[:, 3]
            xyz_asu = torch.matmul(rot, xyz.T).T + trans
            com = torch.mean(xyz_asu.T, dim=1)
            shift = torch.sum(torch.matmul(self.unit_cell_axes.T, self.sym_ops[i]), dim=1)
            xyz_asu += torch.abs(shift) * (com < 0).float()
            xyz_asu -= torch.abs(shift) * (com > self.cell[:3]).float()
            xyz_asus.append(xyz_asu)
        
        return torch.stack(xyz_asus)

    def flatten_model(self):
        """
        Set self variables to correspond to the given frame,
        for instance flattening the atomic coordinates from 
        (n_frames, n_atoms, 3) to (n_atoms, 3).
        """
        n_asu = self.xyz.shape[0]
        self.xyz = self.xyz.reshape(-1, self.xyz.shape[-1])
        self.ff_a = self.ff_a.reshape(-1, self.ff_a.shape[-1])
        self.ff_b = self.ff_b.reshape(-1, self.ff_b.shape[-1])
        self.ff_c = self.ff_c.flatten()
        self.elements = [item for sublist in self.elements for item in sublist]

class Crystal:
    """
    PyTorch implementation of the Crystal class.
    
    This class handles crystal structure and symmetry operations
    using PyTorch tensors for gradient-based calculations.
    """
    
    def get_asu_xyz(self, asu_id: int = 0, unit_cell: Optional[List[int]] = None) -> torch.Tensor:
        """
        Get atomic coordinates for a specific asymmetric unit in a specific unit cell.
        
        Args:
            asu_id: Asymmetric unit index
            unit_cell: Index of the unit cell along the 3 dimensions
            
        Returns:
            xyz: Atomic coordinates for this ASU in given unit cell
        """
        if unit_cell is None:
            unit_cell = [0, 0, 0]
        xyz = self.model._get_xyz_asus(self.model.xyz[0])[asu_id]  # get asu
        xyz += self.get_unitcell_origin(unit_cell)  # move to unit cell
        return xyz
