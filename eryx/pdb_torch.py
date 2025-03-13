"""
PyTorch implementation of PDB handling and Gaussian Network Model.

This module contains PyTorch versions of the PDB handling and GNM classes defined in
eryx/pdb.py. All implementations maintain the same API as the NumPy versions
but use PyTorch tensors and operations to enable gradient flow.

References:
    - Original NumPy implementation in eryx/pdb.py
"""

import torch
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

    def compute_K(self, hessian: torch.Tensor, kvec: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Noting H(d) the block of the hessian matrix
        corresponding the the d-th reference cell
        whose origin is located at r_d, then:
        K(kvec) = \sum_d H(d) exp(i kvec. r_d)
        
        Args:
            hessian: Hessian tensor from compute_hessian()
            kvec: Phonon wavevector, default zeros(3)
            
        Returns:
            Kmat: Dynamical matrix of shape (n_asu, n_atoms_per_asu, n_asu, n_atoms_per_asu)
                with dtype torch.complex64
        """
        if kvec is None:
            kvec = torch.zeros(3, device=self.device)
        Kmat = hessian[:, :, self.id_cell_ref, :, :].clone()

        for j_cell in range(self.n_cell):
            if j_cell == self.id_cell_ref:
                continue
            
            # Handle both dictionary-style and object-style crystal access
            if isinstance(self.crystal, dict):
                # Legacy dictionary-style access
                if 'get_unitcell_origin' in self.crystal and 'id_to_hkl' in self.crystal:
                    r_cell = self.crystal['get_unitcell_origin'](self.crystal['id_to_hkl'](j_cell))
                else:
                    # Fallback to zeros if methods not found
                    r_cell = torch.zeros(3, device=self.device)
            else:
                # New object-style access
                r_cell = self.crystal.get_unitcell_origin(self.crystal.id_to_hkl(j_cell))
                
            phase = torch.sum(kvec * r_cell)
            eikr = torch.complex(torch.cos(phase), torch.sin(phase))
            for i_asu in range(self.n_asu):
                for j_asu in range(self.n_asu):
                    Kmat[i_asu, :, j_asu, :] += hessian[i_asu, :, j_cell, j_asu, :] * eikr
        return Kmat

    def compute_Kinv(self, hessian: torch.Tensor, kvec: Optional[torch.Tensor] = None, 
                    reshape: bool = True) -> torch.Tensor:
        """
        Compute the inverse of K(kvec)
        (see compute_K() for the relationship between K and the hessian).
        
        Args:
            hessian: Hessian tensor from compute_hessian()
            kvec: Phonon wavevector, default zeros(3)
            reshape: Whether to reshape the output to match the input shape
            
        Returns:
            Kinv: Inverse of dynamical matrix K
        """
        if kvec is None:
            kvec = torch.zeros(3, device=self.device)
        Kmat = self.compute_K(hessian, kvec=kvec)
        Kshape = Kmat.shape
        Kmat_2d = Kmat.reshape(Kshape[0] * Kshape[1], Kshape[2] * Kshape[3])
        
        # Add small regularization for numerical stability
        eps = 1e-10
        identity = torch.eye(Kmat_2d.shape[0], device=self.device, dtype=Kmat_2d.dtype)
        Kmat_2d_reg = Kmat_2d + eps * identity
        
        Kinv = torch.linalg.pinv(Kmat_2d_reg)
        if reshape:
            Kinv = Kinv.reshape((Kshape[0], Kshape[1], Kshape[2], Kshape[3]))
        return Kinv

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
