"""
PyTorch implementation of disorder models for diffuse scattering calculations.

This module contains PyTorch versions of the disorder models defined in eryx/models.py.
All implementations maintain the same API as the NumPy versions but use PyTorch tensors
and operations to enable gradient flow.

References:
    - Original NumPy implementation in eryx/models.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional, Union, Any

# Forward references for type hints
from eryx.pdb import AtomicModel, Crystal, GaussianNetworkModel

class OnePhonon:
    """
    PyTorch implementation of the OnePhonon model for diffuse scattering calculations.
    
    This class implements a lattice of interacting rigid bodies in the one-phonon
    approximation (a.k.a small-coupling regime) using PyTorch tensors and operations
    to enable gradient flow.
    
    References:
        - Original NumPy implementation in eryx/models.py:OnePhonon
    """
    
    def __init__(self, pdb_path: str, hsampling: Tuple[float, float, float], 
                 ksampling: Tuple[float, float, float], lsampling: Tuple[float, float, float],
                 expand_p1: bool = True, group_by: str = 'asu',
                 res_limit: float = 0., model: str = 'gnm',
                 gnm_cutoff: float = 4., gamma_intra: float = 1., gamma_inter: float = 1.,
                 batch_size: int = 10000, n_processes: int = 8):
        """
        Initialize the OnePhonon model with PyTorch tensors.
        
        Args:
            pdb_path: Path to coordinates file
            hsampling: (hmin, hmax, oversampling) for h dimension
            ksampling: (kmin, kmax, oversampling) for k dimension
            lsampling: (lmin, lmax, oversampling) for l dimension
            expand_p1: If True, expand to p1 (if PDB is asymmetric unit)
            group_by: Level of rigid-body assembly, 'asu' or None
            res_limit: High-resolution limit in Angstrom
            model: Chosen phonon model ('gnm' or 'rb')
            gnm_cutoff: Distance cutoff for GNM in Angstrom
            gamma_intra: Spring constant for atom pairs in same molecule
            gamma_inter: Spring constant for atom pairs in different molecules
            batch_size: Number of q-vectors to evaluate per batch
            n_processes: Number of processes for parallel computation
            
        References:
            - Original implementation: eryx/models.py:OnePhonon.__init__
        """
        # TODO: Initialize class attributes similar to the NumPy implementation
        # TODO: Convert sampling tuples to PyTorch compatible formats
        # TODO: Call self._setup() and self._setup_phonons() to initialize tensors
        
        self.hsampling = hsampling
        self.ksampling = ksampling
        self.lsampling = lsampling
        self.batch_size = batch_size
        self.n_processes = n_processes
        
        # These will be initialized in _setup() and _setup_phonons()
        self.model = None
        self.q_grid = None
        self.crystal = None
        self.res_mask = None
        self.group_by = group_by
        
        # Placeholder for a proper implementation
        raise NotImplementedError("OnePhonon.__init__ not implemented")
    
    def _setup(self, pdb_path: str, expand_p1: bool, res_limit: float, group_by: str):
        """
        Set up class, computing q-vectors and building the unit cell.
        
        Args:
            pdb_path: Path to coordinates file
            expand_p1: If True, expand to p1 (if PDB is asymmetric unit)
            res_limit: High-resolution limit in Angstrom
            group_by: Level of rigid-body assembly, 'asu' or None
            
        References:
            - Original implementation: eryx/models.py:OnePhonon._setup
        """
        # TODO: Create AtomicModel using adapter
        # TODO: Generate reciprocal space grid and convert to torch.Tensor
        # TODO: Calculate q vectors and q magnitudes as torch tensors
        # TODO: Set up Crystal object and compute necessary dimensions
        
        raise NotImplementedError("OnePhonon._setup not implemented")
    
    def _setup_phonons(self, pdb_path: str, model: str, 
                     gnm_cutoff: float, gamma_intra: float, gamma_inter: float):
        """
        Compute phonons from a Gaussian Network Model using PyTorch operations.
        
        Args:
            pdb_path: Path to coordinates file
            model: Chosen phonon model ('gnm' or 'rb')
            gnm_cutoff: Distance cutoff for GNM in Angstrom
            gamma_intra: Spring constant for atom pairs in same molecule
            gamma_inter: Spring constant for atom pairs in different molecules
            
        References:
            - Original implementation: eryx/models.py:OnePhonon._setup_phonons
        """
        # TODO: Initialize tensor arrays for phonon calculations
        # TODO: Build A and M matrices using PyTorch operations
        # TODO: Compute k-vectors in Brillouin zone as tensors
        # TODO: Setup GNM and compute phonon modes
        
        raise NotImplementedError("OnePhonon._setup_phonons not implemented")
    
    def _build_A(self):
        """
        Build the matrix A that projects small rigid-body displacements to individual atoms.
        
        This matrix converts from rigid-body displacements (translations and rotations)
        to individual atomic displacements based on the atom positions relative to the
        center of mass.
        
        For each atom i in group m, the conversion reads:
        u_i = A(r_i - o_m).w_m
        where A is a 3x6 matrix defined as:
        A(x,y,z) = [[ 1 0 0  0  z -y ]
                    [ 0 1 0 -z  0  x ]
                    [ 0 0 1  y -x  0 ]]
                    
        Returns:
            None - stores the matrix in self.Amat
            
        References:
            - Original implementation: eryx/models.py:OnePhonon._build_A
        """
        # Handle case where group_by is set to 'asu'
        if self.group_by == 'asu':
            # Initialize Amat tensor for all ASUs
            self.Amat = torch.zeros((self.n_asu, self.n_atoms_per_asu, 3, 6), 
                                   device=self.device)
            
            # Identity matrix for the first 3 columns of A
            identity = torch.eye(3, device=self.device)
            
            # For each ASU
            for i_asu in range(self.n_asu):
                # Get coordinates for this ASU
                xyz = self.crystal.get_asu_xyz(i_asu).clone()
                
                # Subtract center of mass
                xyz -= torch.mean(xyz, dim=0)
                
                # For each atom in the ASU
                for i_atom in range(self.n_atoms_per_asu):
                    # Create the skew-symmetric matrix for rotational part
                    # [  0  z -y ]
                    # [ -z  0  x ]
                    # [  y -x  0 ]
                    skew = torch.zeros((3, 3), device=self.device)
                    skew[0, 1] = xyz[i_atom, 2]     # z
                    skew[0, 2] = -xyz[i_atom, 1]    # -y
                    skew[1, 2] = xyz[i_atom, 0]     # x
                    skew = skew - skew.transpose(0, 1)  # Make skew-symmetric
                    
                    # Combine translational (identity) and rotational (skew) parts
                    self.Amat[i_asu, i_atom] = torch.cat([identity, skew], dim=1)
            
            # Reshape Amat to final dimensions
            self.Amat = self.Amat.reshape((self.n_asu,
                                          self.n_dof_per_asu_actual,
                                          self.n_dof_per_asu))
        else:
            self.Amat = None
    
    def _build_M(self):
        """
        Build the mass matrix M and compute its Cholesky decomposition.
        
        If all atoms are considered individually (group_by=None), M = M_0 is diagonal
        and Linv = 1/sqrt(M_0) is also diagonal.
        
        If atoms are grouped as rigid bodies, the all-atoms M matrix is projected
        using the A matrix: M = A.T M_0 A and Linv is obtained via Cholesky 
        decomposition: M = LL.T, Linv = L^(-1)
        
        Returns:
            None - stores the result in self.Linv
            
        References:
            - Original implementation: eryx/models.py:OnePhonon._build_M
        """
        # Get the all-atoms mass matrix
        M_allatoms = self._build_M_allatoms()
        
        # Handle different cases based on group_by parameter
        if self.group_by is None:
            # No grouping, reshape to 2D matrix
            M_allatoms = M_allatoms.reshape((self.n_asu * self.n_dof_per_asu_actual,
                                            self.n_asu * self.n_dof_per_asu_actual))
            
            # For diagonal mass matrix, Linv is simply 1/sqrt(M)
            # Add small epsilon for numerical stability
            epsilon = 1e-10
            self.Linv = 1.0 / torch.sqrt(M_allatoms + epsilon)
            
        else:
            # Project the mass matrix for rigid body case
            Mmat = self._project_M(M_allatoms)
            
            # Reshape to 2D matrix for Cholesky decomposition
            Mmat = Mmat.reshape((self.n_asu * self.n_dof_per_asu,
                                self.n_asu * self.n_dof_per_asu))
            
            # Add small regularization for numerical stability
            epsilon = 1e-10
            eye = torch.eye(Mmat.shape[0], device=self.device)
            Mmat = Mmat + epsilon * eye
            
            # Compute Cholesky decomposition: M = L*L^T
            try:
                L = torch.linalg.cholesky(Mmat)
                
                # Compute inverse of L
                self.Linv = torch.linalg.inv(L)
            except RuntimeError as e:
                # Handle case where matrix is not positive definite
                print(f"Warning: Cholesky decomposition failed: {e}")
                print("Using SVD-based approach instead")
                
                # Alternative approach using SVD
                U, S, Vh = torch.linalg.svd(Mmat)
                
                # Ensure S is positive
                S = torch.clamp(S, min=epsilon)
                
                # Compute L = U * sqrt(S)
                L = U * torch.sqrt(S).unsqueeze(0)
                
                # Compute inverse of L
                self.Linv = torch.matmul(torch.diag(1.0 / torch.sqrt(S)), U.transpose(0, 1))
    
    def _build_M_allatoms(self) -> torch.Tensor:
        """
        Build all-atom mass matrix M_0 from element weights.

        Returns:
            torch.Tensor: Mass matrix with shape (n_asu, n_atoms*3, n_asu, n_atoms*3)
            
        References:
            - Original implementation: eryx/models.py:OnePhonon._build_M_allatoms
        """
        # Extract atomic masses from the crystal model
        # Flatten the nested structure to get a single array of masses
        mass_array = torch.tensor([element.weight for structure in self.crystal.model.elements 
                                 for element in structure], device=self.device)
        
        # Create a 3x3 identity matrix
        eye3 = torch.eye(3, device=self.device)
        
        # Initialize a list to store block matrices
        mass_blocks = []
        
        # For each atom, create a 3x3 block with the atom's mass on the diagonal
        for i in range(self.n_asu * self.n_atoms_per_asu):
            # Create a 3x3 matrix with the atom's mass on the diagonal
            mass_block = mass_array[i] * eye3
            
            # Extract the rows for this atom
            # For each atom i, we create rows for coordinates 3*i, 3*i+1, 3*i+2
            start_row = 3 * i
            
            # For each row, add a block matrix
            for j in range(3):
                row_block = torch.zeros(3 * self.n_asu * self.n_atoms_per_asu, device=self.device)
                
                # Only populate the 3 elements corresponding to this atom
                row_block[start_row:start_row + 3] = mass_block[j]
                
                mass_blocks.append(row_block)
        
        # Stack all blocks into a matrix
        M_allatoms = torch.stack(mass_blocks)
        
        # Reshape to the required 4D shape
        M_allatoms = M_allatoms.reshape((self.n_asu, self.n_dof_per_asu_actual,
                                        self.n_asu, self.n_dof_per_asu_actual))
        
        return M_allatoms
    
    def _project_M(self, M_allatoms: torch.Tensor) -> torch.Tensor:
        """
        Project all-atom mass matrix using the A matrix: M = A.T M_0 A

        Parameters:
            M_allatoms: torch.Tensor - All-atom mass matrix with shape 
                        (n_asu, n_atoms*3, n_asu, n_atoms*3)

        Returns:
            torch.Tensor: Projected mass matrix with shape 
                        (n_asu, n_dof_per_asu, n_asu, n_dof_per_asu)
            
        References:
            - Original implementation: eryx/models.py:OnePhonon._project_M
        """
        # Initialize projected mass matrix with zeros
        Mmat = torch.zeros((self.n_asu, self.n_dof_per_asu,
                          self.n_asu, self.n_dof_per_asu), 
                          device=self.device)
        
        # For each pair of ASUs
        for i_asu in range(self.n_asu):
            for j_asu in range(self.n_asu):
                # Perform the projection: A^T * M * A
                # First multiply M_allatoms by A on the right
                # M_allatoms[i_asu, :, j_asu, :] has shape (n_dof_per_asu_actual, n_dof_per_asu_actual)
                # self.Amat[j_asu] has shape (n_dof_per_asu_actual, n_dof_per_asu)
                intermediate = torch.matmul(M_allatoms[i_asu, :, j_asu, :],
                                          self.Amat[j_asu])
                
                # Then multiply by A^T on the left
                # self.Amat[i_asu].T has shape (n_dof_per_asu, n_dof_per_asu_actual)
                # intermediate has shape (n_dof_per_asu_actual, n_dof_per_asu)
                Mmat[i_asu, :, j_asu, :] = torch.matmul(self.Amat[i_asu].transpose(0, 1),
                                                      intermediate)
        
        return Mmat
    
    def _build_kvec_Brillouin(self):
        """
        Compute all k-vectors and their norm in the first Brillouin zone.
        
        This is achieved by regularly sampling [-0.5,0.5[ for h, k and l,
        computing the corresponding vectors in reciprocal space, and storing
        their norms.
        
        References:
            - Original implementation: eryx/models.py:OnePhonon._build_kvec_Brillouin
        """
        # Get dimensions
        h_dim = self.hsampling[2]
        k_dim = self.ksampling[2]
        l_dim = self.lsampling[2]
        
        # Create centered coordinates for h, k, l
        h_vals = torch.tensor([self._center_kvec(dh, h_dim) for dh in range(h_dim)], device=self.device)
        k_vals = torch.tensor([self._center_kvec(dk, k_dim) for dk in range(k_dim)], device=self.device)
        l_vals = torch.tensor([self._center_kvec(dl, l_dim) for dl in range(l_dim)], device=self.device)
        
        # Create meshgrid
        h_grid, k_grid, l_grid = torch.meshgrid(h_vals, k_vals, l_vals, indexing='ij')
        
        # Stack to create k-vectors
        k_vecs = torch.stack([h_grid, k_grid, l_grid], dim=-1)
        
        # Reshape for matrix multiplication
        k_vecs_flat = k_vecs.reshape(-1, 3)
        
        # Compute 2π * A_inv^T * k for all k-vectors at once
        q_vecs_flat = 2 * torch.pi * torch.matmul(self.A_inv.T, k_vecs_flat.T).T
        
        # Reshape back to grid
        self.kvec = q_vecs_flat.reshape(h_dim, k_dim, l_dim, 3)
        
        # Compute norms
        self.kvec_norm = torch.norm(self.kvec, dim=-1, keepdim=True)
    
    def _center_kvec(self, x: int, L: int) -> float:
        """
        Center k-vector components.
        
        For x and L integers such that 0 < x < L, return -L/2 < x < L/2
        by applying periodic boundary condition in L/2
        
        Args:
            x: Index to center
            L: Length of the periodic box
            
        Returns:
            float: Centered k-vector component
            
        References:
            - Original implementation: eryx/models.py:OnePhonon._center_kvec
        """
        # This function is essentially identical to the NumPy implementation
        # as it's a simple calculation not requiring tensor operations
        return int(((x - L / 2) % L) - L / 2) / L
    
    def _at_kvec_from_miller_points(self, hkl_kvec: tuple):
        """
        Return the indices of all q-vector that are k-vector away from any
        Miller index in the map.
        
        Args:
            hkl_kvec: Tuple of ints with fractional Miller index of the desired k-vector
            
        Returns:
            torch.Tensor: Indices of q-vectors in raveled form
            
        References:
            - Original implementation: eryx/models.py:OnePhonon._at_kvec_from_miller_points
        """
        # Calculate steps for each dimension
        hsteps = int(self.hsampling[2] * (self.hsampling[1] - self.hsampling[0]) + 1)
        ksteps = int(self.ksampling[2] * (self.ksampling[1] - self.ksampling[0]) + 1)
        lsteps = int(self.lsampling[2] * (self.lsampling[1] - self.lsampling[0]) + 1)
        
        # Create meshgrid equivalent to NumPy's mgrid
        h_indices = torch.arange(hkl_kvec[0], hsteps, self.hsampling[2], device=self.device, dtype=torch.long)
        k_indices = torch.arange(hkl_kvec[1], ksteps, self.ksampling[2], device=self.device, dtype=torch.long)
        l_indices = torch.arange(hkl_kvec[2], lsteps, self.lsampling[2], device=self.device, dtype=torch.long)
        
        # Create meshgrid
        h_grid, k_grid, l_grid = torch.meshgrid(h_indices, k_indices, l_indices, indexing='ij')
        
        # Flatten indices
        h_flat = h_grid.reshape(-1)
        k_flat = k_grid.reshape(-1)
        l_flat = l_grid.reshape(-1)
        
        # Create a PyTorch equivalent to np.ravel_multi_index
        # Formula: index = h * (dim_k * dim_l) + k * dim_l + l
        indices = (h_flat * (self.map_shape[1] * self.map_shape[2]) + 
                  k_flat * self.map_shape[2] + 
                  l_flat)
        
        return indices
    
    def compute_gnm_hessian(self) -> torch.Tensor:
        """
        For a pair of atoms the Hessian in a GNM is defined as:
        1. i not j and dij ≤ cutoff: -gamma_ij
        2. i not j and dij > cutoff: 0
        3. i=j: -sum_{j not i} hessian_ij
        
        This method replaces GaussianNetworkModel.compute_hessian in the NumPy implementation.
        
        Returns:
            torch.Tensor: Hessian matrix with shape (n_asu, n_atoms_per_asu,
                                                    n_cell, n_asu, n_atoms_per_asu)
                                                    
        References:
            - Original implementation: eryx/pdb.py:GaussianNetworkModel.compute_hessian
        """
        # Initialize Hessian tensor with complex dtype for later operations with phase factors
        hessian = torch.zeros((self.n_asu, self.n_atoms_per_asu,
                              self.n_cell, self.n_asu, self.n_atoms_per_asu),
                             dtype=torch.complex64, device=self.device)
        
        # Initialize diagonal tensor to accumulate values for diagonal elements
        hessian_diagonal = torch.zeros((self.n_asu, self.n_atoms_per_asu),
                                      dtype=torch.complex64, device=self.device)
        
        # Compute off-diagonal elements
        for i_asu in range(self.n_asu):
            for i_cell in range(self.n_cell):
                for j_asu in range(self.n_asu):
                    for i_at in range(self.n_atoms_per_asu):
                        # Get neighbors from asu_neighbors - this would be available in the full model
                        # Here we'll use a mock implementation for testing
                        iat_neighbors = self._get_atom_neighbors(i_asu, i_cell, j_asu, i_at)
                        
                        if len(iat_neighbors) > 0:
                            # Get appropriate gamma value
                            gamma = self._get_gamma(i_asu, i_cell, j_asu)
                            
                            # Set Hessian values for all neighbors
                            for j_at in iat_neighbors:
                                hessian[i_asu, i_at, i_cell, j_asu, j_at] = -gamma
                            
                            # Accumulate for diagonal elements
                            hessian_diagonal[i_asu, i_at] -= gamma * len(iat_neighbors)
        
        # Set diagonal elements (also correct for over-counted self term)
        for i_asu in range(self.n_asu):
            for i_at in range(self.n_atoms_per_asu):
                gamma_self = self._get_gamma(i_asu, self.id_cell_ref, i_asu)
                hessian[i_asu, i_at, self.id_cell_ref, i_asu, i_at] = hessian_diagonal[i_asu, i_at] - gamma_self
        
        return hessian
    
    # Helper methods for testing - these would be replaced in the full implementation
    def _get_atom_neighbors(self, i_asu, i_cell, j_asu, i_at):
        """Mock implementation to get atom neighbors for testing."""
        # Return empty list for now - will be replaced in tests with mock data
        return []
    
    def _get_gamma(self, i_asu, i_cell, j_asu):
        """Mock implementation to get gamma values for testing."""
        # Return default value for now - will be replaced in tests with mock data
        return torch.tensor(1.0, device=self.device, dtype=torch.complex64)
    
    def compute_gnm_K(self, hessian: torch.Tensor, kvec: torch.Tensor = None) -> torch.Tensor:
        """
        Noting H(d) the block of the hessian matrix corresponding the the d-th reference cell
        whose origin is located at r_d, then:
        K(kvec) = \sum_d H(d) exp(i kvec. r_d)
        
        This method replaces GaussianNetworkModel.compute_K in the NumPy implementation.
        
        Args:
            hessian: torch.Tensor - Hessian matrix from compute_gnm_hessian
            kvec: torch.Tensor - Phonon wavevector, default is zeros(3)
            
        Returns:
            torch.Tensor: Dynamical matrix K with shape (n_asu, n_atoms_per_asu,
                                                        n_asu, n_atoms_per_asu)
                                                        
        References:
            - Original implementation: eryx/pdb.py:GaussianNetworkModel.compute_K
        """
        # Default to zero vector if not provided
        if kvec is None:
            kvec = torch.zeros(3, device=self.device)
        
        # Initialize K matrix with the reference cell contribution
        Kmat = hessian[:, :, self.id_cell_ref, :, :].clone()
        
        # Add contributions from other cells with phase factors
        for j_cell in range(self.n_cell):
            if j_cell == self.id_cell_ref:
                continue
                
            # Get unit cell origin position
            r_cell = self.get_unitcell_origin(self.id_to_hkl(j_cell))
            
            # Compute phase factor e^(i k·r)
            phase = torch.dot(kvec, r_cell)
            
            # Use ComplexTensorOps if needed, or direct calculation
            eikr = torch.complex(torch.cos(phase), torch.sin(phase))
            
            # Add contribution with phase factor
            for i_asu in range(self.n_asu):
                for j_asu in range(self.n_asu):
                    Kmat[i_asu, :, j_asu, :] += hessian[i_asu, :, j_cell, j_asu, :] * eikr
        
        return Kmat
    
    def compute_gnm_Kinv(self, hessian: torch.Tensor, kvec: torch.Tensor = None, 
                         reshape: bool = True) -> torch.Tensor:
        """
        Compute the inverse of K(kvec) (see compute_gnm_K() for the relationship 
        between K and the hessian).
        
        This method replaces GaussianNetworkModel.compute_Kinv in the NumPy implementation.
        
        Args:
            hessian: torch.Tensor - Hessian matrix from compute_gnm_hessian
            kvec: torch.Tensor - Phonon wavevector, default is zeros(3)
            reshape: bool - If True, reshape the result to 4D tensor
            
        Returns:
            torch.Tensor: Inverse of dynamical matrix with appropriate shape
            
        References:
            - Original implementation: eryx/pdb.py:GaussianNetworkModel.compute_Kinv
        """
        # Default to zero vector if not provided
        if kvec is None:
            kvec = torch.zeros(3, device=self.device)
        
        # Compute K matrix
        Kmat = self.compute_gnm_K(hessian, kvec=kvec)
        Kshape = Kmat.shape
        
        # Reshape to 2D matrix for inversion
        Kmat_2d = Kmat.reshape(Kshape[0] * Kshape[1], Kshape[2] * Kshape[3])
        
        # Use torch.linalg.pinv for pseudo-inverse with gradient support
        # Add small regularization for numerical stability
        eps = 1e-10
        identity = torch.eye(Kmat_2d.shape[0], device=self.device, dtype=Kmat_2d.dtype)
        Kmat_2d_reg = Kmat_2d + eps * identity
        
        # Use EigenOps for more controllable pseudo-inverse with gradient support
        from eryx.torch_utils import EigenOps
        Kinv = EigenOps.solve_linear_system(Kmat_2d_reg, identity)
        
        # Reshape if requested
        if reshape:
            Kinv = Kinv.reshape(Kshape[0], Kshape[1], Kshape[2], Kshape[3])
        
        return Kinv
    
    def compute_hessian(self) -> torch.Tensor:
        """
        Build the projected Hessian matrix for the supercell.
        
        Returns:
            torch.Tensor: Hessian matrix with shape (n_asu, n_dof_per_asu,
                                                    n_cell, n_asu, n_dof_per_asu)
                                                    
        References:
            - Original implementation: eryx/models.py:OnePhonon.compute_hessian
        """
        # Initialize Hessian tensor with complex dtype for later operations
        hessian = torch.zeros((self.n_asu, self.n_dof_per_asu,
                              self.n_cell, self.n_asu, self.n_dof_per_asu),
                             dtype=torch.complex64, device=self.device)
        
        # Compute the all-atoms Hessian matrix using GNM method
        hessian_allatoms = self.compute_gnm_hessian()
        
        # Project using Amat (similar to _project_M method)
        for i_cell in range(self.n_cell):
            for i_asu in range(self.n_asu):
                for j_asu in range(self.n_asu):
                    # Create block diagonal matrix of the Hessian
                    hessian_block = torch.kron(
                        hessian_allatoms[i_asu, :, i_cell, j_asu, :],
                        torch.eye(3, device=self.device, dtype=torch.complex64)
                    )
                    
                    # Project using Amat: Amat.T @ hessian_block @ Amat
                    projected = torch.matmul(
                        self.Amat[i_asu].T.to(dtype=torch.complex64),
                        torch.matmul(
                            hessian_block,
                            self.Amat[j_asu].to(dtype=torch.complex64)
                        )
                    )
                    
                    # Store in output tensor
                    hessian[i_asu, :, i_cell, j_asu, :] = projected
        
        return hessian
    
    def compute_gnm_phonons(self):
        """
        Compute the dynamical matrix for each k-vector in the first Brillouin zone,
        from the supercell's GNM.
        
        The squared inverse of the eigenvalues is stored for intensity calculation,
        and the eigenvectors are mass-weighted to be used in the definition of the
        phonon structure factors.
        
        References:
            - Original implementation: eryx/models.py:OnePhonon.compute_gnm_phonons
        """
        # Import EigenOps for eigendecomposition with gradient support
        from eryx.torch_utils import EigenOps
        
        # Compute the Hessian matrix
        hessian = self.compute_hessian()
        
        # Initialize tensors for eigenvalues and eigenvectors if not already done
        if not hasattr(self, 'V') or self.V is None:
            self.V = torch.zeros((self.hsampling[2],
                                 self.ksampling[2],
                                 self.lsampling[2],
                                 self.n_asu * self.n_dof_per_asu,
                                 self.n_asu * self.n_dof_per_asu),
                                dtype=torch.complex64, device=self.device)
        
        if not hasattr(self, 'Winv') or self.Winv is None:
            self.Winv = torch.zeros((self.hsampling[2],
                                    self.ksampling[2],
                                    self.lsampling[2],
                                    self.n_asu * self.n_dof_per_asu),
                                   dtype=torch.complex64, device=self.device)
        
        # Process each k-vector in the Brillouin zone
        for dh in range(self.hsampling[2]):
            for dk in range(self.ksampling[2]):
                for dl in range(self.lsampling[2]):
                    # Extract current k-vector
                    kvec = self.kvec[dh, dk, dl]
                    
                    # Compute dynamical matrix for this k-vector
                    Kmat = self.compute_gnm_K(hessian, kvec=kvec)
                    
                    # Reshape to 2D matrix for eigendecomposition
                    Kmat_2d = Kmat.reshape(self.n_asu * self.n_dof_per_asu,
                                          self.n_asu * self.n_dof_per_asu)
                    
                    # Compute D = L⁻¹ K L⁻ᵀ (mass-weighted dynamical matrix)
                    # Convert Linv to complex for compatibility
                    Linv_complex = self.Linv.to(dtype=torch.complex64)
                    Dmat = torch.matmul(Linv_complex, 
                                       torch.matmul(Kmat_2d, Linv_complex.T))
                    
                    # Perform SVD-based eigendecomposition for better gradient support
                    v, s, _ = EigenOps.svd_decomposition(Dmat)
                    
                    # Post-process eigenvalues and eigenvectors
                    # Compute frequencies (sqrt of eigenvalues)
                    w = torch.sqrt(s)
                    
                    # Handle small/zero eigenvalues
                    eps = 1e-6
                    w_safe = torch.where(w < eps, float('nan'), w)
                    
                    # Reverse order to match NumPy implementation
                    w_safe = torch.flip(w_safe, [0])
                    v = torch.flip(v, [1])
                    
                    # Store inverse squared frequencies
                    self.Winv[dh, dk, dl] = 1.0 / (w_safe ** 2)
                    
                    # Store mass-weighted eigenvectors
                    self.V[dh, dk, dl] = torch.matmul(Linv_complex.T, v)
    
    def compute_covariance_matrix(self):
        """
        Compute covariance matrix for all asymmetric units with PyTorch operations.
        
        This method calculates the atomic displacement covariance matrix from phonon modes
        computed using the Gaussian Network Model. The covariance matrix is scaled to
        match the ADPs (Atomic Displacement Parameters) in the input PDB file.
        
        The method populates the following instance variables:
        - self.covar: Covariance matrix of shape (n_asu, n_dof_per_asu, n_cell, n_asu, n_dof_per_asu)
        - self.ADP: Atomic displacement parameters derived from the covariance matrix
        
        Tensor shapes and dimensions:
        - covar: (n_asu*n_dof_per_asu, n_cell, n_asu*n_dof_per_asu) initially,
                then reshaped to (n_asu, n_dof_per_asu, n_cell, n_asu, n_dof_per_asu)
        - kvec: (n_h, n_k, n_l, 3) - k-vectors in the Brillouin zone
        - hessian: (n_asu, n_dof_per_asu, n_cell, n_asu, n_dof_per_asu) - Hessian matrix
        
        Returns:
            None - results stored in instance variables
            
        References:
            - Original implementation: eryx/models.py:OnePhonon.compute_covariance_matrix
        """
        # Initialize covariance tensor with complex dtype
        self.covar = torch.zeros((self.n_asu*self.n_dof_per_asu,
                                self.n_cell, self.n_asu*self.n_dof_per_asu),
                               dtype=torch.complex64, device=self.device)
        
        # Import ComplexTensorOps if not already imported
        from eryx.torch_utils import ComplexTensorOps
        
        # Compute the Hessian matrix
        hessian = self.compute_hessian()
        
        # Loop through all k-vectors in the Brillouin zone
        for dh in range(self.hsampling[2]):
            for dk in range(self.ksampling[2]):
                for dl in range(self.lsampling[2]):
                    # Extract current k-vector
                    kvec = self.kvec[dh, dk, dl]
                    
                    # Compute inverse dynamical matrix for this k-vector
                    # Use compute_gnm_Kinv which supports gradients
                    Kinv = self.compute_gnm_Kinv(hessian, kvec=kvec, reshape=False)
                    
                    # Add contribution for each unit cell with phase factor
                    for j_cell in range(self.n_cell):
                        # Get unit cell origin position
                        r_cell = self.crystal.get_unitcell_origin(self.crystal.id_to_hkl(j_cell))
                        
                        # Calculate phase factor e^(i k·r)
                        phase = torch.dot(kvec, r_cell)
                        real_part, imag_part = ComplexTensorOps.complex_exp(phase)
                        eikr = torch.complex(real_part, imag_part)
                        
                        # Add contribution to covariance matrix
                        # Use addition to preserve gradient flow (not in-place += which can break gradients)
                        self.covar[:, j_cell, :] = self.covar[:, j_cell, :] + Kinv * eikr
        
        # Get reference cell ID
        id_cell_ref = self.crystal.hkl_to_id([0, 0, 0])
        
        # Extract ADPs from the diagonal of the reference cell covariance
        # Use torch.diagonal for gradient compatibility
        # For a tensor of shape (n_asu*n_dof_per_asu, n_cell, n_asu*n_dof_per_asu),
        # we need to get the diagonal between the first and last dimensions
        self.ADP = torch.real(torch.diagonal(self.covar[:, id_cell_ref, :], dim1=0, dim2=1))
        
        # Project ADPs using Amat
        # Transpose and reshape Amat for matrix multiplication
        Amat = torch.transpose(self.Amat, 0, 1).reshape(
            self.n_dof_per_asu_actual, self.n_asu*self.n_dof_per_asu)
        
        # Apply projection
        self.ADP = torch.matmul(Amat, self.ADP)
        
        # Sum over 3D components (x,y,z) for each atom
        self.ADP = torch.sum(self.ADP.reshape(int(self.ADP.shape[0]/3), 3), dim=1)
        
        # Calculate scaling factor to match experimental ADPs
        # Add small epsilon for numerical stability
        epsilon = 1e-10
        target_adp_mean = torch.mean(self.model.adp)
        current_adp_mean = torch.mean(self.ADP) / 3
        ADP_scale = target_adp_mean / (8*torch.pi*torch.pi*current_adp_mean + epsilon)
        
        # Apply scaling to ADP and covariance
        self.ADP = self.ADP * ADP_scale
        self.covar = self.covar * ADP_scale
        
        # Take real part and reshape to 5D tensor for the final output format
        self.covar = torch.real(self.covar.reshape(
            self.n_asu, self.n_dof_per_asu,
            self.n_cell, self.n_asu, self.n_dof_per_asu))
    
    def apply_disorder(self, rank: int = -1, outdir: Optional[str] = None, 
                     use_data_adp: bool = False) -> torch.Tensor:
        """
        Compute diffuse intensity in the one-phonon approximation using PyTorch.
        
        This method produces the diffuse intensity map by applying the phonon-based
        disorder model, combining structure factors with phonon modes and frequencies.
        
        Args:
            rank: Optional phonon mode selection. If -1 (default), use all modes.
                  Otherwise, use only the specified mode.
            outdir: Optional directory to save output files
            use_data_adp: Whether to use experimental ADPs from input data instead
                         of computed ADPs from the model
            
        Returns:
            PyTorch tensor of shape (n_points,) containing diffuse intensity values.
            Values outside the resolution mask are set to NaN.
            
        Note:
            This is the final computational step in the diffuse scattering calculation,
            combining all previous components:
            - Structure factors from scatter_torch.py
            - Phonon modes and frequencies from compute_gnm_phonons
            - K-vector selection using _at_kvec_from_miller_points
            
        References:
            - Original implementation: eryx/models.py:OnePhonon.apply_disorder
        """
        # Select appropriate ADPs based on flag
        if use_data_adp:
            ADP = self.model.adp[0] / (8 * torch.pi * torch.pi)
        else:
            ADP = self.ADP
            
        # Initialize diffuse intensity tensor with float dtype
        Id = torch.zeros(self.q_grid.shape[0], dtype=torch.float32, device=self.device)
        
        # Import structure_factors from scatter_torch
        from eryx.scatter_torch import structure_factors
        
        # Loop through all k-vectors in the Brillouin zone
        for dh in range(self.hsampling[2]):
            for dk in range(self.ksampling[2]):
                for dl in range(self.lsampling[2]):
                    # Get q-vector indices that are k-vector away from Miller indices
                    q_indices = self._at_kvec_from_miller_points((dh, dk, dl))
                    
                    # Apply resolution mask
                    mask = self.res_mask[q_indices]
                    valid_indices = q_indices[mask]
                    
                    # Skip if no valid points after masking
                    if valid_indices.shape[0] == 0:
                        continue
                        
                    # Initialize structure factor tensor for all ASUs
                    F = torch.zeros((valid_indices.shape[0], 
                                   self.n_asu, 
                                   self.n_dof_per_asu), 
                                  dtype=torch.complex64, 
                                  device=self.device)
                    
                    # Compute structure factors for each asymmetric unit
                    for i_asu in range(self.n_asu):
                        F[:, i_asu, :] = structure_factors(
                            self.q_grid[valid_indices],
                            self.model.xyz[i_asu],
                            self.model.ff_a[i_asu],
                            self.model.ff_b[i_asu],
                            self.model.ff_c[i_asu],
                            U=ADP,
                            batch_size=self.batch_size,
                            compute_qF=True,
                            project_on_components=self.Amat[i_asu],
                            sum_over_atoms=False
                        )
                    
                    # Reshape for matrix multiplication with eigenvectors
                    F = F.reshape((valid_indices.shape[0], self.n_asu * self.n_dof_per_asu))
                    
                    # Handle different rank modes
                    if rank == -1:
                        # Use all phonon modes (full calculation)
                        # Multiply structure factors by eigenvectors
                        FV = torch.matmul(F, self.V[dh, dk, dl])
                        
                        # Compute |F·V|² for all modes
                        FV_abs_squared = torch.abs(FV)**2
                        
                        # Weight by eigenvalues (Winv) and sum
                        # Extract real part of Winv to ensure type compatibility
                        weighted_intensity = torch.matmul(FV_abs_squared, torch.real(self.Winv[dh, dk, dl]))
                        
                        # Update diffuse intensity at valid indices
                        # Using index_add_ for better gradient support than direct indexing
                        Id.index_add_(0, valid_indices, weighted_intensity)
                    else:
                        # Use only the selected phonon mode (rank)
                        # Select the specific eigenvector
                        V_rank = self.V[dh, dk, dl, :, rank]
                        
                        # Multiply structure factors by the selected eigenvector
                        FV = torch.matmul(F, V_rank)
                        
                        # Compute |F·V|² and weight by the eigenvalue
                        # Extract real part of Winv to ensure type compatibility
                        weighted_intensity = torch.abs(FV)**2 * torch.real(self.Winv[dh, dk, dl, rank])
                        
                        # Update diffuse intensity at valid indices
                        Id.index_add_(0, valid_indices, weighted_intensity)
        
        # Apply resolution mask and take real part
        # Set values outside resolution mask to NaN
        Id_masked = torch.full_like(Id, float('nan'), dtype=torch.float32)
        Id_masked[self.res_mask] = torch.real(Id[self.res_mask])
        
        # Save output if directory is provided
        if outdir is not None:
            import os
            import numpy as np
            
            # Create output directory if it doesn't exist
            os.makedirs(outdir, exist_ok=True)
            
            # Save as both PyTorch tensor and NumPy array
            torch.save(Id_masked, os.path.join(outdir, f"rank_{rank:05d}_torch.pt"))
            np.save(os.path.join(outdir, f"rank_{rank:05d}.npy"), 
                   Id_masked.detach().cpu().numpy())
        
        return Id_masked

# Add stubs for additional classes as well:

class RigidBodyTranslations:
    """
    PyTorch implementation of rigid body translation disorder model.
    
    References:
        - Original NumPy implementation in eryx/models.py:RigidBodyTranslations
    """
    # TODO: Implement initialization and methods with PyTorch operations
    pass

class LiquidLikeMotions:
    """
    PyTorch implementation of liquid-like motions disorder model.
    
    References:
        - Original NumPy implementation in eryx/models.py:LiquidLikeMotions
    """
    # TODO: Implement initialization and methods with PyTorch operations
    pass

class RigidBodyRotations:
    """
    PyTorch implementation of rigid body rotations disorder model.
    
    References:
        - Original NumPy implementation in eryx/models.py:RigidBodyRotations
    """
    # TODO: Implement initialization and methods with PyTorch operations
    pass
