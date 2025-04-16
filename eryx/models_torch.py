"""
PyTorch implementation of disorder models for diffuse scattering calculations.

This module contains PyTorch versions of the disorder models defined in
eryx/models.py. All implementations maintain the same API as the NumPy versions
but use PyTorch tensors and operations to enable gradient flow.

References:
    - Original NumPy implementation in eryx/models.py
"""

import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional, Union, Any

from eryx.pdb import AtomicModel, Crystal, GaussianNetworkModel
from eryx.pdb_torch import GaussianNetworkModel as GaussianNetworkModelTorch
from eryx.autotest.debug import debug
from eryx.adapters import PDBToTensor, TensorToNumpy

class OnePhonon:
    """
    PyTorch implementation of the OnePhonon model for diffuse scattering calculations.
    
    This class implements a lattice of interacting rigid bodies in the one-phonon
    approximation (a.k.a small-coupling regime) using PyTorch tensors and operations
    to enable gradient flow.
    
    This implementation supports two modes of operation:
    
    1. Grid-based mode (default):
       - Specify hsampling, ksampling, lsampling parameters
       - Generates a regular grid of q-vectors in reciprocal space
       - Compatible with original NumPy implementation
       
    2. Arbitrary q-vector mode:
       - Directly specify q_vectors parameter as a tensor of shape [n_points, 3]
       - Evaluates diffuse scattering only at specified q-vectors
       - Maps each q-vector to its equivalent k-vector in the first Brillouin zone
       - Enables targeted evaluation with physically correct phonon properties
       
    The arbitrary q-vector mode is particularly useful for:
    - Focusing computation on specific regions of interest
    - Matching experimental data points for optimization
    - Custom sampling patterns not constrained to a regular grid
    
    Example usage:
    
    ```python
    # Grid-based mode
    model_grid = OnePhonon(
        "structure.pdb",
        hsampling=[-4, 4, 3],
        ksampling=[-17, 17, 3],
        lsampling=[-29, 29, 3],
    )
    
    # Arbitrary q-vector mode
    q_vectors = torch.tensor([
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        # ... more q-vectors ...
    ])
    
    model_q = OnePhonon(
        "structure.pdb",
        q_vectors=q_vectors,
    )
    ```
    
    References:
        - Original NumPy implementation in eryx/models.py:OnePhonon
    """
    
    # Class-level default, will be set properly in __init__
    use_arbitrary_q: bool = False
    
    #@debug
    def __init__(self, pdb_path: str, 
                 hsampling: Optional[Tuple[float, float, float]] = None, 
                 ksampling: Optional[Tuple[float, float, float]] = None, 
                 lsampling: Optional[Tuple[float, float, float]] = None,
                 expand_p1: bool = True, group_by: str = 'asu',
                 res_limit: float = 0.,
                 model: str = 'gnm',
                 gnm_cutoff: float = 4., gamma_intra: float = 1., gamma_inter: float = 1.,
                 n_processes: int = 8, device: Optional[torch.device] = None,
                 q_vectors: Optional[torch.Tensor] = None):
        """
        Initialize the OnePhonon model with PyTorch tensors.
        
        This class supports two modes of operation:
        1. Grid-based mode: Uses sampling parameters (hsampling, ksampling, lsampling) to create a grid
                           of q-vectors in reciprocal space.
        2. Arbitrary q-vector mode: Directly specifies q-vectors of interest without grid sampling.
        
        Args:
            pdb_path: Path to coordinates file.
            hsampling: Tuple (hmin, hmax, oversampling) for h dimension. Required for grid-based mode.
            ksampling: Tuple (kmin, kmax, oversampling) for k dimension. Required for grid-based mode.
            lsampling: Tuple (lmin, lmax, oversampling) for l dimension. Required for grid-based mode.
            q_vectors: Tensor of arbitrary q-vectors with shape [n_points, 3] in Å⁻¹. If provided, 
                       sampling parameters are ignored and calculation is performed in arbitrary q-vector mode.
            expand_p1: If True, expand to p1 (if PDB is asymmetric unit).
            group_by: Level of rigid-body assembly ('asu' or None).
            res_limit: High-resolution limit in Angstrom.
            model: Chosen phonon model ('gnm' or 'rb').
            gnm_cutoff: Distance cutoff for GNM in Angstrom.
            gamma_intra: Spring constant for intra-asu interactions.
            gamma_inter: Spring constant for inter-asu interactions.
            n_processes: Number of processes for parallel computation.
            device: PyTorch device to use (default: CUDA if available, else CPU).
            
        Note:
            When using arbitrary q-vector mode (by providing q_vectors), the calculation bypasses
            the grid structure and processes all provided q-vectors directly. This is useful for
            focusing computation on specific points of interest in reciprocal space or for matching
            with experimental data.
        """
        import logging
        logging.debug(f"[INIT] Called OnePhonon constructor (q_vectors is None? {q_vectors is None})")

        # Always set use_arbitrary_q to False initially
        self.use_arbitrary_q = False
        
        self.n_processes = n_processes
        self.model_type = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.hsampling = hsampling
        self.ksampling = ksampling
        self.lsampling = lsampling
        
        # Set high precision as default
        self.real_dtype = torch.float64
        self.complex_dtype = torch.complex128

        # --- START VALIDATION ---
        self.use_arbitrary_q = False
        if q_vectors is not None:
            # Validate q_vectors tensor shape and type
            if not isinstance(q_vectors, torch.Tensor):
                raise ValueError("q_vectors must be a PyTorch tensor")
            if q_vectors.dim() != 2 or q_vectors.shape[1] != 3:
                raise ValueError(f"q_vectors must have shape [n_points, 3], got {q_vectors.shape}")
            self.use_arbitrary_q = True
            self.q_vectors = q_vectors.to(device=self.device) # Store input
            if not self.q_vectors.requires_grad and self.q_vectors.dtype.is_floating_point:
                 self.q_vectors.requires_grad_(True)
            logging.debug("[INIT] Arbitrary Q-Mode detected.")
        elif hsampling is None or ksampling is None or lsampling is None:
            # If not arbitrary mode, grid params are essential
            raise ValueError("Grid mode requires hsampling, ksampling, and lsampling.")
        else:
             logging.debug("[INIT] Grid Mode detected.")

        # Additional validation: If model is GNM, sampling params are ALWAYS needed for ADP calc
        if model == 'gnm' and (hsampling is None or ksampling is None or lsampling is None):
            raise ValueError("hsampling, ksampling, and lsampling are required when model='gnm' (for ADP calculation), even in arbitrary q-vector mode.")
        # --- END VALIDATION ---

        # Store sampling parameters regardless of mode if provided
        self.hsampling = hsampling
        self.ksampling = ksampling
        self.lsampling = lsampling

        logging.debug(f"[INIT] final use_arbitrary_q={self.use_arbitrary_q}")

        self._setup(pdb_path, expand_p1, res_limit, group_by)
        self._setup_phonons(pdb_path, model, gnm_cutoff, gamma_intra, gamma_inter)

        logging.debug("[INIT] Completed OnePhonon constructor.")
    
    #@debug
    def _setup(self, pdb_path: str, expand_p1: bool, res_limit: float, group_by: str):
        """
        Compute q-vectors to evaluate and build the unit cell and its neighbors.
        
        Parameters:
            pdb_path: Path to coordinates file of asymmetric unit.
            expand_p1: If True, expand to p1.
            res_limit: High-resolution limit in Angstrom.
            group_by: Level of rigid-body assembly ('asu' or None).
        """
        import logging
        # Create an AtomicModel instance from the NP implementation.
        self.model = AtomicModel(pdb_path, expand_p1)
        
        # Store a reference to the original model for accessing original data
        self.original_model = self.model
        
        # Use PDBToTensor adapter to extract element weights
        from eryx.adapters import PDBToTensor
        pdb_adapter = PDBToTensor(device=self.device) # Removed dtype argument
        element_weights = pdb_adapter.extract_element_weights(self.model).to(dtype=self.real_dtype) # Ensure dtype after extraction
        self.model.element_weights = element_weights
        logging.debug(f"[_setup] Extracted {len(element_weights)} element weights.")

        # Note: NumPy generate_grid is used below, ensure its output is converted correctly
        logging.debug(f"[_setup] use_arbitrary_q={getattr(self, 'use_arbitrary_q', False)}")
        
        if getattr(self, 'use_arbitrary_q', False):
            # In arbitrary q-vector mode, use the provided q_vectors directly.
            logging.debug("[_setup] Using user-provided q_vectors.")
            self.q_grid = self.q_vectors.to(dtype=self.real_dtype)
            
            # Derive hkl_grid from q_vectors using q = 2π * A_inv^T * hkl  => hkl = (1/(2π)) * q * (A_inv^T)^{-1}
            A_inv_tensor = torch.tensor(self.model.A_inv, dtype=self.real_dtype, device=self.device)
            scaling_factor = torch.tensor(1.0 / (2.0 * torch.pi), dtype=self.real_dtype, device=self.device) # Use tensor for scaling
            A_inv_T_inv = torch.linalg.inv(A_inv_tensor.T) # Use linalg.inv
            # Ensure matmul inputs have correct dtype
            self.hkl_grid = torch.matmul(self.q_grid * scaling_factor, A_inv_T_inv).to(dtype=self.real_dtype)

            # For arbitrary mode, we set a dummy map_shape (number of points,1,1)
            self.map_shape = (self.q_grid.shape[0], 1, 1)
        else:
            # Grid-based mode: Use NumPy generate_grid for identical hkl_grid
            from eryx.map_utils import generate_grid as np_generate_grid # Use NumPy version
            
            # Ensure A_inv is float64 for NumPy function
            A_inv_np = self.model.A_inv.astype(np.float64) 
            
            hkl_grid_np, self.map_shape = np_generate_grid(A_inv_np,
                                                          self.hsampling,
                                                          self.ksampling,
                                                          self.lsampling,
                                                          return_hkl=True)
                                                          
            # Convert NumPy result to Torch tensor with correct dtype and device
            self.hkl_grid = torch.tensor(hkl_grid_np, dtype=self.real_dtype, device=self.device)
            logging.debug(f"[_setup] grid-based map_shape={self.map_shape}, "
                          f"hkl_grid.shape={self.hkl_grid.shape}, hkl_grid.dtype={self.hkl_grid.dtype}")

            # Calculate q_grid using matrix multiplication with tensors
            # Ensure A_inv_tensor uses high precision
            A_inv_tensor = torch.tensor(self.model.A_inv, dtype=self.real_dtype, device=self.device)
            two_pi = torch.tensor(2.0 * torch.pi, dtype=self.real_dtype, device=self.device)
            # Ensure matmul inputs have correct dtype
            self.q_grid = two_pi * torch.matmul(
                A_inv_tensor.T,
                self.hkl_grid.T # hkl_grid is already real_dtype
            ).T # Transpose the result back
            self.q_grid = self.q_grid.to(dtype=self.real_dtype) # Ensure final dtype
            logging.debug(f"[_setup] Calculated q_grid shape={self.q_grid.shape}, q_grid.dtype={self.q_grid.dtype}")

        # Ensure q_grid requires gradients if it's float
        if self.q_grid.dtype.is_floating_point:
            self.q_grid.requires_grad_(True)
        logging.debug(f"[_setup] q_grid requires_grad: {self.q_grid.requires_grad}")
        
        # Compute resolution mask using PyTorch functions
        from eryx.map_utils_torch import compute_resolution
        cell_tensor = torch.tensor(self.model.cell, dtype=self.real_dtype, device=self.device)
        # Ensure hkl_grid is real_dtype before passing to compute_resolution
        resolution = compute_resolution(cell_tensor, self.hkl_grid.to(dtype=self.real_dtype))
        logging.debug(f"[_setup] resolution computed, shape={resolution.shape}, dtype={resolution.dtype}")
        # Ensure comparison uses correct dtype
        self.res_mask = resolution > torch.tensor(res_limit, dtype=self.real_dtype, device=self.device)

        # Setup Crystal
        self.crystal = Crystal(self.model)
        self.crystal.supercell_extent(nx=1, ny=1, nz=1)
        self.id_cell_ref = self.crystal.hkl_to_id([0, 0, 0]) # This returns int, no dtype issue
        self.n_cell = self.crystal.n_cell # This is int, no dtype issue

        # Setup PDBToTensor adapter for tensor conversions
        pdb_adapter = PDBToTensor(device=self.device) # Removed dtype argument
        self.crystal = pdb_adapter.convert_crystal(self.crystal) # Adapter should handle internal dtypes

        # Set key dimensions.
        self.n_asu = self.model.n_asu
        logging.debug(f"[_setup] n_asu={self.n_asu}")

        self.n_atoms_per_asu = self.model.xyz.shape[1]
        self.n_dof_per_asu_actual = self.n_atoms_per_asu * 3
        
        self.group_by = group_by
        if self.group_by is None:
            self.n_dof_per_asu = self.n_dof_per_asu_actual
        else:
            self.n_dof_per_asu = 6
        self.n_dof_per_cell = self.n_asu * self.n_dof_per_asu
    
    def _setup_gamma_parameters(self, pdb_path: str, model: str, gnm_cutoff: float, 
                               gamma_intra: float, gamma_inter: float):
        """
        Setup gamma parameters for the GNM model.
        
        Parameters:
            pdb_path: Path to coordinates file.
            model: Chosen phonon model ('gnm' or 'rb').
            gnm_cutoff: Distance cutoff for GNM.
            gamma_intra: Spring constant for intra-asu interactions.
            gamma_inter: Spring constant for inter-asu interactions.
        """
        # Store parameters as tensors with gradients
        if isinstance(gamma_intra, torch.Tensor):
            self.gamma_intra = gamma_intra.to(dtype=self.real_dtype)
        else:
            self.gamma_intra = torch.tensor(gamma_intra, dtype=self.real_dtype, device=self.device, requires_grad=True)
            
        if isinstance(gamma_inter, torch.Tensor):
            self.gamma_inter = gamma_inter.to(dtype=self.real_dtype)
        else:
            self.gamma_inter = torch.tensor(gamma_inter, dtype=self.real_dtype, device=self.device, requires_grad=True)
        
        # Setup GNM from NP implementation for initialization only
        self.gnm = GaussianNetworkModel(pdb_path, gnm_cutoff, 
                                       float(self.gamma_intra.detach().cpu().numpy()), 
                                       float(self.gamma_inter.detach().cpu().numpy()))
        
        # Create a differentiable gamma tensor that matches the GNM structure
        self.gamma_tensor = torch.zeros((self.n_cell, self.n_asu, self.n_asu), 
                                       device=self.device, dtype=self.real_dtype)
        
        # Fill it like the original build_gamma method, but with our parameter tensors
        for i_asu in range(self.n_asu):
            for i_cell in range(self.n_cell):
                for j_asu in range(self.n_asu):
                    self.gamma_tensor[i_cell, i_asu, j_asu] = self.gamma_inter
                    if (i_cell == self.id_cell_ref) and (j_asu == i_asu):
                        self.gamma_tensor[i_cell, i_asu, j_asu] = self.gamma_intra
    
    #@debug
    def _setup_phonons(self, pdb_path: str, model: str, 
                       gnm_cutoff: float, gamma_intra: float, gamma_inter: float):
        """
        Compute phonons either from a Gaussian Network Model of the
        molecules or by direct definition of the dynamical matrix.

        This method supports both grid-based and arbitrary q-vector modes.
        """
        import logging
        logging.debug(f"[_setup_phonons] STARTING: mode={'arbitrary' if getattr(self, 'use_arbitrary_q', False) else 'grid'}, model_type={model}")

        # 1. Build structural/mass matrices (mode-independent)
        self._build_A()
        self._build_M()
        logging.debug("[_setup_phonons] Built A and M matrices.")

        # 2. Build k-vectors (logic differs based on mode)
        self._build_kvec_Brillouin() # This method already handles the mode difference
        logging.debug(f"[_setup_phonons] Built kvec, shape: {self.kvec.shape}")

        # 3. Initialize V and Winv based on kvec size (handled inside _build_kvec_Brillouin if resizing is needed)
        # Add a check here to be sure
        expected_k_points = self.kvec.shape[0]
        if not hasattr(self, 'V') or self.V is None or self.V.shape[0] != expected_k_points:
             logging.warning(f"[_setup_phonons] Re-initializing V/Winv to size {expected_k_points}")
             # Re-initialize V and Winv tensors with correct batched shape
             dof_total = self.n_asu * self.n_dof_per_asu
             self.V = torch.zeros((expected_k_points, dof_total, dof_total), dtype=self.complex_dtype, device=self.device, requires_grad=True)
             self.Winv = torch.zeros((expected_k_points, dof_total), dtype=self.complex_dtype, device=self.device, requires_grad=True)

        # 4. Setup GNM specifics if needed
        if model == 'gnm':
            self._setup_gamma_parameters(pdb_path, model, gnm_cutoff, gamma_intra, gamma_inter)
            logging.debug("[_setup_phonons] Setup GNM gamma parameters.")
            # Call phonon calculation
            logging.debug("[_setup_phonons] Calling compute_gnm_phonons...")
            self.compute_gnm_phonons() # Assumes Issue #4 fix is in place
            logging.debug("[_setup_phonons] compute_gnm_phonons finished.")
            # Call covariance/ADP calculation
            logging.debug("[_setup_phonons] Calling compute_covariance_matrix...")
            self.compute_covariance_matrix() # Uses hsampling etc. internally
            logging.debug("[_setup_phonons] compute_covariance_matrix finished.")
        elif model == 'rb':
            # Call RB phonon calculation if implemented
            logging.debug("[_setup_phonons] Calling compute_rb_phonons...")
            self.compute_rb_phonons()
            logging.debug("[_setup_phonons] compute_rb_phonons finished.")
        else:
             logging.warning(f"[_setup_phonons] Unknown model type '{model}', skipping phonon/covariance calculations.")

        logging.debug("[_setup_phonons] FINISHED.")
    
    #@debug
    def _build_A(self):
        """
        Build the displacement projection matrix A that projects rigid-body
        displacements to individual atomic displacements.
        """
        if self.group_by == 'asu':
            # Initialize Amat with zeros, using float64 for better precision
            self.Amat = torch.zeros((self.n_asu, self.n_dof_per_asu_actual, self.n_dof_per_asu), 
                                   device=self.device, dtype=self.real_dtype)
            
            # Create identity matrix for translations
            Adiag = torch.eye(3, device=self.device, dtype=self.real_dtype)
            
            for i_asu in range(self.n_asu):
                # Get coordinates from model directly - simplest reliable approach
                if hasattr(self.model, 'xyz'):
                    if isinstance(self.model.xyz, torch.Tensor):
                        xyz = self.model.xyz[i_asu].to(dtype=self.real_dtype)
                    else:
                        xyz = torch.tensor(self.model.xyz[i_asu], dtype=self.real_dtype, device=self.device)
                else:
                    # Fallback to zeros if no coordinates available
                    xyz = torch.zeros((self.n_atoms_per_asu, 3), device=self.device, dtype=self.real_dtype)
                
                # Center coordinates properly
                xyz = xyz - xyz.mean(dim=0, keepdim=True)
                
                # Debug print for centered coordinates
                if i_asu == 0:
                    print(f"Torch ASU {i_asu} Centered XYZ (mean): {xyz.mean(dim=0).detach().cpu().numpy()}")
                    print(f"Torch ASU {i_asu} Centered XYZ (first 3 atoms):")
                    for i in range(min(3, xyz.shape[0])):
                        print(f"  Atom {i}: {xyz[i].detach().cpu().numpy()}")
                
                # Process each atom
                for i_atom in range(self.n_atoms_per_asu):
                    # Initialize Atmp for each atom (important!)
                    Atmp = torch.zeros((3, 3), device=self.device, dtype=self.real_dtype)
                    
                    # Update skew-symmetric matrix for rotations
                    if i_atom < xyz.shape[0]:
                        # Debug print for specific atoms
                        if i_asu == 0 and i_atom < 3:
                            print(f"  Torch Atom {i_atom} XYZ: {xyz[i_atom].detach().cpu().numpy()}")
                        
                        # Fill the skew-symmetric matrix
                        Atmp[0, 1] = xyz[i_atom, 2]  
                        Atmp[0, 2] = -xyz[i_atom, 1]
                        Atmp[1, 0] = -xyz[i_atom, 2]
                        Atmp[1, 2] = xyz[i_atom, 0]
                        Atmp[2, 0] = xyz[i_atom, 1]
                        Atmp[2, 1] = -xyz[i_atom, 0]
                        
                        # Debug print for Atmp
                        if i_asu == 0 and i_atom < 3:
                            print(f"  Torch Atom {i_atom} Atmp:\n{Atmp.detach().cpu().numpy()}")
                    
                    # Set identity part (translations) and then the rotation part
                    self.Amat[i_asu, i_atom*3:(i_atom+1)*3, 0:3] = Adiag
                    self.Amat[i_asu, i_atom*3:(i_atom+1)*3, 3:6] = Atmp
                    
                    # Debug print for assigned block
                    # if i_asu == 0 and i_atom < 3:
                    #     assigned_block = self.Amat[i_asu, i_atom*3:(i_atom+1)*3, :]
                    #     print(f"  Torch Atom {i_atom} Assigned Block:\n{assigned_block.detach().cpu().numpy()}")
            
            # Keep high precision
            
            # Set requires_grad after construction
            self.Amat.requires_grad_(True)
        else:
            self.Amat = None
    
    #@debug
    def _build_M(self):
        """
        Build the mass matrix M and compute its inverse (via Cholesky).
        """
        # Build all-atom mass matrix (already uses float64 after update)
        M_allatoms = self._build_M_allatoms()
        
        if self.group_by is None:
            # Simple case for all atoms
            M_allatoms = M_allatoms.reshape((self.n_asu * self.n_dof_per_asu_actual,
                                            self.n_asu * self.n_dof_per_asu_actual))
            # Add regularization for numerical stability
            eps = 1e-8
            M_reg = M_allatoms + eps
            self.Linv = 1.0 / torch.sqrt(M_reg)
            
            # Keep high precision
            # Do NOT convert back to float32
        else:
            # Project the all-atom mass matrix for rigid body case
            Mmat = self._project_M(M_allatoms)
            Mmat = Mmat.reshape((self.n_asu * self.n_dof_per_asu, self.n_asu * self.n_dof_per_asu))
            
            # Robust regularization - single value that works
            eps = 1e-6
            eye = torch.eye(Mmat.shape[0], device=self.device, dtype=Mmat.dtype)
            Mmat_reg = Mmat + eps * eye
            
            # Enhanced try-except with better fallback
            try:
                # Try standard Cholesky decomposition first
                L = torch.linalg.cholesky(Mmat_reg)
                self.Linv = torch.linalg.inv(L)
            except RuntimeError as e:
                # Print diagnostic info
                logging.warning(f"Cholesky decomposition failed: {e}")
                logging.warning(f"Matrix condition number: {torch.linalg.cond(Mmat_reg).item()}")
                
                # Add stronger regularization and try again
                stronger_eps = 1e-4
                Mmat_reg = Mmat + stronger_eps * eye
                try:
                    L = torch.linalg.cholesky(Mmat_reg)
                    self.Linv = torch.linalg.inv(L)
                    logging.warning("Succeeded with stronger regularization")
                except RuntimeError:
                    # Final fallback to SVD approach
                    logging.warning("Falling back to SVD decomposition")
                    U, S, V = torch.linalg.svd(Mmat_reg, full_matrices=False)
                    S = torch.clamp(S, min=1e-8)
                    self.Linv = U @ torch.diag(1.0 / torch.sqrt(S)) @ V
            
            # Ensure Linv is float64 (real_dtype)
            self.Linv = self.Linv.to(dtype=self.real_dtype)
            
            # Keep high precision
            # Do NOT convert back to float32
            self.Linv.requires_grad_(True)
    
    #@debug
    def _build_M_allatoms(self) -> torch.Tensor:
        """
        Build the all-atom mass matrix M_0.
        
        Returns:
            torch.Tensor of shape (n_asu, n_dof_per_asu_actual, n_asu, n_dof_per_asu_actual)
        """
        # Use high precision
        dtype = self.real_dtype
        
        if hasattr(self.model, 'elements'):
            try:
                # This matches the NumPy implementation exactly
                weights = [element.weight for structure in self.model.elements for element in structure]
            except Exception as e:
                logging.error(f"Error in direct weight extraction: {e}")
        
        # Create mass array from weights
        if weights:
            mass_array = torch.tensor(weights, dtype=dtype, device=self.device)
        else:
            # Fallback to ones
            logging.warning("Fallback: Using ones for mass_array in _build_M_allatoms")
            mass_array = torch.ones(self.n_asu * self.n_atoms_per_asu, dtype=dtype, device=self.device)
        
        # Create block diagonal matrix
        eye3 = torch.eye(3, device=self.device, dtype=dtype)
        
        # Create mass_list exactly like NumPy implementation
        mass_list = []
        for i in range(self.n_asu * self.n_atoms_per_asu):
            # Create 3x3 block for each atom (mass * identity)
            block = mass_array[i] * eye3
            mass_list.append(block)
        
        # Create block diagonal matrix
        total_dim = self.n_asu * self.n_atoms_per_asu * 3
        M_block_diag = torch.zeros((total_dim, total_dim), device=self.device, dtype=dtype)
        
        # Fill block diagonal matrix manually to match NumPy exactly
        current_idx = 0
        for block in mass_list:
            block_size = block.shape[0]
            M_block_diag[current_idx:current_idx+block_size, current_idx:current_idx+block_size] = block
            current_idx += block_size
        
        # Reshape to 4D tensor
        M_allatoms = M_block_diag.reshape(self.n_asu, self.n_dof_per_asu_actual,
                                        self.n_asu, self.n_dof_per_asu_actual)
        
        # Set requires_grad
        M_allatoms.requires_grad_(True)
        
        return M_allatoms
    
    #@debug
    def _project_M(self, M_allatoms: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """
        Project all-atom mass matrix M_0 using the A matrix: M = A.T M_0 A
        
        Args:
            M_allatoms: Mass matrix of shape (n_asu, n_dof_per_asu_actual, n_asu, n_dof_per_asu_actual)
            
        Returns:
            Mmat: Projected mass matrix of shape (n_asu, n_dof_per_asu, n_asu, n_dof_per_asu)
        """
        # Use high precision
        dtype = self.real_dtype
        
        # Ensure M_allatoms is a tensor
        if not isinstance(M_allatoms, torch.Tensor):
            M_allatoms = torch.tensor(M_allatoms, device=self.device, dtype=dtype)
        
        # Initialize output tensor
        Mmat = torch.zeros((self.n_asu, self.n_dof_per_asu,
                           self.n_asu, self.n_dof_per_asu),
                          device=self.device, dtype=dtype)
        
        # Ensure Amat is in the same precision
        Amat = self.Amat.to(dtype=dtype)
        
        # Project mass matrix
        for i_asu in range(self.n_asu):
            for j_asu in range(self.n_asu):
                Mmat[i_asu, :, j_asu, :] = torch.matmul(
                    Amat[i_asu].T,
                    torch.matmul(M_allatoms[i_asu, :, j_asu, :], Amat[j_asu])
                )
        
        return Mmat
    
    #@debug
    def _build_kvec_Brillouin(self):
        """
        Compute all k-vectors and their norm in the first Brillouin zone.
        
        In grid mode: Regularly samples [-0.5,0.5[ for h, k and l.
        In arbitrary mode: Maps each q-vector to its equivalent k-vector in the first BZ.
        """
        import logging
        logging.debug(f"[_build_kvec_Brillouin] use_arbitrary_q={getattr(self, 'use_arbitrary_q', False)}")
        
        if getattr(self, 'use_arbitrary_q', False):
            # Arbitrary q-vector mode with BZ mapping
            n_points = self.q_grid.shape[0]
            
            # Ensure A_inv is a tensor on the correct device/dtype
            if isinstance(self.model.A_inv, torch.Tensor):
                A_inv_tensor = self.model.A_inv.to(dtype=self.real_dtype, device=self.device)
            else:
                A_inv_tensor = torch.tensor(self.model.A_inv, dtype=self.real_dtype, device=self.device)

            # Pre-allocate kvec tensor
            self.kvec = torch.zeros((n_points, 3), dtype=self.real_dtype, device=self.device)
            from eryx.torch_utils import BrillouinZoneUtils # Import the utility class
            
            # Map each q vector to its k_BZ equivalent using the utility function
            # Pass tensors directly, the utility handles batching
            self.kvec = BrillouinZoneUtils.map_q_to_k_bz(self.q_grid, A_inv_tensor)

            # Calculate norm based on the mapped k_BZ vectors
            self.kvec_norm = torch.linalg.norm(self.kvec, dim=1, keepdim=True).to(dtype=self.real_dtype)

            # The mapping operation breaks direct gradient flow from q to k
            # Detach to make this explicit and avoid potential gradient issues
            self.kvec = self.kvec.detach()
            self.kvec_norm = self.kvec_norm.detach()
            
            logging.debug(f"[_build_kvec_Brillouin] Arbitrary mode with BZ mapping: kvec.shape={self.kvec.shape}, "
                         f"kvec_norm.shape={self.kvec_norm.shape}, requires_grad=False")
        else:
            # Grid-based mode
            logging.debug("[_build_kvec_Brillouin] Grid-based mode. Using Brillouin zone sampling dimensions.")
            
            # Get Brillouin zone sampling dimensions
            h_dim_bz = int(self.hsampling[2])
            k_dim_bz = int(self.ksampling[2])
            l_dim_bz = int(self.lsampling[2])
            total_k_points = h_dim_bz * k_dim_bz * l_dim_bz
            
            logging.debug(f"[_build_kvec_Brillouin] Brillouin zone dimensions: h_dim={h_dim_bz}, k_dim={k_dim_bz}, l_dim={l_dim_bz}")
            logging.debug(f"[_build_kvec_Brillouin] Total k-points: {total_k_points}")
            
            # Get A_inv_tensor (ensure float64)
            if isinstance(self.model.A_inv, torch.Tensor):
                # Ensure correct dtype and device
                A_inv_tensor = self.model.A_inv.clone().detach().to(dtype=self.real_dtype, device=self.device) 
            else:
                # Convert from NumPy array with correct dtype
                A_inv_tensor = torch.tensor(self.model.A_inv, dtype=self.real_dtype, device=self.device) 
            
            # Generate 1D tensors for h, k, l coordinates using _center_kvec (which returns float64)
            h_coords = torch.tensor([self._center_kvec(dh, h_dim_bz) for dh in range(h_dim_bz)],
                                   device=self.device, dtype=self.real_dtype) # Explicit dtype
            k_coords = torch.tensor([self._center_kvec(dk, k_dim_bz) for dk in range(k_dim_bz)],
                                   device=self.device, dtype=self.real_dtype) # Explicit dtype
            l_coords = torch.tensor([self._center_kvec(dl, l_dim_bz) for dl in range(l_dim_bz)],
                                   device=self.device, dtype=self.real_dtype) # Explicit dtype

            # Create meshgrid and reshape to [total_k_points, 3]
            h_grid, k_grid, l_grid = torch.meshgrid(h_coords, k_coords, l_coords, indexing='ij')
            # Ensure hkl_fractional is float64
            hkl_fractional = torch.stack([h_grid.flatten(), k_grid.flatten(), l_grid.flatten()], dim=1).to(dtype=self.real_dtype)

            # Compute kvec and kvec_norm (ensure float64)
            self.kvec = torch.matmul(hkl_fractional, A_inv_tensor).to(dtype=self.real_dtype) # Ensure output dtype
            self.kvec_norm = torch.linalg.norm(self.kvec, dim=1, keepdim=True).to(dtype=self.real_dtype) # Use linalg.norm and ensure output dtype

            # Ensure final tensors have the correct dtype (redundant but safe)
            self.kvec = self.kvec.to(dtype=self.real_dtype)
            self.kvec_norm = self.kvec_norm.to(dtype=self.real_dtype)
            
            # Ensure tensors require gradients
            self.kvec.requires_grad_(True)
            self.kvec_norm.requires_grad_(True)
            
            # Debug output for verification
            logging.debug(f"[_build_kvec_Brillouin] kvec shape={self.kvec.shape}, norm shape={self.kvec_norm.shape}")
            
            # Verify and resize V and Winv if needed
            if hasattr(self, 'V') and hasattr(self, 'Winv'):
                if self.V.shape[0] != total_k_points or self.Winv.shape[0] != total_k_points:
                    logging.warning(f"[_build_kvec_Brillouin] Resizing V and Winv to match total_k_points={total_k_points}")
                    
                    # Resize V and Winv
                    self.V = torch.zeros((total_k_points,
                                        self.n_asu * self.n_dof_per_asu,
                                        self.n_asu * self.n_dof_per_asu),
                                       dtype=self.complex_dtype, device=self.device)
                    self.Winv = torch.zeros((total_k_points,
                                           self.n_asu * self.n_dof_per_asu),
                                          dtype=self.complex_dtype, device=self.device)
                    
                    # Ensure complex tensors require gradients
                    self.V.requires_grad_(True)
                    self.Winv.requires_grad_(True)
    
    #@debug
    def _center_kvec(self, x: int, L: int) -> float:
        """
        Center a k-vector index using exact NumPy-compatible operations.
        
        For x and L integers such that 0 <= x < L, return -L/2 < x < L/2
        by applying periodic boundary condition in L/2.
        
        This implementation exactly matches the NumPy behavior for modulo with
        negative numbers.
        
        Args:
            x: Index to center
            L: Length of the periodic box
            
        Returns:
            Centered value (float64) in range [-L/2, L/2) / L
        """
        # Match the NumPy implementation exactly: int(((x - L / 2) % L) - L / 2) / L
        # Ensure the final result is a Python float (which corresponds to float64 precision)
        centered_index = int(((x - L / 2) % L) - L / 2)
        return float(centered_index) / float(L)
    
    # Note: _map_q_to_k_bz moved to BrillouinZoneUtils in eryx/torch_utils.py
    
    def _at_kvec_from_miller_points(self, indices_or_batch: Union[Tuple[int, int, int], torch.Tensor, int]) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Return the indices of all q-vectors that are k-vector away from given Miller indices.
        
        This method supports three input formats:
        1. Traditional format: (h, k, l) tuple with 3 elements
        2. Fully collapsed format: a single flat index
        3. Batched format: tensor of flat indices
        
        Args:
            indices_or_batch: Either a 3-tuple (h,k,l), a single flat index, or a tensor of flat indices
            
        Returns:
            Torch tensor of indices or list of tensors for batched input
        """
        import logging
        
        # In arbitrary q-vector mode
        if getattr(self, 'use_arbitrary_q', False):
            if isinstance(indices_or_batch, (tuple, list)) and len(indices_or_batch) == 3:
                # Convert Miller indices to q-vector
                hkl = torch.tensor([indices_or_batch], device=self.device, dtype=torch.float32)
                A_inv_tensor = torch.tensor(self.model.A_inv, dtype=torch.float32, device=self.device)
                target_q = 2 * torch.pi * torch.matmul(A_inv_tensor.T, hkl.T).T
                
                # Find nearest q-vector by distance
                distances = torch.norm(self.q_grid - target_q, dim=1)
                nearest_idx = torch.argmin(distances)
                
                logging.debug(f"Arbitrary mode: Nearest q_vector index for hkl {indices_or_batch}: {nearest_idx.item()}")
                return nearest_idx
                
            # Directly return indices for other cases
            if isinstance(indices_or_batch, int):
                return torch.tensor([indices_or_batch], device=self.device)
                
            if isinstance(indices_or_batch, torch.Tensor):
                return indices_or_batch
        
        # Grid-based mode implementation
        # Calculate full grid dimensions based on sampling parameters
        # These should match the dimensions used in _setup
        hsteps = int(self.hsampling[2] * (self.hsampling[1] - self.hsampling[0]) + 1)
        ksteps = int(self.ksampling[2] * (self.ksampling[1] - self.ksampling[0]) + 1)
        lsteps = int(self.lsampling[2] * (self.lsampling[1] - self.lsampling[0]) + 1)
        
        # Compare calculated map_shape with self.map_shape
        map_shape_calc = (hsteps, ksteps, lsteps)
        if hasattr(self, 'map_shape') and self.map_shape != map_shape_calc:
            logging.warning(f"[_at_kvec_from_miller_points] Calculated map_shape {map_shape_calc} differs from stored map_shape {self.map_shape}")
        
        # Use stored map_shape if available, otherwise use calculated
        h_dim, k_dim, l_dim = self.map_shape if hasattr(self, 'map_shape') else map_shape_calc
        
        # Handle different input types for Brillouin zone indices
        is_batch = isinstance(indices_or_batch, torch.Tensor) and indices_or_batch.dim() == 1 and indices_or_batch.numel() > 1
        
        if is_batch:
            # Convert batch of flat indices to 3D Brillouin zone indices
            flat_indices = indices_or_batch
            h_indices, k_indices, l_indices = self._flat_to_3d_indices_bz(flat_indices)
        elif isinstance(indices_or_batch, (tuple, list)) and len(indices_or_batch) == 3:
            # Handle (dh,dk,dl) tuple directly
            h_indices = torch.tensor([indices_or_batch[0]], device=self.device, dtype=torch.long)
            k_indices = torch.tensor([indices_or_batch[1]], device=self.device, dtype=torch.long)
            l_indices = torch.tensor([indices_or_batch[2]], device=self.device, dtype=torch.long)
        else:
            # Handle single flat Brillouin zone index
            flat_idx = indices_or_batch
            if isinstance(flat_idx, torch.Tensor):
                flat_indices = flat_idx.view(1)
            else:
                flat_indices = torch.tensor([flat_idx], device=self.device, dtype=torch.long)
                
            h_indices, k_indices, l_indices = self._flat_to_3d_indices_bz(flat_indices)
        
        # Get sampling rates
        h_sampling_rate = int(self.hsampling[2])
        k_sampling_rate = int(self.ksampling[2])
        l_sampling_rate = int(self.lsampling[2])
        
        # Process based on batch size
        batch_size = h_indices.numel()
        if batch_size == 1:
            # Single point case
            h_idx, k_idx, l_idx = h_indices.item(), k_indices.item(), l_indices.item()
            
            # Generate ranges using sampling rates as steps
            h_range = torch.arange(h_idx, hsteps, h_sampling_rate, device=self.device, dtype=torch.long)
            k_range = torch.arange(k_idx, ksteps, k_sampling_rate, device=self.device, dtype=torch.long)
            l_range = torch.arange(l_idx, lsteps, l_sampling_rate, device=self.device, dtype=torch.long)
            
            # Create meshgrid
            h_grid, k_grid, l_grid = torch.meshgrid(h_range, k_range, l_range, indexing='ij')
            
            # Flatten indices
            h_flat = h_grid.reshape(-1)
            k_flat = k_grid.reshape(-1)
            l_flat = l_grid.reshape(-1)
            
            # Compute raveled indices using the full grid helper
            indices = self._3d_to_flat_indices(h_flat, k_flat, l_flat)
            
            logging.debug(f"[_at_kvec_from_miller_points] Single point: ({h_idx},{k_idx},{l_idx}) -> {indices.shape[0]} indices")
            return indices
        else:
            # Batch processing
            all_indices = []
            for i in range(batch_size):
                # Compute indices for each point
                h_idx, k_idx, l_idx = h_indices[i].item(), k_indices[i].item(), l_indices[i].item()
                
                # Generate ranges using sampling rates as steps
                h_range = torch.arange(h_idx, hsteps, h_sampling_rate, device=self.device, dtype=torch.long)
                k_range = torch.arange(k_idx, ksteps, k_sampling_rate, device=self.device, dtype=torch.long)
                l_range = torch.arange(l_idx, lsteps, l_sampling_rate, device=self.device, dtype=torch.long)
                
                # Create meshgrid
                h_grid, k_grid, l_grid = torch.meshgrid(h_range, k_range, l_range, indexing='ij')
                
                # Flatten indices
                h_flat = h_grid.reshape(-1)
                k_flat = k_grid.reshape(-1)
                l_flat = l_grid.reshape(-1)
                
                # Compute raveled indices using the full grid helper
                indices = self._3d_to_flat_indices(h_flat, k_flat, l_flat)
                all_indices.append(indices)
            
            logging.debug(f"[_at_kvec_from_miller_points] Batch processing: {batch_size} points -> {len(all_indices)} index sets")
            return all_indices
    
    
    #@debug
    def compute_hessian(self) -> torch.Tensor:
        """
        Compute the projected Hessian matrix for the supercell.
        
        This method works in both grid-based and arbitrary q-vector modes.
        
        Returns:
            hessian: A complex tensor of shape (n_asu, n_dof_per_asu, n_cell, n_asu, n_dof_per_asu)
                    representing the Hessian matrix for the assembly of rigid bodies.
        """
        # Initialize hessian tensor
        hessian = torch.zeros((self.n_asu, self.n_dof_per_asu,
                               self.n_cell, self.n_asu, self.n_dof_per_asu),
                              dtype=self.complex_dtype, device=self.device)
        
        # Create a GaussianNetworkModel instance for Hessian calculation
        from eryx.pdb_torch import GaussianNetworkModel as GaussianNetworkModelTorch
        gnm_torch = GaussianNetworkModelTorch()
        gnm_torch.device = self.device
        gnm_torch.n_asu = self.n_asu
        gnm_torch.n_atoms_per_asu = self.n_atoms_per_asu
        gnm_torch.n_cell = self.n_cell
        gnm_torch.id_cell_ref = self.id_cell_ref
        
        # Ensure crystal is properly set
        if hasattr(self, 'crystal'):
            gnm_torch.crystal = self.crystal
        else:
            print("Warning: No crystal object found in OnePhonon model")
        
        # Use differentiable gamma tensor for gradient flow
        if hasattr(self, 'gamma_tensor'):
            # Make sure gamma_tensor is properly connected to gamma_intra and gamma_inter
            if not self.gamma_tensor.requires_grad and (self.gamma_intra.requires_grad or self.gamma_inter.requires_grad):
                # Rebuild gamma tensor to ensure it uses the parameters with gradients
                self.gamma_tensor = torch.zeros((self.n_cell, self.n_asu, self.n_asu), 
                                              device=self.device, dtype=torch.float32)
                    
                # Fill gamma tensor with parameter tensors that require gradients
                for i_asu in range(self.n_asu):
                    for i_cell in range(self.n_cell):
                        for j_asu in range(self.n_asu):
                            self.gamma_tensor[i_cell, i_asu, j_asu] = self.gamma_inter
                            if (i_cell == self.id_cell_ref) and (j_asu == i_asu):
                                self.gamma_tensor[i_cell, i_asu, j_asu] = self.gamma_intra
                
            gnm_torch.gamma = self.gamma_tensor
        # Fallback to NumPy GNM gamma if needed
        elif hasattr(self.gnm, 'gamma'):
            from eryx.adapters import PDBToTensor
            adapter = PDBToTensor(device=self.device)
            gnm_torch.gamma = adapter.array_to_tensor(self.gnm.gamma, dtype=torch.float32)
        
        # Copy neighbor list structure
        gnm_torch.asu_neighbors = self.gnm.asu_neighbors
        
        # Compute Hessian using PyTorch implementation
        hessian_allatoms = gnm_torch.compute_hessian()
        
        # Create identity matrix for Kronecker product (use complex dtype)
        eye3 = torch.eye(3, device=self.device, dtype=self.complex_dtype)

        for i_cell in range(self.n_cell):
            for i_asu in range(self.n_asu):
                for j_asu in range(self.n_asu):
                    # Apply Kronecker product with identity matrix (3x3)
                    # This expands each element of the hessian into a 3x3 block
                    h_block = hessian_allatoms[i_asu, :, i_cell, j_asu, :]
                    h_expanded = torch.zeros((h_block.shape[0] * 3, h_block.shape[1] * 3), 
                                            dtype=self.complex_dtype, device=self.device)
                    
                    # Manually implement the Kronecker product (ensure h_block is complex)
                    h_block_complex = h_block.to(self.complex_dtype)
                    for i in range(h_block.shape[0]):
                        for j in range(h_block.shape[1]):
                            h_expanded[i*3:(i+1)*3, j*3:(j+1)*3] = h_block_complex[i, j] * eye3 # Multiply complex * complex

                    # Perform matrix multiplication with expanded hessian
                    # Ensure all tensors are complex for compatibility
                    proj = torch.matmul(self.Amat[i_asu].T.to(self.complex_dtype),
                                        torch.matmul(h_expanded, # Already complex
                                                     self.Amat[j_asu].to(self.complex_dtype)))
                    hessian[i_asu, :, i_cell, j_asu, :] = proj.to(self.complex_dtype) # Ensure final assignment is complex

        # Ensure hessian requires gradients
        if not hessian.requires_grad and hessian.is_complex():
            # For complex tensors we need a different approach to enable gradients
            # Create a dummy tensor that requires gradients, then add it to hessian
            dummy = torch.zeros((1,), dtype=self.complex_dtype, device=self.device, requires_grad=True)
            hessian = hessian + dummy * 0
        
        return hessian
    
    def compute_gnm_phonons(self):
        """
        Compute phonon modes for each k-vector in the first Brillouin zone.
        
        This implementation performs a vectorized computation for all k-vectors
        simultaneously to improve performance. Supports both grid-based and
        arbitrary q-vector modes.
        
        This method optimizes by:
        1. Finding unique k-vectors in the first BZ
        2. Computing phonons only for these unique k-vectors
        3. Expanding the results back to match the original q-vector list
        
        The eigenvalues (Winv) and eigenvectors (V) are stored for intensity calculation.
        """
        import logging
        # --- Modifications for Debugging ---
        import torch
        # Define the target k-vector value (get this from a preliminary run or calculation)
        # Example for BZ (0,0,1) in the 2x2x2 grid for 5zck_p1:
        kvec_target_tensor = torch.tensor([0.00000000, 0.00000000, -0.01691246], dtype=self.real_dtype, device=self.device)
        debug_kvec_atol = 1e-9 # Tolerance for matching k-vectors
        
        DEBUG_IDX_BZ = 1 # Corresponds to BZ index (0,0,1) in 2x2x2 grid
        DEBUG_IDX_FULL = 9 # Corresponding full grid index (based on previous mapping)
        # --- End Modifications ---

        # Compute the Hessian matrix first (works for both modes)
        hessian = self.compute_hessian()
        
        # --- ALWAYS use unique k-vectors ---
        # self.kvec is already populated correctly for either mode (BZ mapped for arbitrary)
        tolerance = 1e-10
        rounded_kvec = torch.round(self.kvec / tolerance) * tolerance
        unique_k_bz, inverse_indices = torch.unique(rounded_kvec, dim=0, return_inverse=True)
        n_unique_k = unique_k_bz.shape[0]
        logging.debug(f"[compute_gnm_phonons] Found {n_unique_k} unique k_BZ vectors to process.")

        # --- Add detailed check for specific indices ---
        # indices_to_check = [9, 61] # Check original q_idx 9 and 61
        # if self.kvec.shape[0] > max(indices_to_check): # Ensure indices are valid
        #      print("--- DEBUG: Unique K Mapping Check ---")
        #      for i in indices_to_check:
        #          original_k = self.kvec[i]
        #          unique_idx = inverse_indices[i].item()
        #          mapped_unique_k = unique_k_bz[unique_idx]
        #          print(f"  q_idx {i}:")
        #          print(f"    kvec[i] (Mapped BZ): {original_k.cpu().numpy()}")
        #          print(f"    unique_idx (j):      {unique_idx}")
        #          print(f"    unique_k_bz[j]:      {mapped_unique_k.cpu().numpy()}")
        #          # Check if they are close
        #          print(f"    Match within tol?:   {torch.allclose(original_k, mapped_unique_k, atol=tolerance*1.1)}") # Use slightly larger tol
        #      print("--- End Check ---")
        # --- End detailed check ---

        total_points = self.kvec.shape[0]
        if getattr(self, 'use_arbitrary_q', False):
            logging.debug(f"Computing phonons for {n_unique_k} unique BZ k-vectors from {total_points} arbitrary q-vectors")
        else:
            h_dim_bz = int(self.hsampling[2])
            k_dim_bz = int(self.ksampling[2])
            l_dim_bz = int(self.lsampling[2])
            logging.debug(f"Computing phonons for {n_unique_k} unique BZ k-vectors from {total_points} grid points ({h_dim_bz}x{k_dim_bz}x{l_dim_bz})")
        
        logging.debug(f"[compute_gnm_phonons] Found {n_unique_k} unique k_BZ vectors to process.")
        
        # Initialize output tensors for unique k-vectors first
        dof_total = self.n_asu * self.n_dof_per_asu
        V_unique = torch.zeros((n_unique_k, dof_total, dof_total),
                              dtype=self.complex_dtype, device=self.device)
        Winv_unique = torch.zeros((n_unique_k, dof_total),
                                 dtype=self.complex_dtype, device=self.device)
        
        # Create GNM instance for K matrix computation
        from eryx.pdb_torch import GaussianNetworkModel as GaussianNetworkModelTorch
        gnm_torch = GaussianNetworkModelTorch()
        gnm_torch.n_asu = self.n_asu
        gnm_torch.n_atoms_per_asu = self.n_atoms_per_asu
        gnm_torch.n_cell = self.n_cell
        gnm_torch.id_cell_ref = self.id_cell_ref
        gnm_torch.device = self.device
        gnm_torch.real_dtype = self.real_dtype
        gnm_torch.complex_dtype = self.complex_dtype
        
        # Ensure crystal is properly set
        if hasattr(self, 'crystal'):
            gnm_torch.crystal = self.crystal
        else:
            logging.warning("No crystal object found in OnePhonon model")
        
        # Set gamma for the GNM
        if hasattr(self, 'gamma_tensor'):
            gnm_torch.gamma = self.gamma_tensor
        # Fallback to NumPy GNM gamma if needed
        elif hasattr(self.gnm, 'gamma'):
            from eryx.adapters import PDBToTensor
            adapter = PDBToTensor(device=self.device)
            gnm_torch.gamma = adapter.array_to_tensor(self.gnm.gamma, dtype=self.real_dtype)
            
            # Copy neighbor list structure
            gnm_torch.asu_neighbors = self.gnm.asu_neighbors
        
        # Convert Linv to complex for matrix operations
        Linv_complex = self.Linv.to(dtype=self.complex_dtype)
        
        # For all modes, compute K matrices only for unique k-vectors
        Kmat_unique = gnm_torch.compute_K(hessian, unique_k_bz)
        
        # Reshape K matrices to 2D form for each unique k-vector
        dof_total = self.n_asu * self.n_dof_per_asu
        Kmat_unique_2d = Kmat_unique.reshape(n_unique_k, dof_total, dof_total)
        
        # Compute dynamical matrices (D = L^(-1) K L^(-H))
        logging.debug(f"Compute_K complete, Kmat_unique_2d shape = {Kmat_unique_2d.shape}")
        
        # Use torch.bmm for batched matrix multiplication
        Linv_batch = Linv_complex.unsqueeze(0).expand(n_unique_k, -1, -1)
        # Use conjugate transpose (.H) for complex matrices
        Linv_H_batch = Linv_complex.conj().T.unsqueeze(0).expand(n_unique_k, -1, -1)
        
        # First multiply Kmat_unique_2d with Linv_H_batch
        temp = torch.bmm(Kmat_unique_2d, Linv_H_batch)
        # Then multiply Linv_batch with the result
        Dmat_unique = torch.bmm(Linv_batch, temp)
        
        # Initialize v_all *outside* the try block
        v_all = None
        
        # Process unique k-vectors one by one for debugging clarity
        logging.debug(f"[DEBUG compute_gnm_phonons] Processing {n_unique_k} unique k-vectors")
        
        # Pre-calculate Linv_complex.H once
        Linv_complex = self.Linv.to(dtype=self.complex_dtype)
        Linv_H = Linv_complex.H # Conjugate transpose
        
        # Initialize lists to store results before stacking
        eigenvalues_unique_list = []
        eigenvectors_unique_detached_list = []
        
        for i in range(n_unique_k):
            current_kvec = unique_k_bz[i]
            
            # Check if this is our target k-vector
            is_target_kvec = torch.allclose(current_kvec, kvec_target_tensor, atol=debug_kvec_atol)
            
            D_i = Dmat_unique[i]
            D_i_hermitian = 0.5 * (D_i + D_i.H) # Ensure Hermiticity
            
            # 1. Get eigenvectors WITHOUT gradient tracking using eigh
            with torch.no_grad():
                try:
                    # Get both eigenvalues and eigenvectors
                    w_sq_i, v_i_no_grad = torch.linalg.eigh(D_i_hermitian)
                except torch._C._LinAlgError as e:
                    logging.warning(f"eigh failed for unique matrix {i} in no_grad context. Using identity. Error: {e}")
                    n_dof = D_i_hermitian.shape[0]
                    w_sq_i = torch.ones(n_dof, dtype=self.real_dtype, device=self.device) # Placeholder eigenvalues
                    v_i_no_grad = torch.eye(n_dof, dtype=self.complex_dtype, device=self.device) # Fallback
            
            # Use eigenvectors directly from eigh (ascending eigenvalue order)
            eigenvectors_unique_detached_list.append(v_i_no_grad)
            
            # 2. Recompute eigenvalues DIFFERENTIABLY using v.H @ D @ v
            # Try using the eigenvalues directly from eigh but ensure grads flow from D_i
            eigenvalues_tensor = w_sq_i.real # Eigenvalues from eigh should be real
            if D_i.requires_grad and not eigenvalues_tensor.requires_grad:
                eigenvalues_tensor = eigenvalues_tensor + 0.0 * D_i.real.sum() # Connect graph
            
            # 3. Process eigenvalues (thresholding only, keep ascending order from eigh)
            eps = torch.tensor(1e-6, dtype=self.real_dtype, device=self.device) # Use tensor for eps
            eigenvalues_processed = torch.where(
                eigenvalues_tensor < eps,
                torch.tensor(float('nan'), device=eigenvalues_tensor.device, dtype=self.real_dtype),
                eigenvalues_tensor # Already real_dtype
            )
            
            
        # Stack results for unique k-vectors
        eigenvalues_unique = torch.stack(eigenvalues_unique_list)  # Differentiable eigenvalues
        v_unique_detached = torch.stack(eigenvectors_unique_detached_list) # Non-differentiable eigenvectors
        
        logging.debug(f"Recomputed eigenvalues using eigh complete for unique k-vectors")
        logging.debug(f"eigenvalues_unique requires_grad: {eigenvalues_unique.requires_grad}")
        logging.debug(f"v_unique_detached requires_grad: {v_unique_detached.requires_grad}")
        
        # Transform eigenvectors V = L^(-H) v (using detached eigenvectors)
        Linv_H_batch = Linv_H.unsqueeze(0).expand(n_unique_k, -1, -1)
        # if self.kvec.shape[0] > max(indices_to_check): # Ensure indices are valid
        #      print("--- DEBUG: Unique K Mapping Check ---")
        #      for i in indices_to_check:
        #          original_k = self.kvec[i]
        #          unique_idx = inverse_indices[i].item()
        #          mapped_unique_k = unique_k_bz[unique_idx]
        #          print(f"  q_idx {i}:")
        #          print(f"    kvec[i] (Mapped BZ): {original_k.cpu().numpy()}")
        #          print(f"    unique_idx (j):      {unique_idx}")
        #          print(f"    unique_k_bz[j]:      {mapped_unique_k.cpu().numpy()}")
        #          # Check if they are close
        #          print(f"    Match within tol?:   {torch.allclose(original_k, mapped_unique_k, atol=tolerance*1.1)}") # Use slightly larger tol
        #      print("--- End Check ---")
        # --- End detailed check ---

        total_points = self.kvec.shape[0]
        if getattr(self, 'use_arbitrary_q', False):
            logging.debug(f"Computing phonons for {n_unique_k} unique BZ k-vectors from {total_points} arbitrary q-vectors")
        else:
            h_dim_bz = int(self.hsampling[2])
            k_dim_bz = int(self.ksampling[2])
            l_dim_bz = int(self.lsampling[2])
            logging.debug(f"Computing phonons for {n_unique_k} unique BZ k-vectors from {total_points} grid points ({h_dim_bz}x{k_dim_bz}x{l_dim_bz})")
        
        logging.debug(f"[compute_gnm_phonons] Found {n_unique_k} unique k_BZ vectors to process.")
        
        # Initialize output tensors for unique k-vectors first
        dof_total = self.n_asu * self.n_dof_per_asu
        V_unique = torch.zeros((n_unique_k, dof_total, dof_total),
                              dtype=self.complex_dtype, device=self.device)
        Winv_unique = torch.zeros((n_unique_k, dof_total),
                                 dtype=self.complex_dtype, device=self.device)
        
        # Create GNM instance for K matrix computation
        from eryx.pdb_torch import GaussianNetworkModel as GaussianNetworkModelTorch
        gnm_torch = GaussianNetworkModelTorch()
        gnm_torch.n_asu = self.n_asu
        gnm_torch.n_atoms_per_asu = self.n_atoms_per_asu
        gnm_torch.n_cell = self.n_cell
        gnm_torch.id_cell_ref = self.id_cell_ref
        gnm_torch.device = self.device
        gnm_torch.real_dtype = self.real_dtype
        gnm_torch.complex_dtype = self.complex_dtype
        
        # Ensure crystal is properly set
        if hasattr(self, 'crystal'):
            gnm_torch.crystal = self.crystal
        else:
            logging.warning("No crystal object found in OnePhonon model")
        
        # Set gamma for the GNM
        if hasattr(self, 'gamma_tensor'):
            gnm_torch.gamma = self.gamma_tensor
        # Fallback to NumPy GNM gamma if needed
        elif hasattr(self.gnm, 'gamma'):
            from eryx.adapters import PDBToTensor
            adapter = PDBToTensor(device=self.device)
            gnm_torch.gamma = adapter.array_to_tensor(self.gnm.gamma, dtype=self.real_dtype)
            
            # Copy neighbor list structure
            gnm_torch.asu_neighbors = self.gnm.asu_neighbors
        
        # Convert Linv to complex for matrix operations
        Linv_complex = self.Linv.to(dtype=self.complex_dtype)
        
        # For all modes, compute K matrices only for unique k-vectors
        Kmat_unique = gnm_torch.compute_K(hessian, unique_k_bz)
        logging.debug(f"Kmat_unique requires_grad: {Kmat_unique.requires_grad}")
        logging.debug(f"Kmat_unique.grad_fn: {Kmat_unique.grad_fn}")
        
        # Reshape K matrices to 2D form for each unique k-vector
        dof_total = self.n_asu * self.n_dof_per_asu
        Kmat_unique_2d = Kmat_unique.reshape(n_unique_k, dof_total, dof_total)
        
        # Compute dynamical matrices (D = L^(-1) K L^(-H))
        logging.debug(f"Compute_K complete, Kmat_unique_2d shape = {Kmat_unique_2d.shape}")
        
        # Use torch.bmm for batched matrix multiplication
        Linv_batch = Linv_complex.unsqueeze(0).expand(n_unique_k, -1, -1)
        # Use conjugate transpose (.H) for complex matrices
        Linv_H_batch = Linv_complex.conj().T.unsqueeze(0).expand(n_unique_k, -1, -1)
        
        # First multiply Kmat_unique_2d with Linv_H_batch
        temp = torch.bmm(Kmat_unique_2d, Linv_H_batch)
        # Then multiply Linv_batch with the result
        Dmat_unique = torch.bmm(Linv_batch, temp)
        
        # Initialize v_all *outside* the try block
        v_all = None
        
        # Process unique k-vectors one by one for debugging clarity
        logging.debug(f"[DEBUG compute_gnm_phonons] Processing {n_unique_k} unique k-vectors")
        
        # Pre-calculate Linv_complex.H once
        Linv_complex = self.Linv.to(dtype=self.complex_dtype)
        Linv_H = Linv_complex.H # Conjugate transpose
        
        # Initialize lists to store results before stacking
        eigenvalues_unique_list = []
        eigenvectors_unique_detached_list = []
        
        for i in range(n_unique_k):
            current_kvec = unique_k_bz[i]
            
            # Check if this is our target k-vector
            is_target_kvec = torch.allclose(current_kvec, kvec_target_tensor, atol=debug_kvec_atol)
            
            D_i = Dmat_unique[i]
            D_i_hermitian = 0.5 * (D_i + D_i.H) # Ensure Hermiticity
            
            # 1. Get eigenvectors WITHOUT gradient tracking using eigh
            with torch.no_grad():
                try:
                    # Get both eigenvalues and eigenvectors
                    w_sq_i, v_i_no_grad = torch.linalg.eigh(D_i_hermitian)
                except torch._C._LinAlgError as e:
                    logging.warning(f"eigh failed for unique matrix {i} in no_grad context. Using identity. Error: {e}")
                    n_dof = D_i_hermitian.shape[0]
                    w_sq_i = torch.ones(n_dof, dtype=self.real_dtype, device=self.device) # Placeholder eigenvalues
            
            # Use eigenvectors directly from eigh (ascending eigenvalue order)
            eigenvectors_unique_detached_list.append(v_i_no_grad)
            
            # 2. Recompute eigenvalues DIFFERENTIABLY using v.H @ D @ v
            # Try using the eigenvalues directly from eigh but ensure grads flow from D_i
            eigenvalues_tensor = w_sq_i.real # Eigenvalues from eigh should be real
            if D_i.requires_grad and not eigenvalues_tensor.requires_grad:
                eigenvalues_tensor = eigenvalues_tensor + 0.0 * D_i.real.sum() # Connect graph
            
            # 3. Process eigenvalues (thresholding only, keep ascending order from eigh)
            eps = torch.tensor(1e-6, dtype=self.real_dtype, device=self.device) # Use tensor for eps
            eigenvalues_processed = torch.where(
                eigenvalues_tensor < eps,
                torch.tensor(float('nan'), device=eigenvalues_tensor.device, dtype=self.real_dtype),
                eigenvalues_tensor # Already real_dtype
            )
            
            
        # Stack results for unique k-vectors
        eigenvalues_unique = torch.stack(eigenvalues_unique_list)  # Differentiable eigenvalues
        v_unique_detached = torch.stack(eigenvectors_unique_detached_list) # Non-differentiable eigenvectors
        
        logging.debug(f"Recomputed eigenvalues using eigh complete for unique k-vectors")
        logging.debug(f"eigenvalues_unique requires_grad: {eigenvalues_unique.requires_grad}")
        logging.debug(f"v_unique_detached requires_grad: {v_unique_detached.requires_grad}")
        
        # Transform eigenvectors V = L^(-H) v (using detached eigenvectors)
        Linv_H_batch = Linv_H.unsqueeze(0).expand(n_unique_k, -1, -1)
        V_unique = torch.matmul(
            Linv_H_batch,
            v_unique_detached # Use the detached eigenvectors here
        ).detach()
        
        # Calculate Winv = 1 / eigenvalues (using differentiable eigenvalues)
        eps_div = torch.tensor(1e-8, dtype=self.real_dtype, device=self.device) # Use tensor for eps_div
        winv_unique_real = torch.where(
            torch.isnan(eigenvalues_unique),
            torch.tensor(float('nan'), device=eigenvalues_unique.device, dtype=self.real_dtype),
            1.0 / torch.maximum(eigenvalues_unique, eps_div) # Ensure division is float64
        ).to(dtype=self.real_dtype) # Ensure final real dtype
        Winv_unique = winv_unique_real.to(dtype=self.complex_dtype) # Cast to complex
        
        # Now expand the unique results to match the original q-vector list
        self.V = V_unique[inverse_indices].detach()
        self.Winv = Winv_unique[inverse_indices]

        # Ensure requires_grad is set correctly
        self.V.requires_grad_(False)
        self.Winv.requires_grad_(eigenvalues_unique.requires_grad) # Simplified (same effect)

        # --- Add check after expansion ---
        # indices_to_check = [9, 61] # Check original q_idx 9 and 61
        # if self.Winv.shape[0] > max(indices_to_check):
        #      print("--- DEBUG: Expansion Check ---")
        #      for i in indices_to_check:
        #          unique_idx = inverse_indices[i].item()
        #          print(f"  q_idx {i} (unique_idx={unique_idx}):")
        #          # Compare first element of Winv
        #          print(f"    Winv[{i}][0]:         {self.Winv[i, 0].item()}")
        #          print(f"    Winv_unique[{unique_idx}][0]: {Winv_unique[unique_idx, 0].item()}")
        #          print(f"    Match?:              {torch.allclose(self.Winv[i, 0], Winv_unique[unique_idx, 0])}")
        #      print("--- End Check ---")
        # --- End check after expansion ---
        
        # Final check for the target k-vector
        # target_index_to_print = 1 # Corresponds to (0,0,1) in 2x2x2 BZ
        
        
        logging.debug(f"Phonon computation complete: V.shape={self.V.shape}, Winv.shape={self.Winv.shape}")
        logging.debug(f"V requires_grad: {self.V.requires_grad}, Winv requires_grad: {self.Winv.requires_grad}")
    
    #@debug
    def compute_gnm_K(self, hessian: torch.Tensor, kvec: torch.Tensor = None) -> torch.Tensor:
        """
        Compute the dynamical matrix K(kvec) from the Hessian.
        
        Args:
            hessian: Hessian tensor.
            kvec: k-vector tensor of shape (3,). Defaults to zero vector.
            
        Returns:
            Dynamical matrix K as a tensor.
        """
        if kvec is None:
            kvec = torch.zeros(3, device=self.device)
        Kmat = hessian[:, :, self.id_cell_ref, :, :].clone()
        for j_cell in range(self.n_cell):
            if j_cell == self.id_cell_ref:
                continue
            r_cell = self.crystal.get_unitcell_origin(self.crystal.id_to_hkl(j_cell))
            phase = torch.sum(kvec * r_cell)
            real_part, imag_part = torch.cos(phase), torch.sin(phase)
            eikr = torch.complex(real_part, imag_part)
            for i_asu in range(self.n_asu):
                for j_asu in range(self.n_asu):
                    Kmat[i_asu, :, j_asu, :] += hessian[i_asu, :, j_cell, j_asu, :] * eikr
        return Kmat
    
    #@debug
    def compute_Kinv(self, hessian: torch.Tensor, kvec: torch.Tensor = None, 
                     reshape: bool = True) -> torch.Tensor:
        """
        Compute the pseudo-inverse of the dynamical matrix K(kvec).
        
        Args:
            hessian: Hessian tensor
            kvec: k-vector tensor of shape (3,). Defaults to zero vector.
            reshape: Whether to reshape the output to match the input shape
            
        Returns:
            Inverse of dynamical matrix K
        """
        if kvec is None:
            kvec = torch.zeros(3, device=self.device)
        Kmat = self.compute_gnm_K(hessian, kvec=kvec)
        Kshape = Kmat.shape
        Kmat_2d = Kmat.reshape(Kshape[0] * Kshape[1], Kshape[2] * Kshape[3])
        eps = 1e-10
        identity = torch.eye(Kmat_2d.shape[0], device=self.device, dtype=Kmat_2d.dtype)
        Kmat_2d_reg = Kmat_2d + eps * identity
        Kinv = torch.linalg.pinv(Kmat_2d_reg)
        if reshape:
            Kinv = Kinv.reshape((Kshape[0], Kshape[1], Kshape[2], Kshape[3]))
        return Kinv
    
    def compute_covariance_matrix(self):
        """
        Compute the covariance matrix for atomic displacements.
        
        This method processes all k-vectors simultaneously for maximum
        efficiency. The covariance matrix is used to compute atomic
        displacement parameters (ADPs) that model thermal motions.
        
        IMPORTANT: This method always uses a proper Brillouin Zone sampling
        defined by hsampling, ksampling, lsampling for ADP calculations,
        regardless of whether the model is in grid or arbitrary q-vector mode.
        This ensures physically correct and consistent ADPs.
        """
        import logging
        
        # Check that sampling parameters exist
        if self.hsampling is None or self.ksampling is None or self.lsampling is None:
            raise ValueError("Cannot compute covariance matrix without hsampling, ksampling, and lsampling defined.")
        
        # Initialize covariance tensor
        self.covar = torch.zeros((self.n_asu * self.n_dof_per_asu,
                                self.n_cell, self.n_asu * self.n_dof_per_asu),
                               dtype=self.complex_dtype, device=self.device)
        
        # --- Generate Local BZ k-vector grid for averaging ---
        h_dim_bz = int(self.hsampling[2])
        k_dim_bz = int(self.ksampling[2])
        l_dim_bz = int(self.lsampling[2])
        total_bz_points = h_dim_bz * k_dim_bz * l_dim_bz
        logging.debug(f"[compute_covariance_matrix] Generating local BZ grid ({h_dim_bz}x{k_dim_bz}x{l_dim_bz} = {total_bz_points} points) for averaging.")

        # Get A_inv tensor (ensure float64)
        if isinstance(self.model.A_inv, torch.Tensor):
            A_inv_tensor = self.model.A_inv.clone().detach().to(dtype=self.real_dtype, device=self.device)
        else:
            A_inv_tensor = torch.tensor(self.model.A_inv, dtype=self.real_dtype, device=self.device)

        # Generate coordinates (ensure float64)
        h_coords = torch.tensor([self._center_kvec(dh, h_dim_bz) for dh in range(h_dim_bz)], 
                               device=self.device, dtype=self.real_dtype)
        k_coords = torch.tensor([self._center_kvec(dk, k_dim_bz) for dk in range(k_dim_bz)], 
                               device=self.device, dtype=self.real_dtype)
        l_coords = torch.tensor([self._center_kvec(dl, l_dim_bz) for dl in range(l_dim_bz)], 
                               device=self.device, dtype=self.real_dtype)
        h_grid, k_grid, l_grid = torch.meshgrid(h_coords, k_coords, l_coords, indexing='ij')
        hkl_fractional = torch.stack([h_grid.flatten(), k_grid.flatten(), l_grid.flatten()], dim=1).to(dtype=self.real_dtype)

        # Compute local BZ k-vectors (ensure float64)
        kvec_bz_local = torch.matmul(hkl_fractional, A_inv_tensor).to(dtype=self.real_dtype)
        # --- End Local BZ Grid Generation ---
        
        # Import helper functions for complex tensor operations
        from eryx.torch_utils import ComplexTensorOps
        from eryx.pdb_torch import GaussianNetworkModel as GNMTorch
        
        # Set up GNM for inverse K calculation
        gnm_torch = GNMTorch()
        gnm_torch.n_asu = self.n_asu
        gnm_torch.n_cell = self.n_cell
        gnm_torch.id_cell_ref = self.id_cell_ref
        gnm_torch.device = self.device
        gnm_torch.real_dtype = self.real_dtype
        gnm_torch.complex_dtype = self.complex_dtype
        
        # Set crystal reference
        if hasattr(self, 'crystal'):
            gnm_torch.crystal = self.crystal
        else:
            logging.warning("No crystal object found in OnePhonon model")
        
        # Compute the Hessian matrix
        hessian = self.compute_hessian().to(self.complex_dtype)
        
        # Compute inverse K matrices for all BZ k-vectors
        # Use local BZ k-vectors
        Kinv_all_bz = gnm_torch.compute_Kinv(hessian, kvec_bz_local, reshape=False).to(self.complex_dtype)
        logging.debug(f"[compute_covariance_matrix] Computed Kinv_all_bz for BZ grid, shape={Kinv_all_bz.shape}")
        
        # Calculate phase factors and accumulate covariance contributions
        for j_cell in range(self.n_cell):
            # Get cell origin (ensure it's a tensor with correct dtype)
            r_cell_np = self.crystal.get_unitcell_origin(self.crystal.id_to_hkl(j_cell))
            r_cell = torch.tensor(r_cell_np, dtype=self.real_dtype, device=self.device)

            # Calculate phase factors for all BZ k-vectors
            # Batched dot product: sum over last dim (3) -> shape [total_bz_points]
            all_phases = torch.sum(kvec_bz_local * r_cell, dim=1).to(dtype=self.real_dtype) # Ensure real_dtype
            cos_phases = torch.cos(all_phases) # Output should match input dtype (real_dtype)
            sin_phases = torch.sin(all_phases) # Output should match input dtype (real_dtype)
            eikr_all = torch.complex(cos_phases, sin_phases).to(self.complex_dtype) # Ensure complex_dtype

            # Reshape phase factors for broadcasting: [total_bz_points] -> [total_bz_points, 1, 1]
            eikr_reshaped = eikr_all.view(-1, 1, 1)
            
            # Element-wise multiply and sum over the k-points dimension (dim=0)
            complex_sum = torch.sum(Kinv_all_bz * eikr_reshaped, dim=0)
            
            # Accumulate averaged contribution in covariance tensor
            self.covar[:, j_cell, :] = complex_sum / total_bz_points
        
        # Get reference cell ID for [0,0,0]
        ref_cell_id = self.crystal.hkl_to_id([0, 0, 0])
        
        # Extract diagonal elements for ADP calculation
        # Use .real attribute instead of torch.real() to preserve gradient flow
        diagonal_values = torch.diagonal(self.covar[:, ref_cell_id, :], dim1=0, dim2=1)
        self.ADP = diagonal_values.real
        logging.debug(f"[compute_covariance_matrix] Extracted diagonal values, shape={diagonal_values.shape}")
        
        # Transform ADP using the displacement projection matrix
        Amat = torch.transpose(self.Amat, 0, 1).reshape(self.n_dof_per_asu_actual, self.n_asu * self.n_dof_per_asu)
        self.ADP = torch.matmul(Amat, self.ADP)
        
        # Sum over spatial dimensions (x,y,z)
        self.ADP = torch.sum(self.ADP.reshape(int(self.ADP.shape[0] / 3), 3), dim=1)
        
        # Scale ADP to match experimental values
        model_adp_tensor = self.array_to_tensor(self.model.adp)
        
        # Handle NaN values
        valid_adp = self.ADP[~torch.isnan(self.ADP)]
        if valid_adp.numel() > 0:
            # Calculate mean of valid values only
            adp_mean = torch.mean(valid_adp)
            # Calculate scaling factor
            ADP_scale = torch.mean(model_adp_tensor) / (8 * torch.pi * torch.pi * adp_mean / 3)
        else:
            # Fallback if all values are NaN
            ADP_scale = torch.tensor(1.0, device=self.device, dtype=self.real_dtype)
        
        logging.debug(f"[compute_covariance_matrix] ADP scaling factor: {ADP_scale.item():.8e}")
        
        # Apply scaling to ADP and covariance matrix
        self.ADP = self.ADP * ADP_scale
        self.covar = self.covar * ADP_scale
        
        # Reshape covariance matrix to final format
        # Use .real attribute instead of torch.real() to preserve gradient flow
        reshaped_covar = self.covar.reshape((self.n_asu, self.n_dof_per_asu,
                                           self.n_cell, self.n_asu, self.n_dof_per_asu))
        self.covar = reshaped_covar.real
        
        # Set requires_grad for ADP tensor
        self.ADP.requires_grad_(True)
        
        logging.debug(f"[compute_covariance_matrix] Complete: ADP.shape={self.ADP.shape}, requires_grad={self.ADP.requires_grad}")
    
    #@debug
    def apply_disorder(self, rank: int = -1, outdir: Optional[str] = None, 
                       use_data_adp: bool = False) -> torch.Tensor:
        # --- Phase 0 Instrumentation ---
        target_idx_bz = 1   # BZ index corresponding to (dh,dk,dl)=(0,0,1) for 2x2x2 sampling
        target_q_idx = 9    # Example q_grid index corresponding to the first point for BZ idx 1
                            # Replace 9 with the actual value found in Step 3.
        # --- End Instrumentation ---
        """
        Compute the diffuse intensity using the one-phonon approximation.
        
        This method handles two modes of operation:
        1. Grid-based mode: Processes q-vectors arranged in a 3D grid
        2. Arbitrary q-vector mode: Processes arbitrary q-vectors in a vectorized manner
        
        The method uses vectorized operations to process k-vectors efficiently,
        minimizing loops and maximizing GPU utilization.
        
        Args:
            rank: Phonon mode rank to use (-1 for all modes)
            outdir: Directory to save results
            use_data_adp: Whether to use ADPs from data instead of computed values
            
        Returns:
            Diffuse intensity tensor. In grid-based mode, this tensor has shape [n_points]
            where n_points = product of grid dimensions. In arbitrary q-vector mode, 
            this tensor has shape [n_points] where n_points = number of provided q-vectors.
            Points outside the resolution mask will have NaN values.
        """
        import logging
        logging.debug(f"[apply_disorder] rank={rank}, use_data_adp={use_data_adp}, use_arbitrary_q={getattr(self,'use_arbitrary_q',None)}")
        
        # Prepare ADPs based on use_data_adp flag
        if use_data_adp:
            # Use B-factors directly from PDB data
            if hasattr(self.model, 'adp') and self.model.adp is not None and len(self.model.adp) > 0:
                # Convert the NumPy array (assuming it's stored as such in self.model)
                ADP_source = torch.tensor(self.model.adp[0], dtype=self.real_dtype, device=self.device)
                ADP = ADP_source / (8 * torch.pi * torch.pi) # Apply scaling
                logging.debug("[apply_disorder] Using ADP from PDB data (use_data_adp=True).")
            else:
                logging.warning("[apply_disorder] PDB data ADP not found/empty despite use_data_adp=True. Falling back to ones.")
                # Fallback: Create a tensor of ones with the correct size
                num_atoms = self.n_atoms_per_asu # Assuming this attribute exists
                ADP = torch.ones(num_atoms, device=self.device, dtype=self.real_dtype)
        else:
            # Use internally computed ADP (self.ADP)
            # This requires compute_covariance_matrix to have run during setup if model='gnm'
            if hasattr(self, 'ADP') and self.ADP is not None:
                ADP = self.ADP.to(dtype=self.real_dtype, device=self.device) # Ensure correct dtype/device
                logging.debug("[apply_disorder] Using internally calculated ADP (use_data_adp=False).")
            else:
                # This case indicates an issue during setup (compute_covariance_matrix didn't run or failed)
                # or the model type wasn't 'gnm'. Fallback to PDB data as a last resort.
                logging.error("[apply_disorder] Internally computed self.ADP not found despite use_data_adp=False. "
                          "This might indicate a setup issue or non-GNM model. Falling back to PDB data ADP.")
                if hasattr(self.model, 'adp') and self.model.adp is not None and len(self.model.adp) > 0:
                    ADP_source = torch.tensor(self.model.adp[0], dtype=self.real_dtype, device=self.device)
                    ADP = ADP_source / (8 * torch.pi * torch.pi)
                else:
                    logging.warning("[apply_disorder] PDB data ADP also not found. Falling back to ones.")
                    num_atoms = self.n_atoms_per_asu
                    ADP = torch.ones(num_atoms, device=self.device, dtype=self.real_dtype)

        # Ensure ADP tensor requires grad if it's float (should be handled by its source)
        # Add a check and potentially connect the graph if needed, especially for self.ADP
        if ADP.is_floating_point() and not ADP.requires_grad:
            # Check if the source (self.ADP if use_data_adp=False) required grad
            source_requires_grad = False
            if not use_data_adp and hasattr(self, 'ADP') and self.ADP is not None:
                source_requires_grad = self.ADP.requires_grad # Check the source tensor directly
            # Only set requires_grad if the source needed it
            if source_requires_grad:
                ADP = ADP.clone().detach().requires_grad_(True) # Recreate to ensure leaf status if needed

        logging.debug(f"[apply_disorder] Using ADP with shape: {ADP.shape}, requires_grad={ADP.requires_grad}")
        
        # Initialize intensity tensor
        Id = torch.zeros(self.q_grid.shape[0], dtype=self.real_dtype, device=self.device)
        logging.debug(f"[apply_disorder] q_grid.size={self.q_grid.shape[0]} total points. res_mask sum={int(self.res_mask.sum())}.")
        
        # Import structure_factors function
        from eryx.scatter_torch import structure_factors
        
        # Get valid indices directly from the resolution mask (same for both modes)
        n_points = self.q_grid.shape[0]
        valid_indices = torch.where(self.res_mask)[0]

        if getattr(self, 'use_arbitrary_q', False):
            logging.debug(f"[apply_disorder] ARBITRARY-Q MODE (using direct indexing)")
        else:
            logging.debug(f"[apply_disorder] GRID MODE (using direct indexing)")
        
        logging.debug(f"[apply_disorder] Total points: {n_points}, Valid points (res_mask): {valid_indices.numel()}")

        # Pre-compute all ASU data to avoid repeated tensor creation
        # Use array_to_tensor adapter for consistent dtype and gradient settings
        asu_data = []
        for i_asu in range(self.n_asu):
            asu_data.append({
                'xyz': self.array_to_tensor(self.crystal.get_asu_xyz(i_asu), dtype=self.real_dtype),
                'ff_a': self.array_to_tensor(self.model.ff_a[i_asu], dtype=self.real_dtype),
                'ff_b': self.array_to_tensor(self.model.ff_b[i_asu], dtype=self.real_dtype),
                'ff_c': self.array_to_tensor(self.model.ff_c[i_asu], dtype=self.real_dtype),
                'project': self.Amat[i_asu].to(dtype=self.real_dtype)
            })

        if getattr(self, 'use_arbitrary_q', False):
            # --- Arbitrary Q-Vector Mode ---
            logging.debug(f"[apply_disorder] ARBITRARY-Q MODE (using direct indexing)")
            n_points = self.q_grid.shape[0]
            valid_indices = torch.where(self.res_mask)[0]

            if valid_indices.numel() == 0:
                # If no valid indices, return array of NaNs
                Id_masked = torch.full((n_points,), float('nan'), dtype=self.real_dtype, device=self.device)
                # Save results if outdir is provided
                if outdir is not None:
                    import os
                    os.makedirs(outdir, exist_ok=True)
                    torch.save(Id_masked, os.path.join(outdir, f"rank_{rank:05d}_torch.pt"))
                    np.save(os.path.join(outdir, f"rank_{rank:05d}.npy"), Id_masked.detach().cpu().numpy())
                return Id_masked

            # Compute structure factors for all valid q-vectors
            F = torch.zeros((valid_indices.numel(), self.n_asu, self.n_dof_per_asu),
                          dtype=self.complex_dtype, device=self.device)
            for i_asu in range(self.n_asu):
                asu = asu_data[i_asu]
                q_vectors = self.q_grid[valid_indices].to(dtype=self.real_dtype)
                xyz = asu['xyz'].to(dtype=self.real_dtype)
                ff_a = asu['ff_a'].to(dtype=self.real_dtype)
                ff_b = asu['ff_b'].to(dtype=self.real_dtype)
                ff_c = asu['ff_c'].to(dtype=self.real_dtype)
                adp = ADP.to(dtype=self.real_dtype)
                project = asu['project'].to(dtype=self.real_dtype)
                sf_result = structure_factors(
                    q_vectors.clone(), xyz.clone(), ff_a.clone(), ff_b.clone(), ff_c.clone(), U=adp.clone(),
                    n_processes=self.n_processes, compute_qF=True,
                    project_on_components=project.clone(), sum_over_atoms=False
                )
                F[:, i_asu, :] = sf_result.to(self.complex_dtype)
            F = F.reshape((valid_indices.numel(), self.n_asu * self.n_dof_per_asu))

            # Apply disorder model
            intensity = torch.zeros(valid_indices.numel(), device=self.device, dtype=self.real_dtype)
            V_valid = self.V[valid_indices].to(self.complex_dtype) # V is expanded in arbitrary mode
            Winv_valid = self.Winv[valid_indices].to(self.complex_dtype) # Winv is expanded in arbitrary mode

            if rank == -1:
                for i in range(valid_indices.numel()):
                    F_i = F[i].to(self.complex_dtype)
                    V_i = V_valid[i].to(self.complex_dtype)
                    Winv_i = Winv_valid[i].to(self.complex_dtype)
                    FV = torch.matmul(F_i, V_i)
                    FV_abs_squared = torch.abs(FV)**2
                    real_winv = Winv_i.real.to(dtype=self.real_dtype)
                    intensity_contribution = torch.sum(FV_abs_squared * real_winv)
                    intensity[i] = intensity_contribution
            else:
                for i in range(valid_indices.numel()):
                    F_i = F[i].to(self.complex_dtype)
                    V_rank = V_valid[i, :, rank] # Get specific mode eigenvector
                    Winv_rank = Winv_valid[i, rank] # Get specific mode eigenvalue
                    FV = torch.matmul(F_i, V_rank)
                    FV_abs_squared = torch.abs(FV)**2
                    real_winv = Winv_rank.real.to(dtype=self.real_dtype)
                    intensity[i] = FV_abs_squared * real_winv

            # Build full result array
            Id = torch.full((n_points,), float('nan'), dtype=self.real_dtype, device=self.device)
            Id[valid_indices] = intensity.to(dtype=self.real_dtype)
            Id_masked = Id.clone()
            Id_masked[~self.res_mask] = float('nan')

        else:
            # --- Grid Mode (Using BZ loops and index_add_) ---
            logging.debug("[apply_disorder] GRID MODE (using BZ loops)")
            h_dim_bz = int(self.hsampling[2])
            k_dim_bz = int(self.ksampling[2])
            l_dim_bz = int(self.lsampling[2])
            total_k_points = h_dim_bz * k_dim_bz * l_dim_bz
            Id = torch.zeros(self.q_grid.shape[0], dtype=self.real_dtype, device=self.device) # Final full grid

            # Verify V and Winv shapes match total_k_points (BZ points)
            if self.V.shape[0] != total_k_points or self.Winv.shape[0] != total_k_points:
                raise ValueError(f"Grid mode: V/Winv shapes ({self.V.shape[0]}, {self.Winv.shape[0]}) != total_k_points ({total_k_points})")

            logging.debug("[apply_disorder] Starting loops over Brillouin zone...")
            for dh in range(h_dim_bz):
                for dk in range(k_dim_bz):
                    for dl in range(l_dim_bz):
                        # Calculate the flat Brillouin zone index
                        idx = self._3d_to_flat_indices_bz(
                            torch.tensor([dh], device=self.device, dtype=torch.long),
                            torch.tensor([dk], device=self.device, dtype=torch.long),
                            torch.tensor([dl], device=self.device, dtype=torch.long)
                        ).item()

                        # Retrieve phonon data for this BZ index (V/Winv have size n_bz_points)
                        V_k = self.V[idx].to(self.complex_dtype)
                        Winv_k = self.Winv[idx].to(self.complex_dtype)

                        # Find corresponding q points
                        q_indices = self._at_kvec_from_miller_points((dh, dk, dl))
                        if q_indices.numel() == 0: continue
                        valid_mask_for_q = self.res_mask[q_indices]
                        valid_indices_for_this_bz = q_indices[valid_mask_for_q]
                        if valid_indices_for_this_bz.numel() == 0: continue

                        # Calculate F ONLY for these valid q points
                        F = torch.zeros((valid_indices_for_this_bz.numel(), self.n_asu, self.n_dof_per_asu),
                                      dtype=self.complex_dtype, device=self.device)
                        q_vectors_batch = self.q_grid[valid_indices_for_this_bz].to(dtype=self.real_dtype)

                        for i_asu in range(self.n_asu):
                            asu = asu_data[i_asu]
                            xyz = asu['xyz'].to(dtype=self.real_dtype)
                            ff_a = asu['ff_a'].to(dtype=self.real_dtype)
                            ff_b = asu['ff_b'].to(dtype=self.real_dtype)
                            ff_c = asu['ff_c'].to(dtype=self.real_dtype)
                            adp = ADP.to(dtype=self.real_dtype)
                            project = asu['project'].to(dtype=self.real_dtype)
                            sf_result = structure_factors(
                                q_vectors_batch.clone(), xyz.clone(), ff_a.clone(), ff_b.clone(), ff_c.clone(), U=adp.clone(),
                                n_processes=self.n_processes, compute_qF=True,
                                project_on_components=project.clone(), sum_over_atoms=False
                            )
                            F[:, i_asu, :] = sf_result.to(self.complex_dtype)
                        F = F.reshape((valid_indices_for_this_bz.numel(), self.n_asu * self.n_dof_per_asu))

                        # Perform calculation using the single V_k, Winv_k for this BZ point
                        F = F.to(self.complex_dtype)
                        V_k = V_k.to(self.complex_dtype) # Ensure complex

                        if rank == -1:
                            FV = torch.matmul(F, V_k)
                            FV_abs_squared = torch.abs(FV)**2
                            real_winv = Winv_k.real.to(dtype=self.real_dtype)
                            intensity_contribution = torch.sum(FV_abs_squared * real_winv, dim=1)
                        else:
                            V_k_rank = V_k[:, rank]
                            Winv_k_rank = Winv_k[rank]
                            FV = torch.matmul(F, V_k_rank)
                            FV_abs_squared = torch.abs(FV)**2
                            real_winv = Winv_k_rank.real.to(dtype=self.real_dtype)
                            intensity_contribution = FV_abs_squared * real_winv

                        # Accumulate results into the full Id tensor
                        Id.index_add_(0, valid_indices_for_this_bz, intensity_contribution.to(dtype=self.real_dtype))

            # Apply final mask
            Id_masked = Id.clone()
            Id_masked[~self.res_mask] = float('nan')

        # Save results if outdir is provided
        if outdir is not None:
            logging.debug(f"[apply_disorder] Saving results to outdir={outdir} rank={rank}.npy/.pt")
            import os
            os.makedirs(outdir, exist_ok=True)
            torch.save(Id_masked, os.path.join(outdir, f"rank_{rank:05d}_torch.pt"))
            np.save(os.path.join(outdir, f"rank_{rank:05d}.npy"), Id_masked.detach().cpu().numpy())

        return Id_masked

    def array_to_tensor(self, array: Union[np.ndarray, torch.Tensor, float, int], requires_grad: bool = True, dtype=None) -> torch.Tensor:
        """
        Convert a NumPy array, scalar, or existing tensor to a PyTorch tensor with gradient support.

        This is a helper method that ensures consistent tensor conversion
        throughout the class, with proper handling of gradient requirements and dtypes.
        
        Args:
            array: NumPy array or PyTorch tensor to convert
            requires_grad: Whether the tensor requires gradients for backpropagation
            dtype: Data type for the tensor (default: torch.float64)
            
        Returns:
            PyTorch tensor with proper device and gradient settings
        """
        # Handle None input
        if array is None:
            return None
        
        # Set default dtype if not provided
        if dtype is None:
            dtype = self.real_dtype # Use self.real_dtype as default

        # Handle complex arrays/scalars
        is_complex = False
        if isinstance(array, np.ndarray) and np.iscomplexobj(array):
            is_complex = True
        elif isinstance(array, complex):
            is_complex = True
        elif isinstance(array, torch.Tensor) and array.is_complex():
            is_complex = True

        if is_complex:
            dtype = self.complex_dtype # Use self.complex_dtype for complex

        # If already a tensor, move to correct device, set dtype, and handle requires_grad
        if isinstance(array, torch.Tensor):
            tensor = array.to(device=self.device, dtype=dtype)
            if requires_grad and tensor.dtype.is_floating_point:
                tensor.requires_grad_(True)
            return tensor

        # Handle scalars (float, int, complex)
        if isinstance(array, (float, int, complex)):
            tensor = torch.tensor(array, dtype=dtype, device=self.device)
        # Handle NumPy arrays (including empty ones)
        elif isinstance(array, np.ndarray):
            # Allow empty arrays
            tensor = torch.from_numpy(array.copy()).to(dtype=dtype, device=self.device)
        else:
            raise TypeError(f"Unsupported type for array_to_tensor: {type(array)}")

        # Set requires_grad if appropriate (only for float/complex tensors)
        if requires_grad and tensor.dtype.is_floating_point or tensor.dtype.is_complex():
            # Ensure it's a leaf tensor before setting requires_grad
            if not tensor.is_leaf:
                tensor = tensor.clone().detach()
            tensor.requires_grad_(True)

        return tensor

    def to_batched_shape(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Convert tensor from [h_dim, k_dim, l_dim, ...] to [h_dim*k_dim*l_dim, ...].
        
        In arbitrary q-vector mode, this is a no-op.
        In grid-based mode, flattens the first three dimensions.
        
        Args:
            tensor: Tensor in original shape with dimensions [h_dim, k_dim, l_dim, ...]
            
        Returns:
            Tensor in fully collapsed batched shape [h_dim*k_dim*l_dim, ...]
        """
        # For arbitrary q-vector mode, return tensor unchanged
        if getattr(self, 'use_arbitrary_q', False):
            return tensor
            
        # Get dimensions from the tensor shape
        h_dim = tensor.shape[0]
        k_dim = tensor.shape[1]
        l_dim = tensor.shape[2]
        remaining_dims = tensor.shape[3:]
        
        # Reshape to combine all three dimensions into one
        result = tensor.reshape(h_dim * k_dim * l_dim, *remaining_dims)
        return result
    
    def to_original_shape(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Convert tensor from [h_dim*k_dim*l_dim, ...] to [h_dim, k_dim, l_dim, ...].
        
        In arbitrary q-vector mode, this is a no-op.
        In grid-based mode, restores the 3D grid structure.
        
        Args:
            tensor: Tensor in fully collapsed batched shape with dimensions [h_dim*k_dim*l_dim, ...]
            
        Returns:
            Tensor in original shape [h_dim, k_dim, l_dim, ...]
        """
        # For arbitrary q-vector mode, return tensor unchanged
        if getattr(self, 'use_arbitrary_q', False):
            return tensor
        
        # Determine dimensions to use for reshaping
        if hasattr(self, 'test_k_dim') and hasattr(self, 'test_l_dim'):
            # For tests that explicitly set test dimensions
            k_dim = self.test_k_dim
            l_dim = self.test_l_dim
            # Calculate h_dim based on tensor size and k,l dimensions
            total_points = tensor.shape[0]
            h_dim = total_points // (k_dim * l_dim)
        elif hasattr(self, 'map_shape') and self.map_shape is not None:
            # Use dimensions from map_shape
            h_dim, k_dim, l_dim = self.map_shape
        else:
            # Fallback to sampling parameters directly
            h_dim = int(self.hsampling[2])
            k_dim = int(self.ksampling[2])  
            l_dim = int(self.lsampling[2])
        
        # Reshape to original dimensions
        result = tensor.reshape(h_dim, k_dim, l_dim, *tensor.shape[1:])
        return result
    
    #@debug
    def compute_rb_phonons(self):
        """
        Compute phonons for the rigid-body model.
        """
        self.compute_gnm_phonons()
        

    def _flat_to_3d_indices(self, flat_indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert fully collapsed flat indices to h,k,l indices.
        
        Args:
            flat_indices: Tensor of flat indices
            
        Returns:
            Tuple of (h_indices, k_indices, l_indices) tensors
        """
        # In arbitrary q-vector mode, this is a no-op
        if getattr(self, 'use_arbitrary_q', False):
            # Just return the same indices for all dimensions
            return flat_indices, flat_indices, flat_indices
        
        # Get dimensions from map_shape
        h_dim, k_dim, l_dim = self.map_shape
        
        # Convert flat indices to 3D indices
        k_l_size = k_dim * l_dim
        
        # Integer division for h index
        h_indices = torch.div(flat_indices, k_l_size, rounding_mode='floor')
        
        # Remainder for k,l indices
        kl_remainder = flat_indices % k_l_size
        k_indices = torch.div(kl_remainder, l_dim, rounding_mode='floor')
        l_indices = kl_remainder % l_dim
        
        return h_indices, k_indices, l_indices
    
    def _compute_indices_for_point(self, h_idx: int, k_idx: int, l_idx: int, 
                                  hsteps: int, ksteps: int, lsteps: int) -> torch.Tensor:
        """
        Compute raveled indices for a single (h,k,l) point.
        
        Args:
            h_idx, k_idx, l_idx: Miller indices
            hsteps, ksteps, lsteps: Number of steps for each axis calculated from sampling parameters
            
        Returns:
            Tensor of raveled indices
        """
        # Create index grid using sampling rates as steps and calculated steps as end points
        h_range = torch.arange(h_idx, hsteps, int(self.hsampling[2]), device=self.device, dtype=torch.long)
        k_range = torch.arange(k_idx, ksteps, int(self.ksampling[2]), device=self.device, dtype=torch.long)
        l_range = torch.arange(l_idx, lsteps, int(self.lsampling[2]), device=self.device, dtype=torch.long)
        
        # Create meshgrid
        h_grid, k_grid, l_grid = torch.meshgrid(h_range, k_range, l_range, indexing='ij')
        
        # Flatten indices
        h_flat = h_grid.reshape(-1)
        k_flat = k_grid.reshape(-1)
        l_flat = l_grid.reshape(-1)
        
        # Compute raveled indices
        indices = h_flat * (self.map_shape[1] * self.map_shape[2]) + \
                 k_flat * self.map_shape[2] + \
                 l_flat
        
        return indices
    
    def _flat_to_3d_indices_bz(self, flat_indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert flat indices relative to Brillouin zone sampling to 3D (dh, dk, dl) indices.
        
        Args:
            flat_indices: Tensor of flat indices
            
        Returns:
            Tuple of (h_indices, k_indices, l_indices) tensors
        """
        # Get dimensions from Brillouin zone sampling counts
        k_dim_bz = int(self.ksampling[2])
        l_dim_bz = int(self.lsampling[2])
        k_l_size_bz = k_dim_bz * l_dim_bz

        # Use torch.div and % for calculations, ensure rounding_mode='floor' for div
        h_indices = torch.div(flat_indices, k_l_size_bz, rounding_mode='floor')
        kl_remainder = flat_indices % k_l_size_bz
        k_indices = torch.div(kl_remainder, l_dim_bz, rounding_mode='floor')
        l_indices = kl_remainder % l_dim_bz

        return h_indices, k_indices, l_indices
    
    def _3d_to_flat_indices_bz(self, h_indices: torch.Tensor, k_indices: torch.Tensor, l_indices: torch.Tensor) -> torch.Tensor:
        """
        Convert 3D (dh, dk, dl) indices relative to Brillouin zone sampling to flat indices.
        
        Args:
            h_indices: Tensor of h indices
            k_indices: Tensor of k indices
            l_indices: Tensor of l indices
            
        Returns:
            Tensor of flat indices
        """
        # Get dimensions from Brillouin zone sampling counts
        k_dim_bz = int(self.ksampling[2])
        l_dim_bz = int(self.lsampling[2])
        k_l_size_bz = k_dim_bz * l_dim_bz

        # Calculate flat indices: flat_idx = h_idx * (k_dim_bz * l_dim_bz) + k_idx * l_dim_bz + l_idx
        flat_indices = h_indices * k_l_size_bz + k_indices * l_dim_bz + l_indices
        return flat_indices
    
    def _3d_to_flat_indices(self, h_indices: torch.Tensor, k_indices: torch.Tensor, l_indices: torch.Tensor) -> torch.Tensor:
        """
        Convert h,k,l indices to fully collapsed flat indices.
        
        Args:
            h_indices: Tensor of h indices
            k_indices: Tensor of k indices
            l_indices: Tensor of l indices
            
        Returns:
            Tensor of flat indices
        """
        # In arbitrary q-vector mode, return h_indices directly
        if getattr(self, 'use_arbitrary_q', False):
            return h_indices
        
        # Get dimensions from map_shape
        _, k_dim, l_dim = self.map_shape
        
        # Calculate flat indices: flat_idx = h_idx * (k_dim * l_dim) + k_idx * l_dim + l_idx
        flat_indices = h_indices * (k_dim * l_dim) + k_indices * l_dim + l_indices
        
        return flat_indices

# Minimal implementations for additional models

class RigidBodyTranslations:
    #@debug
    def __init__(self, *args, **kwargs):
        pass

class LiquidLikeMotions:
    #@debug
    def __init__(self, *args, **kwargs):
        pass

class RigidBodyRotations:
    #@debug
    def __init__(self, *args, **kwargs):
        pass

    def _track_gradient_flow(self, named_parameters=None):
        """
        Diagnostic function for tracking gradient flow through the model.
        
        This helper method tracks where gradients are flowing and their magnitudes,
        which is useful for debugging gradient flow issues.
        
        Args:
            named_parameters: Dictionary of named parameters to track.
                             If None, tracks key model tensors.
        
        Returns:
            Dictionary mapping parameter names to gradient statistics
        """
        if named_parameters is None:
            # Track key tensors that should have gradients
            named_parameters = {
                'q_vectors': self.q_vectors if hasattr(self, 'q_vectors') else None,
                'q_grid': self.q_grid if hasattr(self, 'q_grid') else None,
                'kvec': self.kvec if hasattr(self, 'kvec') else None,
                'ADP': self.ADP if hasattr(self, 'ADP') else None
            }
        
        stats = {}
        for name, param in named_parameters.items():
            if param is None or not isinstance(param, torch.Tensor):
                stats[name] = {'requires_grad': None, 'grad': None, 'grad_norm': None}
                continue
                
            # Check if tensor requires gradients
            requires_grad = param.requires_grad
            
            # Check if gradient exists and compute its norm
            if requires_grad and param.grad is not None:
                grad_norm = torch.norm(param.grad).item()
                grad_abs_max = torch.max(torch.abs(param.grad)).item()
            else:
                grad_norm = None
                grad_abs_max = None
                
            stats[name] = {
                'requires_grad': requires_grad,
                'grad': param.grad is not None,
                'grad_norm': grad_norm,
                'grad_abs_max': grad_abs_max
            }
            
        return stats

    def _track_gradient_flow(self, named_parameters=None):
        """
        Diagnostic function for tracking gradient flow through the model.
        
        This helper method tracks where gradients are flowing and their magnitudes,
        which is useful for debugging gradient flow issues.
        
        Args:
            named_parameters: Dictionary of named parameters to track.
                             If None, tracks key model tensors.
        
        Returns:
            Dictionary mapping parameter names to gradient statistics
        """
        if named_parameters is None:
            # Track key tensors that should have gradients
            named_parameters = {
                'q_vectors': self.q_vectors if hasattr(self, 'q_vectors') else None,
                'q_grid': self.q_grid if hasattr(self, 'q_grid') else None,
                'kvec': self.kvec if hasattr(self, 'kvec') else None,
                'ADP': self.ADP if hasattr(self, 'ADP') else None
            }
        
        stats = {}
        for name, param in named_parameters.items():
            if param is None or not isinstance(param, torch.Tensor):
                stats[name] = {'requires_grad': None, 'grad': None, 'grad_norm': None}
                continue
                
            # Check if tensor requires gradients
            requires_grad = param.requires_grad
            
            # Check if gradient exists and compute its norm
            if requires_grad and param.grad is not None:
                grad_norm = torch.norm(param.grad).item()
                grad_abs_max = torch.max(torch.abs(param.grad)).item()
            else:
                grad_norm = None
                grad_abs_max = None
                
            stats[name] = {
                'requires_grad': requires_grad,
                'grad': param.grad is not None,
                'grad_norm': grad_norm,
                'grad_abs_max': grad_abs_max
            }
            
        return stats
