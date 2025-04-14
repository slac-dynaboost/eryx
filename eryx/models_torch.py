"""
PyTorch implementation of disorder models for diffuse scattering calculations.

This module contains PyTorch versions of the disorder models defined in
eryx/models.py. All implementations maintain the same API as the NumPy versions
but use PyTorch tensors and operations to enable gradient flow.

References:
    - Original NumPy implementation in eryx/models.py
"""

import os
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
       - Enables targeted evaluation and gradient-based optimization
       
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
    """
    PyTorch implementation of the OnePhonon model for diffuse scattering calculations.
    
    This class implements a lattice of interacting rigid bodies in the one-phonon
    approximation (a.k.a small-coupling regime) using PyTorch tensors and operations
    to enable gradient flow.
    
    References:
        - Original NumPy implementation in eryx/models.py:OnePhonon
    """
    
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
        
        # Validate inputs
        if q_vectors is not None:
            # Validate q_vectors
            if not isinstance(q_vectors, torch.Tensor):
                raise ValueError("q_vectors must be a PyTorch tensor")
            if q_vectors.dim() != 2 or q_vectors.shape[1] != 3:
                raise ValueError(f"q_vectors must have shape [n_points, 3], got {q_vectors.shape}")
            
            # Set arbitrary q-vector mode
            self.use_arbitrary_q = True
            self.q_vectors = q_vectors.to(device=self.device)
            if not self.q_vectors.requires_grad and self.q_vectors.dtype.is_floating_point:
                self.q_vectors.requires_grad_(True)
            logging.debug("[INIT] use_arbitrary_q=True, loaded q_vectors "
                          f"with shape={self.q_vectors.shape}")
        elif hsampling is None or ksampling is None or lsampling is None:
            raise ValueError("Either q_vectors or all three sampling parameters (hsampling, ksampling, lsampling) must be provided")
        else:
            # Grid-based mode
            self.use_arbitrary_q = False
            logging.debug("[INIT] use_arbitrary_q=False; will generate grid-based q_vectors.")
        
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
        pdb_adapter = PDBToTensor(device=self.device)
        element_weights = pdb_adapter.extract_element_weights(self.model)
        self.model.element_weights = element_weights
        logging.debug(f"[_setup] Extracted {len(element_weights)} element weights.")
        
        from eryx.map_utils import generate_grid
        logging.debug(f"[_setup] use_arbitrary_q={getattr(self, 'use_arbitrary_q', False)}")
        
        if getattr(self, 'use_arbitrary_q', False):
            # In arbitrary q-vector mode, use the provided q_vectors directly.
            logging.debug("[_setup] Using user-provided q_vectors.")
            self.q_grid = self.q_vectors.to(dtype=self.real_dtype)
            
            # Derive hkl_grid from q_vectors using q = 2π * A_inv^T * hkl  => hkl = (1/(2π)) * q * (A_inv^T)^{-1}
            A_inv_tensor = torch.tensor(self.model.A_inv, dtype=self.real_dtype, device=self.device)
            scaling_factor = 1.0 / (2.0 * torch.pi)
            A_inv_T_inv = torch.inverse(A_inv_tensor.T)
            self.hkl_grid = torch.matmul(self.q_grid * scaling_factor, A_inv_T_inv)
            
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
            self.q_grid = 2 * torch.pi * torch.matmul(
                A_inv_tensor.T, 
                self.hkl_grid.T 
            ).T # Transpose the result back
            logging.debug(f"[_setup] Calculated q_grid shape={self.q_grid.shape}, q_grid.dtype={self.q_grid.dtype}")
        
        # Ensure q_grid requires gradients if it's float
        if self.q_grid.dtype.is_floating_point:
            self.q_grid.requires_grad_(True)
        logging.debug(f"[_setup] q_grid requires_grad: {self.q_grid.requires_grad}")
        
        # Compute resolution mask using PyTorch functions
        from eryx.map_utils_torch import compute_resolution
        cell_tensor = torch.tensor(self.model.cell, dtype=self.real_dtype, device=self.device)
        resolution = compute_resolution(cell_tensor, self.hkl_grid)
        logging.debug(f"[_setup] resolution computed, shape={resolution.shape}")
        self.res_mask = resolution > res_limit
        
        # Setup Crystal
        self.crystal = Crystal(self.model)
        self.crystal.supercell_extent(nx=1, ny=1, nz=1)
        self.id_cell_ref = self.crystal.hkl_to_id([0, 0, 0])
        self.n_cell = self.crystal.n_cell
        
        # Setup PDBToTensor adapter for tensor conversions
        pdb_adapter = PDBToTensor(device=self.device)
        self.crystal = pdb_adapter.convert_crystal(self.crystal)
        
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
        import logging
        logging.debug(f"[_setup_phonons] use_arbitrary_q={getattr(self, 'use_arbitrary_q', False)}, model={model}")
        
        # Decide dimensions based on mode
        if getattr(self, 'use_arbitrary_q', False):
            # In arbitrary mode, use the number of provided q-vectors
            total_points = self.q_grid.shape[0]
            logging.debug(f"[_setup_phonons] Arbitrary mode: q_grid.shape={self.q_grid.shape}, total_points={total_points}")
            
            # Initialize tensors based on number of q-vectors
            self.kvec = torch.zeros((total_points, 3), dtype=self.real_dtype, device=self.device)
            self.kvec_norm = torch.zeros((total_points, 1), dtype=self.real_dtype, device=self.device)
            
            # Initialize V and Winv tensors with batched shape
            self.V = torch.zeros((total_points,
                                self.n_asu * self.n_dof_per_asu,
                                self.n_asu * self.n_dof_per_asu),
                              dtype=self.complex_dtype, device=self.device)
            self.Winv = torch.zeros((total_points,
                                    self.n_asu * self.n_dof_per_asu),
                                   dtype=self.complex_dtype, device=self.device)
            
            # Ensure complex tensors require gradients
            self.V.requires_grad_(True)
            self.Winv.requires_grad_(True)
            
            # Ensure complex tensors require gradients
            self.V.requires_grad_(True)
            self.Winv.requires_grad_(True)
        else:
            # In grid mode, use the dimensions from map_shape
            h_dim, k_dim, l_dim = self.map_shape
            total_points = h_dim * k_dim * l_dim
            logging.debug(f"[_setup_phonons] Grid mode: map_shape={self.map_shape}, total_points={total_points}")
            
            # Initialize tensors for grid mode
            self.kvec = torch.zeros((total_points, 3), dtype=self.real_dtype, device=self.device)
            self.kvec_norm = torch.zeros((total_points, 1), dtype=self.real_dtype, device=self.device)
            
            # Initialize V and Winv tensors with batched shape
            self.V = torch.zeros((total_points,
                                self.n_asu * self.n_dof_per_asu,
                                self.n_asu * self.n_dof_per_asu),
                              dtype=self.complex_dtype, device=self.device)
            self.Winv = torch.zeros((total_points,
                                    self.n_asu * self.n_dof_per_asu),
                                   dtype=self.complex_dtype, device=self.device)
        
        # Common initialization for both modes
        self._build_A()
        self._build_M()
        self._build_kvec_Brillouin()
        
        # Setup gamma parameters
        self._setup_gamma_parameters(pdb_path, model, gnm_cutoff, gamma_intra, gamma_inter)
        
        if model == 'gnm':
            # Skip full phonon computation in arbitrary mode for now
            if getattr(self, 'use_arbitrary_q', False):
                logging.debug("[_setup_phonons] Skipping full phonon computation in arbitrary q-vector mode")
            else:
                self.compute_gnm_phonons()
                self.compute_covariance_matrix()
        else:
            self.compute_rb_phonons()
        
        # Calculate and store BZ-averaged ADP if using GNM model
        if self.model_type == 'gnm':
             try:
                 self.bz_averaged_adp = self._compute_bz_averaged_adp()
                 # Ensure it requires grad if inputs did (it should trace back)
                 # Check if any relevant input requires grad
                 input_requires_grad = self.gamma_intra.requires_grad or self.gamma_inter.requires_grad # Add others if needed
                 if input_requires_grad and not self.bz_averaged_adp.requires_grad:
                      # This might indicate a break in the computation graph, add dummy op
                      self.bz_averaged_adp = self.bz_averaged_adp + 0.0 * (self.gamma_intra + self.gamma_inter)
                 logging.debug(f"[_setup_phonons] Calculated bz_averaged_adp, shape={self.bz_averaged_adp.shape}, requires_grad={self.bz_averaged_adp.requires_grad}")
             except Exception as e:
                 logging.error(f"[_setup_phonons] Failed to compute BZ-averaged ADP: {e}")
                 self.bz_averaged_adp = None # Set to None on failure
        else:
             self.bz_averaged_adp = None # Not applicable for non-GNM models
             logging.debug("[_setup_phonons] Skipping BZ-averaged ADP calculation for non-GNM model.")

        logging.debug("[_setup_phonons] done.")
    
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
                    if i_asu == 0 and i_atom < 3:
                        assigned_block = self.Amat[i_asu, i_atom*3:(i_atom+1)*3, :]
                        print(f"  Torch Atom {i_atom} Assigned Block:\n{assigned_block.detach().cpu().numpy()}")
            
            # Add debug prints to compare with NumPy implementation
            if self.n_asu > 0 and self.n_atoms_per_asu > 0:
                print(f"\nDEBUG _build_A: First ASU, first atom Amat block:")
                print(self.Amat[0, 0:3, :].detach().cpu().numpy())
                if self.n_atoms_per_asu > 1:
                    print(f"\nDEBUG _build_A: First ASU, second atom Amat block:")
                    print(self.Amat[0, 3:6, :].detach().cpu().numpy())
                if self.n_atoms_per_asu > 2:
                    print(f"\nDEBUG _build_A: First ASU, third atom Amat block:")
                    print(self.Amat[0, 6:9, :].detach().cpu().numpy())
                
                # Print the entire Amat shape and a summary of its values
                print(f"\nDEBUG _build_A: Amat shape: {self.Amat.shape}")
                print(f"DEBUG _build_A: Amat min: {self.Amat.min().item()}, max: {self.Amat.max().item()}")
                print(f"DEBUG _build_A: Amat mean: {self.Amat.mean().item()}, std: {self.Amat.std().item()}")
            
            # Keep high precision
            # Do NOT convert back to float32
            
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
            
            # Debug print for reshaped Mmat
            print(f"\nPyTorch reshaped Mmat shape: {Mmat.shape}")
            print(f"PyTorch reshaped Mmat[0,0]: {Mmat[0,0].item()}")
            
            # Robust regularization - single value that works
            eps = 1e-6
            eye = torch.eye(Mmat.shape[0], device=self.device, dtype=Mmat.dtype)
            Mmat_reg = Mmat + eps * eye
            
            # Enhanced try-except with better fallback
            try:
                # Try standard Cholesky decomposition first
                L = torch.linalg.cholesky(Mmat_reg)
                self.Linv = torch.linalg.inv(L)
                print(f"PyTorch Cholesky L[0,0]: {L[0,0].item()}")
            except RuntimeError as e:
                # Print diagnostic info
                print(f"Cholesky decomposition failed: {e}")
                print(f"Matrix condition number: {torch.linalg.cond(Mmat_reg).item()}")
                
                # Add stronger regularization and try again
                stronger_eps = 1e-4
                Mmat_reg = Mmat + stronger_eps * eye
                try:
                    L = torch.linalg.cholesky(Mmat_reg)
                    self.Linv = torch.linalg.inv(L)
                    print("Succeeded with stronger regularization")
                    print(f"PyTorch Cholesky L[0,0] (stronger reg): {L[0,0].item()}")
                except RuntimeError:
                    # Final fallback to SVD approach
                    print("Falling back to SVD decomposition")
                    U, S, V = torch.linalg.svd(Mmat_reg, full_matrices=False)
                    S = torch.clamp(S, min=1e-8)
                    self.Linv = U @ torch.diag(1.0 / torch.sqrt(S)) @ V
                    print(f"PyTorch SVD U[0,0]: {U[0,0].item()}, S[0]: {S[0].item()}")
            
            # Debug print for Linv
            print(f"PyTorch Linv shape: {self.Linv.shape}")
            print(f"PyTorch Linv dtype: {self.Linv.dtype}")
            print(f"PyTorch Linv[0,0]: {self.Linv[0,0].item()}")
            
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
        
        # Debug print for NumPy comparison
        print("\n--- _build_M_allatoms Debug ---")
        
        # Try to directly access the same weights as NumPy implementation
        weights = []
        if hasattr(self.model, 'elements'):
            print("Accessing weights directly from model.elements (NumPy approach)")
            try:
                # This matches the NumPy implementation exactly
                weights = [element.weight for structure in self.model.elements for element in structure]
                print(f"Direct weights extraction: count={len(weights)}")
                print(f"First few weights: {weights[:5]}")
                print(f"Weight stats: min={min(weights)}, max={max(weights)}, mean={sum(weights)/len(weights)}")
            except Exception as e:
                print(f"Error in direct weight extraction: {e}")
        
        # Create mass array from weights
        if weights:
            mass_array = torch.tensor(weights, dtype=dtype, device=self.device)
            print(f"mass_array from weights: shape={mass_array.shape}, dtype={mass_array.dtype}")
            print(f"mass_array stats: min={mass_array.min().item()}, max={mass_array.max().item()}, mean={mass_array.mean().item()}")
        else:
            # Fallback to ones
            print("Fallback: Using ones for mass_array")
            mass_array = torch.ones(self.n_asu * self.n_atoms_per_asu, dtype=dtype, device=self.device)
        
        # Create block diagonal matrix
        eye3 = torch.eye(3, device=self.device, dtype=dtype)
        
        # Debug print for block construction approach
        print("\nCreating block diagonal matrix...")
        
        # Create mass_list exactly like NumPy implementation
        mass_list = []
        for i in range(self.n_asu * self.n_atoms_per_asu):
            # Create 3x3 block for each atom (mass * identity)
            block = mass_array[i] * eye3
            mass_list.append(block)
        
        # Print first few blocks for debugging
        print(f"First block shape: {mass_list[0].shape}")
        print(f"First block:\n{mass_list[0].detach().cpu().numpy()}")
        
        # Create block diagonal matrix
        total_dim = self.n_asu * self.n_atoms_per_asu * 3
        M_block_diag = torch.zeros((total_dim, total_dim), device=self.device, dtype=dtype)
        
        # Fill block diagonal matrix manually to match NumPy exactly
        current_idx = 0
        for block in mass_list:
            block_size = block.shape[0]
            M_block_diag[current_idx:current_idx+block_size, current_idx:current_idx+block_size] = block
            current_idx += block_size
        
        # Debug print for block diagonal matrix
        print(f"M_block_diag shape: {M_block_diag.shape}")
        print(f"M_block_diag diagonal elements (first few): {torch.diagonal(M_block_diag)[:15].detach().cpu().numpy()}")
        
        # Reshape to 4D tensor
        M_allatoms = M_block_diag.reshape(self.n_asu, self.n_dof_per_asu_actual,
                                        self.n_asu, self.n_dof_per_asu_actual)
        
        # Debug prints for M_allatoms
        print(f"\nPyTorch M_allatoms shape: {M_allatoms.shape}")
        print(f"PyTorch M_allatoms diag[0:5]: {torch.diagonal(M_allatoms[0,:,0,:]).cpu().numpy()[0:5]}")
        if M_allatoms.shape[0] > 1 and M_allatoms.shape[2] > 1:
            print(f"PyTorch M_allatoms[0,0,1,0]: {M_allatoms[0,0,1,0].item()}")
        
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
        
        # Debug prints for Mmat
        print(f"\nPyTorch Mmat shape: {Mmat.shape}")
        if Mmat.shape[0] > 0 and Mmat.shape[2] > 0:
            print(f"PyTorch Mmat[0,0,0,0]: {Mmat[0,0,0,0].item()}")
            if Mmat.shape[1] > 1 and Mmat.shape[3] > 1:
                print(f"PyTorch Mmat[0,1,0,1]: {Mmat[0,1,0,1].item()}")
        
        return Mmat
    
    #@debug
    def _build_kvec_Brillouin(self):
        """
        Compute all k-vectors and their norm in the first Brillouin zone.
        
        In grid mode: Regularly samples [-0.5,0.5[ for h, k and l.
        In arbitrary mode: Derives k-vectors directly from q-vectors as k = q/(2π).
        """
        import logging
        logging.debug(f"[_build_kvec_Brillouin] use_arbitrary_q={getattr(self, 'use_arbitrary_q', False)}")
        
        if getattr(self, 'use_arbitrary_q', False):
            # Arbitrary q-vector mode
            # k = q/(2π)
            self.kvec = self.q_grid / (2.0 * torch.pi)
            self.kvec_norm = torch.norm(self.kvec, dim=1, keepdim=True)
            
            # Ensure tensors require gradients
            self.kvec.requires_grad_(True)
            self.kvec_norm.requires_grad_(True)
            
            logging.debug(f"[_build_kvec_Brillouin] Arbitrary mode: kvec.shape={self.kvec.shape}, kvec_norm.shape={self.kvec_norm.shape}")
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
            
            # Generate 1D tensors for h, k, l coordinates using _center_kvec (ensure float64)
            h_coords = torch.tensor([self._center_kvec(dh, h_dim_bz) for dh in range(h_dim_bz)], 
                                   device=self.device, dtype=self.real_dtype)
            k_coords = torch.tensor([self._center_kvec(dk, k_dim_bz) for dk in range(k_dim_bz)], 
                                   device=self.device, dtype=self.real_dtype)
            l_coords = torch.tensor([self._center_kvec(dl, l_dim_bz) for dl in range(l_dim_bz)], 
                                   device=self.device, dtype=self.real_dtype)
            
            # Create meshgrid and reshape to [total_k_points, 3]
            h_grid, k_grid, l_grid = torch.meshgrid(h_coords, k_coords, l_coords, indexing='ij')
            # Ensure hkl_fractional is float64
            hkl_fractional = torch.stack([h_grid.flatten(), k_grid.flatten(), l_grid.flatten()], dim=1).to(dtype=self.real_dtype)
            
            # Compute kvec and kvec_norm (ensure float64)
            self.kvec = torch.matmul(hkl_fractional, A_inv_tensor)
            self.kvec_norm = torch.norm(self.kvec, dim=1, keepdim=True)
            
            # Ensure final tensors have the correct dtype
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
        
        # Create identity matrix for Kronecker product
        eye3 = torch.eye(3, device=self.device, dtype=self.complex_dtype)
        
        for i_cell in range(self.n_cell):
            for i_asu in range(self.n_asu):
                for j_asu in range(self.n_asu):
                    # Apply Kronecker product with identity matrix (3x3)
                    # This expands each element of the hessian into a 3x3 block
                    h_block = hessian_allatoms[i_asu, :, i_cell, j_asu, :]
                    h_expanded = torch.zeros((h_block.shape[0] * 3, h_block.shape[1] * 3), 
                                            dtype=self.complex_dtype, device=self.device)
                    
                    # Manually implement the Kronecker product
                    for i in range(h_block.shape[0]):
                        for j in range(h_block.shape[1]):
                            h_expanded[i*3:(i+1)*3, j*3:(j+1)*3] = h_block[i, j] * eye3
                    
                    # Perform matrix multiplication with expanded hessian
                    # Ensure all tensors are complex for compatibility
                    proj = torch.matmul(self.Amat[i_asu].T.to(self.complex_dtype),
                                        torch.matmul(h_expanded,
                                                     self.Amat[j_asu].to(self.complex_dtype)))
                    hessian[i_asu, :, i_cell, j_asu, :] = proj
        
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
        
        The eigenvalues (Winv) and eigenvectors (V) are stored for intensity calculation.
        """
        # Compute the Hessian matrix first (works for both modes)
        hessian = self.compute_hessian()
        
        # Number of points depends on mode
        if getattr(self, 'use_arbitrary_q', False):
            total_points = self.q_grid.shape[0]
            print(f"Computing phonons for {total_points} arbitrary q-vectors")
        else:
            # Use Brillouin zone sampling dimensions instead of map_shape
            h_dim_bz = int(self.hsampling[2])
            k_dim_bz = int(self.ksampling[2])
            l_dim_bz = int(self.lsampling[2])
            total_points = h_dim_bz * k_dim_bz * l_dim_bz
            print(f"Computing phonons for {total_points} grid points based on BZ sampling ({h_dim_bz}x{k_dim_bz}x{l_dim_bz})")
        
        # Initialize output tensors with proper shapes for both modes
        self.V = torch.zeros((total_points,
                              self.n_asu * self.n_dof_per_asu,
                              self.n_asu * self.n_dof_per_asu),
                            dtype=self.complex_dtype, device=self.device)
        self.Winv = torch.zeros((total_points,
                                self.n_asu * self.n_dof_per_asu),
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
            print("Warning: No crystal object found in OnePhonon model")
        
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
        
        # Compute K matrices for all k-vectors at once
        Kmat_all = gnm_torch.compute_K(hessian, self.kvec)
        print(f"Kmat_all requires_grad: {Kmat_all.requires_grad}")
        print(f"Kmat_all.grad_fn: {Kmat_all.grad_fn}")
        
        # --- Debug Print Start ---
        print_idx = 1  # Choose a non-zero index for detailed prints
        if total_points > print_idx: 
            print(f"\n--- PyTorch Debug Index {print_idx} ---")
            # Correctly index the Kmat_all tensor (5D) and then call .item()
            # Kmat_all shape is [batch, n_asu, n_atoms, n_asu, n_atoms]
            # Kmat_all[print_idx] is [n_asu, n_atoms, n_asu, n_atoms]
            # Kmat_all[print_idx, 0, 0] is [n_asu, n_atoms]
            # Need Kmat_all[print_idx, 0, 0, 0, 0] if n_atoms > 0 else handle empty case
            if Kmat_all.dim() >= 3 and Kmat_all.shape[1] > 0 and Kmat_all.shape[2] > 0 and Kmat_all.shape[3] > 0 and Kmat_all.shape[4] > 0:
                print(f"PyTorch Kmat_all[{print_idx},0,0,0,0]: {Kmat_all[print_idx,0,0,0,0].item():.8e}")
            else:
                print(f"PyTorch Kmat_all shape: {Kmat_all.shape}")
        # --- Debug Print End -----
        
        # Reshape K matrices to 2D form for each k-vector
        dof_total = self.n_asu * self.n_dof_per_asu
        Kmat_all_2d = Kmat_all.reshape(total_points, dof_total, dof_total)
        
        print(f"Compute_K complete, Kmat_all_2d shape = {Kmat_all_2d.shape}")
        
        # Compute dynamical matrices (D = L^(-1) K L^(-H)) for all k-vectors
        # Use torch.bmm for batched matrix multiplication
        Linv_batch = Linv_complex.unsqueeze(0).expand(total_points, -1, -1)
        # Use conjugate transpose (.H) for complex matrices instead of just transpose (.T)
        Linv_H_batch = Linv_complex.conj().T.unsqueeze(0).expand(total_points, -1, -1)
        
        # First multiply Kmat_all_2d with Linv_H_batch
        temp = torch.bmm(Kmat_all_2d, Linv_H_batch)
        # Then multiply Linv_batch with the result
        Dmat_all = torch.bmm(Linv_batch, temp)
        
        # --- Debug Print Start ---
        if total_points > print_idx: 
            # Print first element of the matrix at index print_idx
            if Dmat_all.dim() >= 3:
                print(f"PyTorch Dmat_all[{print_idx},0,0]: {Dmat_all[print_idx,0,0].item() if Dmat_all[print_idx,0,0].numel() == 1 else Dmat_all[print_idx,0,0][0,0].item():.8e}")
            else:
                print(f"PyTorch Dmat_all shape: {Dmat_all.shape}")
        # --- Debug Print End -----
        
        print(f"Dmat_all computation complete, shape = {Dmat_all.shape}")
        print(f"Dmat_all requires_grad: {Dmat_all.requires_grad}")
        
        # Initialize v_all *outside* the try block
        v_all = None
        
        print(f"Dmat_all computation complete, shape = {Dmat_all.shape}")
        print(f"Dmat_all requires_grad: {Dmat_all.requires_grad}")

        eigenvalues_all_list = []
        eigenvectors_detached_list = [] # Store detached eigenvectors

        # Process each matrix individually
        for i in range(total_points):
            D_i = Dmat_all[i]
            D_i_hermitian = 0.5 * (D_i + D_i.H) # Ensure Hermiticity

            # 1. Get eigenvectors WITHOUT gradient tracking using eigh
            with torch.no_grad():
                try:
                    # We only need the eigenvectors (v_i) here
                    _, v_i_no_grad = torch.linalg.eigh(D_i_hermitian)
                except torch._C._LinAlgError as e:
                     print(f"Warning: eigh failed for matrix {i} in no_grad context. Using identity. Error: {e}")
                     n_dof = D_i_hermitian.shape[0]
                     v_i_no_grad = torch.eye(n_dof, dtype=self.complex_dtype, device=self.device) # Fallback

            # Flip eigenvectors to match descending eigenvalue order later
            v_flipped_no_grad = torch.flip(v_i_no_grad, dims=[-1])
            eigenvectors_detached_list.append(v_flipped_no_grad)

            # 2. Recompute eigenvalues DIFFERENTIABLY using v.H @ D @ v
            current_eigenvalues = []
            n_modes = v_i_no_grad.shape[1]
            for mode_idx in range(n_modes):
                # Use the DETACHED eigenvector from the no_grad calculation
                v_mode_no_grad = v_i_no_grad[:, mode_idx:mode_idx+1] # Shape [n_dof, 1]
                # Calculate lambda = v.H @ D @ v (This IS differentiable w.r.t D_i)
                # Use the original D_i (which has grad info), not D_i_hermitian if grads matter
                lambda_mode = torch.matmul(torch.matmul(v_mode_no_grad.H, D_i), v_mode_no_grad)
                # Result should be real, take real part for safety & correct dtype
                current_eigenvalues.append(lambda_mode[0, 0].real)

            eigenvalues_tensor = torch.stack(current_eigenvalues) # Real eigenvalues

            # 3. Process eigenvalues (thresholding, flipping)
            eps = 1e-6
            eigenvalues_processed = torch.where(
                eigenvalues_tensor < eps,
                torch.tensor(float('nan'), device=eigenvalues_tensor.device, dtype=self.real_dtype),
                eigenvalues_tensor
            )
            # Flip eigenvalues to match descending order (as SVD would give)
            eigenvalues_processed_flipped = torch.flip(eigenvalues_processed, dims=[-1])
            eigenvalues_all_list.append(eigenvalues_processed_flipped)

        # Stack results
        eigenvalues_all = torch.stack(eigenvalues_all_list)  # Differentiable eigenvalues
        v_all_detached = torch.stack(eigenvectors_detached_list) # Non-differentiable eigenvectors

        print(f"Recomputed eigenvalues using eigh complete")
        print(f"eigenvalues_all requires_grad: {eigenvalues_all.requires_grad}") # Should be True
        print(f"v_all_detached requires_grad: {v_all_detached.requires_grad}") # Should be False

        # Transform eigenvectors V = L^(-H) v (using detached eigenvectors)
        self.V = torch.matmul(
            Linv_complex.H.unsqueeze(0).expand(total_points, -1, -1),
            v_all_detached # Use the detached eigenvectors here
        )

        # Calculate Winv = 1 / eigenvalues (using differentiable eigenvalues)
        eps_div = 1e-8
        winv_all = torch.where(
            torch.isnan(eigenvalues_all),
            torch.tensor(float('nan'), device=eigenvalues_all.device, dtype=self.real_dtype),
            1.0 / (eigenvalues_all + eps_div)
        )
        self.Winv = winv_all.to(dtype=self.complex_dtype) # Cast to complex

        # Ensure requires_grad is set correctly for Winv (V is handled by using detached vecs)
        if not self.Winv.requires_grad and eigenvalues_all.requires_grad:
             self.Winv = self.Winv + 0 * eigenvalues_all.sum().to(self.complex_dtype) # Reconnect if needed

        print(f"V requires_grad: {self.V.requires_grad}") # Should be False now
        print(f"Winv requires_grad: {self.Winv.requires_grad}") # Should be True if inputs required grad
        print(f"Phonon computation complete: V.shape={self.V.shape}, Winv.shape={self.Winv.shape}")
        
        # Set requires_grad for Winv tensor only (V is already handled)
        self.Winv.requires_grad_(True)
    
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
        
        Supports both grid-based and arbitrary q-vector modes.
        """
        # Initialize covariance tensor
        self.covar = torch.zeros((self.n_asu * self.n_dof_per_asu,
                                self.n_cell, self.n_asu * self.n_dof_per_asu),
                               dtype=self.complex_dtype, device=self.device)
        
        # Define debug index for comparison
        debug_idx = 1
        
        # Get total number of k-vectors based on mode
        if getattr(self, 'use_arbitrary_q', False):
            total_points = self.q_grid.shape[0]
            print(f"Computing covariance matrix for {total_points} arbitrary q-vectors")
        else:
            # Use Brillouin zone dimensions to match NumPy implementation exactly
            h_dim_bz = int(self.hsampling[2])
            k_dim_bz = int(self.ksampling[2])
            l_dim_bz = int(self.lsampling[2])
            total_points = h_dim_bz * k_dim_bz * l_dim_bz
            print(f"Computing covariance matrix for {total_points} grid points based on BZ sampling")
        
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
            print("Warning: No crystal object found in OnePhonon model")
        
        # Compute the Hessian matrix
        hessian = self.compute_hessian()
        
        if getattr(self, 'use_arbitrary_q', False):
            # Arbitrary-q mode
            # Compute inverse K matrices for all k-vectors at once
            # This is a batch operation for efficiency
            Kinv_all = gnm_torch.compute_Kinv(hessian, self.kvec, reshape=False)
            print(f"Kinv_all computation complete, shape = {Kinv_all.shape}")
            print(f"Kinv_all requires_grad: {Kinv_all.requires_grad}")
            print(f"Kinv_all.grad_fn: {Kinv_all.grad_fn}")
            
            # Debug print for specific k-vector
            if total_points > debug_idx:
                print(f"\n--- PyTorch Kinv Debug Index {debug_idx} ---")
                print(f"DEBUG Arbitrary-Q: Kinv[{debug_idx}, 0, 0]: {Kinv_all[debug_idx,0,0].item():.8e}")
                print(f"DEBUG Arbitrary-Q: Kinv[{debug_idx}, 5, 5]: {Kinv_all[debug_idx,5,5].item():.8e}")
            
            # Calculate phase factors and accumulate covariance contributions
            for j_cell in range(self.n_cell):
                # Get cell origin
                r_cell = self.crystal.get_unitcell_origin(self.crystal.id_to_hkl(j_cell))
                
                # Calculate phase factors for all k-vectors
                all_phases = torch.sum(self.kvec * r_cell, dim=1)
                
                # Use complex exponential with exact NumPy-compatible behavior
                # eikr = exp(i * phase) = cos(phase) + i*sin(phase)
                cos_phases = torch.cos(all_phases)
                sin_phases = torch.sin(all_phases)
                eikr_all = torch.complex(cos_phases, sin_phases)
                
                # Debug print for specific phase
                if total_points > debug_idx and j_cell < 3:
                    print(f"DEBUG Arbitrary-Q: Cell {j_cell}, eikr[{debug_idx}]: {eikr_all[debug_idx].item()}")
                    print(f"DEBUG Arbitrary-Q: Cell {j_cell}, phase[{debug_idx}]: {all_phases[debug_idx].item():.8e}")
                    print(f"DEBUG Arbitrary-Q: Cell {j_cell}, r_cell: {r_cell.detach().cpu().numpy()}")
                    print(f"DEBUG Arbitrary-Q: Cell {j_cell}, kvec[{debug_idx}]: {self.kvec[debug_idx].detach().cpu().numpy()}")
                
                # Reshape phase factors for matrix multiplication
                eikr_reshaped = eikr_all.view(-1, 1, 1)
                
                # Calculate the average contribution across all k-points for this cell offset
                # using torch.mean over the batch dimension (dim=0)
                average_contribution = torch.mean(Kinv_all * eikr_reshaped, dim=0)
                
                # Debug print for accumulated sum
                if j_cell < 3:
                    print(f"Cell {j_cell} average_contribution[0,0]: {average_contribution[0,0].item()}")
                
                # Accumulate in covariance tensor
                self.covar[:, j_cell, :] = average_contribution
        else:
            # Grid-based mode - Revised to match arbitrary-q mode structure
            # First compute all Kinv matrices for all BZ points
            Kinv_bz = []
            kvec_bz = [] # Store the kvecs corresponding to BZ indices
            
            # Loop over Brillouin zone points to collect all Kinv values
            for dh in range(h_dim_bz):
                for dk in range(k_dim_bz):
                    for dl in range(l_dim_bz):
                        # Calculate the flat Brillouin zone index
                        idx = self._3d_to_flat_indices_bz(
                            torch.tensor([dh], device=self.device, dtype=torch.long),
                            torch.tensor([dk], device=self.device, dtype=torch.long),
                            torch.tensor([dl], device=self.device, dtype=torch.long)
                        ).item()
                        
                        # Get k-vector for this index
                        kvec = self.kvec[idx]
                        kvec_bz.append(kvec)
                        
                        # Compute Kinv for this specific k-vector
                        Kinv = gnm_torch.compute_Kinv(hessian, kvec.unsqueeze(0), reshape=False)[0]
                        Kinv_bz.append(Kinv)
                        
                        # Debug print for our chosen debug index
                        if idx == debug_idx:
                            print(f"\nDEBUG Grid Mode (Revised): Matched debug_idx={debug_idx} at (dh,dk,dl)=({dh},{dk},{dl})")
                            print(f"DEBUG Grid Mode (Revised): Kinv[{debug_idx}, 0, 0]: {Kinv[0, 0].item():.8e}")
                            print(f"DEBUG Grid Mode (Revised): Kinv[{debug_idx}, 5, 5]: {Kinv[5, 5].item():.8e}")

            # Stack all Kinv and kvec values
            Kinv_all_bz = torch.stack(Kinv_bz) # Shape [total_k_points_bz, n_dof, n_dof]
            kvec_all_bz = torch.stack(kvec_bz) # Shape [total_k_points_bz, 3]
            
            print(f"Grid Mode (Revised) Kinv_all_bz shape = {Kinv_all_bz.shape}")
            
            # Calculate phase factors and accumulate covariance contributions (like arbitrary-q mode)
            for j_cell in range(self.n_cell):
                # Get cell origin
                r_cell = self.crystal.get_unitcell_origin(self.crystal.id_to_hkl(j_cell))
                
                # Calculate phase factors for all k-vectors
                all_phases = torch.sum(kvec_all_bz * r_cell, dim=1)
                cos_phases = torch.cos(all_phases)
                sin_phases = torch.sin(all_phases)
                eikr_all = torch.complex(cos_phases, sin_phases)
                
                # Debug print for specific phase
                if total_points > debug_idx and j_cell < 3:
                    print(f"DEBUG Grid Mode (Revised): Cell {j_cell}, eikr[{debug_idx}]: {eikr_all[debug_idx].item()}")
                    print(f"DEBUG Grid Mode (Revised): Cell {j_cell}, phase[{debug_idx}]: {all_phases[debug_idx].item():.8e}")
                    print(f"DEBUG Grid Mode (Revised): Cell {j_cell}, r_cell: {r_cell.detach().cpu().numpy()}")
                    print(f"DEBUG Grid Mode (Revised): Cell {j_cell}, kvec[{debug_idx}]: {kvec_all_bz[debug_idx].detach().cpu().numpy()}")
                
                # Reshape phase factors for matrix multiplication
                eikr_reshaped = eikr_all.view(-1, 1, 1)
                
                # Sum over the BZ points (dim=0)
                complex_sum = torch.sum(Kinv_all_bz * eikr_reshaped, dim=0)
                
                # Debug print for accumulated sum
                if j_cell < 3:
                    print(f"Grid Mode (Revised) Cell {j_cell} complex_sum[0,0]: {complex_sum[0,0].item()}")
                
                # Accumulate in covariance tensor (restore division by total_points)
                self.covar[:, j_cell, :] = complex_sum / total_points
        
        # Get reference cell ID for [0,0,0]
        ref_cell_id = self.crystal.hkl_to_id([0, 0, 0])
        
        # Debug print for final covar values
        print(f"\nDEBUG {'Arbitrary-Q' if getattr(self, 'use_arbitrary_q', False) else 'Grid Mode'}: Final covar[0,0,0]: {self.covar[0, 0, 0].item()}")
        print(f"DEBUG {'Arbitrary-Q' if getattr(self, 'use_arbitrary_q', False) else 'Grid Mode'}: Final covar[5,0,5]: {self.covar[5, 0, 5].item()}")
        
        # Extract diagonal elements for ADP calculation
        # Use .real attribute instead of torch.real() to preserve gradient flow
        diagonal_values = torch.diagonal(self.covar[:, ref_cell_id, :], dim1=0, dim2=1)
        self.ADP = diagonal_values.real
        
        # Debug print for diagonal values
        print(f"Diagonal values shape: {diagonal_values.shape}")
        print(f"First few diagonal values: {diagonal_values[:5].detach().cpu().numpy()}")
        
        # Transform ADP using the displacement projection matrix
        Amat = torch.transpose(self.Amat, 0, 1).reshape(self.n_dof_per_asu_actual, self.n_asu * self.n_dof_per_asu)
        self.ADP = torch.matmul(Amat, self.ADP)
        
        # Debug print after Amat transformation
        print(f"ADP after Amat transform shape: {self.ADP.shape}")
        print(f"First few ADP values after transform: {self.ADP[:5].detach().cpu().numpy()}")
        
        # Sum over spatial dimensions (x,y,z)
        self.ADP = torch.sum(self.ADP.reshape(int(self.ADP.shape[0] / 3), 3), dim=1)
        
        # Debug print after spatial sum
        print(f"ADP after spatial sum shape: {self.ADP.shape}")
        print(f"First few ADP values after spatial sum: {self.ADP[:5].detach().cpu().numpy()}")
        
        # Scale ADP to match experimental values - exactly match NumPy behavior
        model_adp_tensor = self.array_to_tensor(self.model.adp)
        
        # Handle NaN values exactly like NumPy
        valid_adp = self.ADP[~torch.isnan(self.ADP)]
        if valid_adp.numel() > 0:
            # Calculate mean of valid values only
            adp_mean = torch.mean(valid_adp)
            # Calculate scaling factor exactly as in NumPy
            ADP_scale = torch.mean(model_adp_tensor) / (8 * torch.pi * torch.pi * adp_mean / 3)
        else:
            # Fallback if all values are NaN
            ADP_scale = torch.tensor(1.0, device=self.device, dtype=self.real_dtype)
        
        # Debug print for scaling
        print(f"ADP scaling factor: {ADP_scale.item():.8e}")
        
        # Apply scaling to ADP and covariance matrix
        self.ADP = self.ADP * ADP_scale
        self.covar = self.covar * ADP_scale
        
        # Debug print after scaling
        print(f"ADP after scaling, first few values: {self.ADP[:5].detach().cpu().numpy()}")
        
        # Reshape covariance matrix to final format
        # Use .real attribute instead of torch.real() to preserve gradient flow
        reshaped_covar = self.covar.reshape((self.n_asu, self.n_dof_per_asu,
                                           self.n_cell, self.n_asu, self.n_dof_per_asu))
        self.covar = reshaped_covar.real
        
        # Set requires_grad for ADP tensor
        self.ADP.requires_grad_(True)
        
        print(f"ADP requires_grad: {self.ADP.requires_grad}")
        print(f"ADP.grad_fn: {self.ADP.grad_fn}")
        print(f"Covariance computation complete: ADP.shape={self.ADP.shape}")
    
    #@debug
    def apply_disorder(self, rank: int = -1, outdir: Optional[str] = None, 
                       use_data_adp: bool = False) -> torch.Tensor:
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
        
        # Prepare ADPs
        if use_data_adp:
            ADP = torch.tensor(self.model.adp[0], dtype=self.real_dtype, device=self.device) / (8 * torch.pi * torch.pi)
        else:
            if hasattr(self, "ADP"):
                ADP = self.ADP.to(dtype=self.real_dtype, device=self.device)
            else:
                # fallback if not computed
                ADP = torch.ones(self.n_atoms_per_asu, device=self.device, dtype=self.real_dtype)
        logging.debug(f"[apply_disorder] ADP shape= {self.ADP.shape if hasattr(self,'ADP') else '(none)'}")
        
        # Initialize intensity tensor
        Id = torch.zeros(self.q_grid.shape[0], dtype=self.real_dtype, device=self.device)
        logging.debug(f"[apply_disorder] q_grid.size={self.q_grid.shape[0]} total points. res_mask sum={int(self.res_mask.sum())}.")
        
        # Import structure_factors function
        from eryx.scatter_torch import structure_factors
        
        if getattr(self, 'use_arbitrary_q', False):
            # In arbitrary q-vector mode, we process all provided q-vectors directly
            n_points = self.q_grid.shape[0]
            logging.debug(f"[apply_disorder] Processing {n_points} arbitrary q-vectors as a batch")
            
            # Get valid indices directly from the resolution mask
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
            
            # Pre-compute all ASU data to avoid repeated tensor creation
            # Use array_to_tensor adapter for consistent dtype and gradient settings
            asu_data = []
            for i_asu in range(self.n_asu):
                asu_data.append({
                    'xyz': self.array_to_tensor(self.crystal.get_asu_xyz(i_asu), dtype=torch.float32),
                    'ff_a': self.array_to_tensor(self.model.ff_a[i_asu], dtype=torch.float32),
                    'ff_b': self.array_to_tensor(self.model.ff_b[i_asu], dtype=torch.float32),
                    'ff_c': self.array_to_tensor(self.model.ff_c[i_asu], dtype=torch.float32),
                    'project': self.Amat[i_asu]
                })
            
            # Compute structure factors for all valid q-vectors
            # Initialize F with the CORRECT high-precision complex dtype
            F = torch.zeros((valid_indices.numel(), self.n_asu, self.n_dof_per_asu),
                          dtype=self.complex_dtype, device=self.device) # <--- FIXED DTYPE HERE
            
            # Process all ASUs
            for i_asu in range(self.n_asu):
                asu = asu_data[i_asu]
                # Ensure all inputs to structure_factors are high precision
                q_vectors = self.q_grid[valid_indices].to(dtype=self.real_dtype)
                xyz = asu['xyz'].to(dtype=self.real_dtype)
                ff_a = asu['ff_a'].to(dtype=self.real_dtype)
                ff_b = asu['ff_b'].to(dtype=self.real_dtype)
                ff_c = asu['ff_c'].to(dtype=self.real_dtype)
                adp = ADP.to(dtype=self.real_dtype)
                project = asu['project'].to(dtype=self.real_dtype)
                
                # Compute structure factors (expects float64 inputs, returns complex128)
                sf_result = structure_factors(
                    q_vectors, xyz, ff_a, ff_b, ff_c, U=adp,
                    n_processes=self.n_processes, compute_qF=True,
                    project_on_components=project, sum_over_atoms=False
                )
                # Assign result to F slice with explicit dtype cast
                F[:, i_asu, :] = sf_result.to(self.complex_dtype)
                # Assert F's dtype after assignment
                assert F.dtype == self.complex_dtype, f"F dtype after assignment (ASU {i_asu}) is {F.dtype}, expected {self.complex_dtype}"
            
            # Reshape for matrix operations
            F = F.reshape((valid_indices.numel(), self.n_asu * self.n_dof_per_asu)) # complex128
            assert F.dtype == self.complex_dtype, f"Reshaped F dtype is {F.dtype}, expected {self.complex_dtype}"
            
            # Apply disorder model depending on rank parameter
            if rank == -1:
                # Get eigenvectors and eigenvalues for all valid points
                V_valid = self.V[valid_indices].to(self.complex_dtype)  # shape: [n_valid, n_dof, n_dof]
                Winv_valid = self.Winv[valid_indices].to(self.complex_dtype)  # shape: [n_valid, n_dof]
                
                # Define target q-index for debugging
                target_q_idx = 9
                
                # Process each point
                intensity = torch.zeros(valid_indices.numel(), device=self.device, dtype=self.real_dtype)
                for i, idx in enumerate(valid_indices):
                    # Get eigenvectors and eigenvalues for this q-vector
                    V_idx = V_valid[i].to(self.complex_dtype)
                    Winv_idx = Winv_valid[i].to(self.complex_dtype)
                    
                    # Debug print for target q-index
                    current_q_idx = idx.item()
                    if current_q_idx == target_q_idx:
                        print(f"\n--- ARBITRARY-Q MODE (q_idx={current_q_idx}, i={i}) ---")
                        print(f"V_idx shape: {V_idx.shape}, dtype: {V_idx.dtype}")
                        print(f"Winv_idx shape: {Winv_idx.shape}, dtype: {Winv_idx.dtype}")
                        print(f"V_idx[0,0] (abs): {torch.abs(V_idx[0,0]).item():.6e}")
                        print(f"Winv_idx[0] (real): {Winv_idx[0].real.item():.6e}")
                    
                    # Always ensure F has the correct complex dtype before matmul with V
                    F_i = F[i]
                    assert F_i.dtype == self.complex_dtype, f"F_i dtype is {F_i.dtype}, expected {self.complex_dtype}"
                    assert V_idx.dtype == self.complex_dtype, f"V_idx dtype is {V_idx.dtype}, expected {self.complex_dtype}"
                    
                    # Debug print for target q-index
                    if current_q_idx == target_q_idx:
                        print(f"F_i shape: {F_i.shape}, dtype: {F_i.dtype}")
                        print(f"F_i[0] (abs): {torch.abs(F_i[0]).item():.6e}")
                
                    # Compute F·V for all modes at once
                    FV = torch.matmul(F_i, V_idx)
                    
                    # Calculate absolute squared values - ensure real output
                    FV_abs_squared = torch.abs(FV)**2
                    
                    # Extract real part of eigenvalues with NaN handling
                    if torch.is_complex(Winv_idx):
                        real_winv = Winv_idx.real  # Use .real attribute to preserve gradients
                    else:
                        real_winv = Winv_idx
                    
                    # Cast real_winv to the target real dtype before multiplication
                    real_winv_float64 = real_winv.to(dtype=self.real_dtype)
                    
                    # Debug print for target q-index
                    if current_q_idx == target_q_idx:
                        print(f"FV shape: {FV.shape}, dtype: {FV.dtype}")
                        print(f"FV[0] (abs): {torch.abs(FV[0]).item():.6e}")
                        print(f"real_winv shape: {real_winv_float64.shape}, dtype: {real_winv_float64.dtype}")
                        print(f"real_winv[0]: {real_winv_float64[0].item():.6e}")
                        intensity_contributions = FV_abs_squared * real_winv_float64
                        print(f"Intensity Contributions[0]: {intensity_contributions[0].item():.6e}")
                        final_intensity = torch.sum(intensity_contributions)
                        print(f"Final Intensity for q_idx={target_q_idx}: {final_intensity.item():.6e}")
                    
                    # Weight by eigenvalues and sum - ensure real output
                    intensity[i] = torch.sum(FV_abs_squared * real_winv_float64)
                    
                    # Debug print for target q-index
                    if current_q_idx == target_q_idx:
                        print(f"Value assigned to Id: {intensity[i].item():.6e}")
            else:
                # Process single mode
                intensity = torch.zeros(valid_indices.numel(), device=self.device)
                for i, idx in enumerate(valid_indices):
                    # Get mode for this q-vector
                    V_rank = self.V[idx, :, rank]
                    
                    # Always ensure F has the correct complex dtype before matmul with V
                    F_i = F[i]
                    assert F_i.dtype == self.complex_dtype, f"F_i dtype is {F_i.dtype}, expected {self.complex_dtype}"
                    assert V_rank.dtype == self.complex_dtype, f"V_rank dtype is {V_rank.dtype}, expected {self.complex_dtype}"
                
                    # Compute FV
                    FV = torch.matmul(F_i, V_rank)
                    
                    # Calculate absolute squared value
                    FV_abs_squared = torch.abs(FV)**2
                    
                    # Get eigenvalue for this mode
                    if torch.is_complex(self.Winv[idx, rank]):
                        real_winv = self.Winv[idx, rank].real  # Use .real attribute to preserve gradients
                    else:
                        real_winv = self.Winv[idx, rank]
                    
                    # Cast real_winv to the target real dtype before multiplication
                    real_winv_float64 = real_winv.to(dtype=self.real_dtype)
                    
                    # Compute intensity
                    intensity[i] = FV_abs_squared * real_winv_float64
                    
                    # Add debug print for specific indices
                    if i < 3:  # Print first few indices for debugging
                        print(f"Arbitrary-q mode: idx={idx}, intensity[{i}]={intensity[i].item():.8e}")
                    
                    # Add detailed debug for specific q-indices that match the grid mode debug indices
                    # Check if debug_q_indices exists in globals or locals before using it
                    debug_q_indices = globals().get('debug_q_indices', locals().get('debug_q_indices', []))
                    if idx in debug_q_indices:
                        print(f"\n--- Detailed Debug for Arbitrary-q mode at idx={idx} ---")
                        print(f"V_rank shape: {V_rank.shape}, dtype: {V_rank.dtype}")
                        print(f"F_i shape: {F_i.shape}, dtype: {F_i.dtype}")
                        print(f"F_i[0] (abs): {torch.abs(F_i[0]).item():.8e}")
                        print(f"FV (abs): {torch.abs(FV).item():.8e}")
                        print(f"FV_abs_squared: {FV_abs_squared.item():.8e}")
                        print(f"real_winv: {real_winv_float64.item():.8e}")
                        print(f"Final intensity: {intensity[i].item():.8e}")
            
            # Build full result array
            Id = torch.full((n_points,), float('nan'), dtype=self.real_dtype, device=self.device)
            # Ensure intensity has the correct dtype before assignment
            if intensity.dtype != self.real_dtype:
                intensity = intensity.to(dtype=self.real_dtype)
            Id[valid_indices] = intensity
            
            # Apply resolution mask (redundant but kept for consistency)
            Id_masked = Id.clone()
            Id_masked[~self.res_mask] = float('nan')
            
            # Save results if outdir is provided
            if outdir is not None:
                import os
                os.makedirs(outdir, exist_ok=True)
                torch.save(Id_masked, os.path.join(outdir, f"rank_{rank:05d}_torch.pt"))
                np.save(os.path.join(outdir, f"rank_{rank:05d}.npy"), Id_masked.detach().cpu().numpy())
            
            return Id_masked
        
        else:
            # Grid-based mode implementation
            # Get Brillouin zone dimensions
            h_dim = int(self.hsampling[2])
            k_dim = int(self.ksampling[2])
            l_dim = int(self.lsampling[2])
            total_k_points = h_dim * k_dim * l_dim
            
            # Verify V and Winv shapes match total_k_points
            if self.V.shape[0] != total_k_points or self.Winv.shape[0] != total_k_points:
                raise ValueError(f"V and Winv shapes ({self.V.shape[0]}, {self.Winv.shape[0]}) do not match total_k_points ({total_k_points})")
            
            logging.debug(f"[apply_disorder] Grid mode: h_dim={h_dim}, k_dim={k_dim}, l_dim={l_dim}, total_k_points={total_k_points}")
            
            # Pre-compute all ASU data to avoid repeated tensor creation
            asu_data = []
            for i_asu in range(self.n_asu):
                asu_data.append({
                    'xyz': self.array_to_tensor(self.crystal.get_asu_xyz(i_asu), dtype=self.real_dtype),
                    'ff_a': self.array_to_tensor(self.model.ff_a[i_asu], dtype=self.real_dtype),
                    'ff_b': self.array_to_tensor(self.model.ff_b[i_asu], dtype=self.real_dtype),
                    'ff_c': self.array_to_tensor(self.model.ff_c[i_asu], dtype=self.real_dtype),
                    'project': self.Amat[i_asu]
                })
            
            logging.debug("[apply_disorder] Starting loops over Brillouin zone...")
            
            # Restore nested loops over Brillouin zone
            # Define target indices for debugging
            target_idx = 2
            target_q_idx = 9
            
            for dh in range(h_dim):
                for dk in range(k_dim):
                    for dl in range(l_dim):
                        # Calculate the flat Brillouin zone index
                        idx = self._3d_to_flat_indices_bz(
                            torch.tensor([dh], device=self.device, dtype=torch.long),
                            torch.tensor([dk], device=self.device, dtype=torch.long),
                            torch.tensor([dl], device=self.device, dtype=torch.long)
                        ).item()
                        
                        # Debug print for target BZ index
                        if idx == target_idx:
                            print(f"\n--- GRID MODE (idx={idx}) ---")
                            q_indices = self._at_kvec_from_miller_points((dh, dk, dl))
                            valid_mask_for_q = self.res_mask[q_indices]
                            valid_indices = q_indices[valid_mask_for_q]
                            print(f"Corresponding q_indices (first 5): {q_indices[:5].cpu().numpy()}")
                            print(f"Corresponding valid_indices (first 5): {valid_indices[:5].cpu().numpy()}")
                        
                        # Get phonon modes for this k-vector
                        V_k = self.V[idx].to(self.complex_dtype)  # shape [n_dof, n_dof]
                        Winv_k = self.Winv[idx].to(self.complex_dtype)  # shape [n_dof]
                        
                        # Debug print for target BZ index
                        if idx == target_idx:
                            print(f"V_k shape: {V_k.shape}, dtype: {V_k.dtype}")
                            print(f"Winv_k shape: {Winv_k.shape}, dtype: {Winv_k.dtype}")
                            print(f"V_k[0,0] (abs): {torch.abs(V_k[0,0]).item():.6e}")
                            print(f"Winv_k[0] (real): {Winv_k[0].real.item():.6e}")
                        
                        # Debug print for specific BZ point
                        if dh == 0 and dk == 1 and dl == 0:
                            print(f"\n--- PyTorch Debug for BZ point (0,1,0) ---")
                            print(f"BZ flat index: {idx}")
                            print(f"V_k shape: {V_k.shape}, dtype: {V_k.dtype}")
                            print(f"Winv_k shape: {Winv_k.shape}, dtype: {Winv_k.dtype}")
                            print(f"V_k[0,0] (abs): {torch.abs(V_k[0,0]).item():.8e}")
                            print(f"Winv_k[0]: {Winv_k[0].item():.8e}")
                            
                            # Additional debug info for comparison with arbitrary-q mode
                            print(f"kvec for BZ point (0,1,0): {self.kvec[idx].detach().cpu().numpy()}")
                        
                        # Get q-indices for this k-vector point
                        q_indices = self._at_kvec_from_miller_points((dh, dk, dl))
                        
                        # Debug print for specific BZ point
                        if dh == 0 and dk == 1 and dl == 0:
                            print(f"q_indices shape: {q_indices.shape}")
                            if q_indices.numel() > 0:
                                print(f"First few q_indices: {q_indices[:5].cpu().numpy()}")
                                # Store these indices for comparison with arbitrary-q mode
                                debug_q_indices = q_indices[:5].cpu().numpy()
                        
                        # Skip if no q-indices found
                        if q_indices.numel() == 0:
                            continue
                        
                        # Apply resolution mask
                        valid_mask_for_q = self.res_mask[q_indices]
                        valid_indices = q_indices[valid_mask_for_q]
                        
                        # Skip if no valid indices after mask
                        if valid_indices.numel() == 0:
                            continue
                        
                        # Compute structure factors for all ASUs in parallel
                        # Initialize F with the CORRECT high-precision complex dtype
                        F = torch.zeros((valid_indices.numel(), self.n_asu, self.n_dof_per_asu),
                                      dtype=self.complex_dtype, device=self.device)
                        
                        # Process all ASUs with pre-computed data
                        for i_asu in range(self.n_asu):
                            asu = asu_data[i_asu]
                            # Get q-vectors for valid indices
                            q_vectors = self.q_grid[valid_indices].to(dtype=self.real_dtype)
                            xyz = asu['xyz'].to(dtype=self.real_dtype)
                            ff_a = asu['ff_a'].to(dtype=self.real_dtype)
                            ff_b = asu['ff_b'].to(dtype=self.real_dtype)
                            ff_c = asu['ff_c'].to(dtype=self.real_dtype)
                            adp = ADP.to(dtype=self.real_dtype)
                            project = asu['project'].to(dtype=self.real_dtype)
                            
                            # Compute structure factors (expects float64 inputs, returns complex128)
                            sf_result = structure_factors(
                                q_vectors, xyz, ff_a, ff_b, ff_c, U=adp,
                                n_processes=self.n_processes, compute_qF=True,
                                project_on_components=project, sum_over_atoms=False
                            )
                            # Assign result to F slice with explicit dtype cast
                            F[:, i_asu, :] = sf_result.to(self.complex_dtype)
                        
                        # Reshape for matrix operations
                        F = F.reshape((valid_indices.numel(), self.n_asu * self.n_dof_per_asu))
                        
                        # Debug print for specific BZ point
                        if dh == 0 and dk == 1 and dl == 0 and valid_indices.numel() > 0:
                            print(f"F shape: {F.shape}, dtype: {F.dtype}")
                            print(f"F[0,0] (abs): {torch.abs(F[0,0]).item() if F.numel() > 0 else 'empty':.8e}")
                            print(f"valid_indices shape: {valid_indices.shape}")
                            print(f"First few valid_indices: {valid_indices[:5].cpu().numpy()}")
                        
                        # Apply disorder model depending on rank parameter
                        if rank == -1:
                            # Debug print for target BZ index and q_idx
                            if idx == target_idx:
                                # Find the position of target_q_idx within the valid_indices for this BZ point
                                target_pos_in_valid = torch.where(valid_indices == target_q_idx)[0]
                                
                                if target_pos_in_valid.numel() > 0:
                                    pos = target_pos_in_valid[0].item()
                                    print(f"--- GRID MODE (idx={idx}, q_idx={target_q_idx}, pos={pos}) ---")
                                    print(f"F shape: {F.shape}, dtype: {F.dtype}")
                                    print(f"F[{pos}, 0] (abs): {torch.abs(F[pos, 0]).item():.6e}")
                                else:
                                    print(f"--- GRID MODE (idx={idx}): target_q_idx={target_q_idx} not found in valid_indices for this BZ point.")
                            
                            # Compute F·V for all modes at once
                            FV = torch.matmul(F, V_k)
                            
                            # Calculate absolute squared values
                            FV_abs_squared = torch.abs(FV)**2
                            
                            # Extract real part of eigenvalues
                            real_winv = Winv_k.real if torch.is_complex(Winv_k) else Winv_k
                            real_winv = real_winv.to(self.real_dtype)
                            
                            # Debug print for target BZ index and q_idx
                            if idx == target_idx:
                                target_pos_in_valid = torch.where(valid_indices == target_q_idx)[0]
                                if target_pos_in_valid.numel() > 0:
                                    pos = target_pos_in_valid[0].item()
                                    print(f"FV shape: {FV.shape}, dtype: {FV.dtype}")
                                    print(f"FV[{pos}, 0] (abs): {torch.abs(FV[pos, 0]).item():.6e}")
                                    print(f"real_winv shape: {real_winv.shape}, dtype: {real_winv.dtype}")
                                    print(f"real_winv[0]: {real_winv[0].item():.6e}")
                                    intensity_contributions = FV_abs_squared[pos] * real_winv
                                    print(f"Intensity Contributions[0]: {intensity_contributions[0].item():.6e}")
                                    final_intensity_for_q = torch.sum(intensity_contributions)
                                    print(f"Final Intensity Contribution for q_idx={target_q_idx}: {final_intensity_for_q.item():.6e}")
                            
                            # Debug print for specific BZ point
                            if dh == 0 and dk == 1 and dl == 0 and valid_indices.numel() > 0:
                                print(f"FV shape: {FV.shape}, dtype: {FV.dtype}")
                                print(f"FV_abs_squared shape: {FV_abs_squared.shape}")
                                print(f"real_winv shape: {real_winv.shape}, dtype: {real_winv.dtype}")
                                print(f"FV[0,0] (abs): {torch.abs(FV[0,0]).item() if FV.numel() > 0 else 'empty':.8e}")
                                print(f"FV_abs_squared[0,0]: {FV_abs_squared[0,0].item() if FV_abs_squared.numel() > 0 else 'empty':.8e}")
                                print(f"real_winv[0]: {real_winv[0].item() if real_winv.numel() > 0 else 'empty':.8e}")
                                
                                # Print a few more values for detailed comparison
                                for i in range(min(5, FV.shape[1])):
                                    print(f"  Grid mode: FV[0,{i}] (abs): {torch.abs(FV[0,i]).item():.8e}")
                                    print(f"  Grid mode: FV_abs_squared[0,{i}]: {FV_abs_squared[0,i].item():.8e}")
                                    print(f"  Grid mode: real_winv[{i}]: {real_winv[i].item():.8e}")
                                    print(f"  Grid mode: contribution[0,{i}]: {(FV_abs_squared[0,i] * real_winv[i]).item():.8e}")
                            
                            # Weight by eigenvalues and sum
                            intensity_contribution = torch.sum(FV_abs_squared * real_winv, dim=1)
                            
                            # Debug print for specific BZ point
                            if dh == 0 and dk == 1 and dl == 0 and valid_indices.numel() > 0:
                                print(f"intensity_contribution shape: {intensity_contribution.shape}")
                                print(f"intensity_contribution[0]: {intensity_contribution[0].item() if intensity_contribution.numel() > 0 else 'empty':.8e}")
                                print(f"Adding intensity to Id at indices: {valid_indices[:5].cpu().numpy()}")
                        else:
                            # Get specific mode
                            V_k_rank = V_k[:, rank]
                            Winv_k_rank = Winv_k[rank]
                            
                            # Compute FV for single mode
                            FV = torch.matmul(F, V_k_rank)
                            
                            # Calculate absolute squared value
                            FV_abs_squared = torch.abs(FV)**2
                            
                            # Extract real part of eigenvalue
                            real_winv = Winv_k_rank.real if torch.is_complex(Winv_k_rank) else Winv_k_rank
                            real_winv = real_winv.to(self.real_dtype)
                            
                            # Compute intensity
                            intensity_contribution = FV_abs_squared * real_winv
                        
                        # Ensure correct dtype before accumulation
                        intensity_contribution = intensity_contribution.to(dtype=self.real_dtype)
                        
                        # Accumulate intensity
                        Id.index_add_(0, valid_indices, intensity_contribution)
            
            # Apply resolution mask
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

    def array_to_tensor(self, array: Union[np.ndarray, torch.Tensor], requires_grad: bool = True, dtype=None) -> torch.Tensor:
        """
        Convert a NumPy array to a PyTorch tensor with gradient support.
        
        This is a helper method that ensures consistent tensor conversion
        throughout the class, with proper handling of gradient requirements.
        
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
            dtype = torch.float64
            
        # Handle complex arrays
        if isinstance(array, np.ndarray) and np.iscomplexobj(array):
            dtype = torch.complex128
        
        # If already a tensor, move to correct device and set requires_grad
        if isinstance(array, torch.Tensor):
            tensor = array.to(device=self.device, dtype=dtype)
            if requires_grad and tensor.dtype.is_floating_point:
                tensor.requires_grad_(True)
            return tensor
        
        # Handle empty arrays
        if isinstance(array, np.ndarray) and array.size == 0:
            tensor = torch.from_numpy(array.copy()).to(dtype=dtype, device=self.device)
            return tensor
        
        # Create tensor and set requires_grad if appropriate
        tensor = torch.tensor(array, dtype=dtype, device=self.device)
        if requires_grad and tensor.dtype.is_floating_point:
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
        
    def _compute_bz_averaged_adp(self) -> torch.Tensor:
        """
        Calculates the BZ-averaged ADP based on the GNM model.

        This method performs a calculation similar to the grid-mode
        compute_covariance_matrix but focuses only on deriving the
        correctly averaged ADP, integrating over the Brillouin Zone
        defined by self.hsampling, self.ksampling, self.lsampling.

        Returns:
            torch.Tensor: The calculated BZ-averaged ADP (real_dtype).

        Raises:
            ValueError: If sampling parameters are not defined on self.
        """
        logging.debug("Starting _compute_bz_averaged_adp calculation...")

        # 1. Check for required sampling parameters
        if self.hsampling is None or self.ksampling is None or self.lsampling is None:
            raise ValueError("Cannot compute BZ-averaged ADP without hsampling, ksampling, and lsampling defined.")

        # 2. Get BZ dimensions and total points
        h_dim_bz = int(self.hsampling[2])
        k_dim_bz = int(self.ksampling[2])
        l_dim_bz = int(self.lsampling[2])
        total_k_points = h_dim_bz * k_dim_bz * l_dim_bz
        logging.debug(f"  BZ dimensions: {h_dim_bz}x{k_dim_bz}x{l_dim_bz}, Total points: {total_k_points}")

        # 3. Generate BZ k-vectors (ensure float64)
        A_inv_tensor = torch.tensor(self.model.A_inv, dtype=self.real_dtype, device=self.device)
        h_coords = torch.tensor([self._center_kvec(dh, h_dim_bz) for dh in range(h_dim_bz)], device=self.device, dtype=self.real_dtype)
        k_coords = torch.tensor([self._center_kvec(dk, k_dim_bz) for dk in range(k_dim_bz)], device=self.device, dtype=self.real_dtype)
        l_coords = torch.tensor([self._center_kvec(dl, l_dim_bz) for dl in range(l_dim_bz)], device=self.device, dtype=self.real_dtype)
        h_grid, k_grid, l_grid = torch.meshgrid(h_coords, k_coords, l_coords, indexing='ij')
        hkl_fractional = torch.stack([h_grid.flatten(), k_grid.flatten(), l_grid.flatten()], dim=1).to(dtype=self.real_dtype)
        kvec_bz = torch.matmul(hkl_fractional, A_inv_tensor) # Shape: [total_k_points, 3]
        logging.debug(f"  Generated kvec_bz, shape={kvec_bz.shape}, dtype={kvec_bz.dtype}")

        # 4. Instantiate GaussianNetworkModelTorch helper
        from eryx.pdb_torch import GaussianNetworkModel as GaussianNetworkModelTorch # Local import ok
        gnm_torch = GaussianNetworkModelTorch()
        gnm_torch.n_asu = self.n_asu
        gnm_torch.n_atoms_per_asu = self.n_atoms_per_asu # Ensure this is set correctly
        gnm_torch.n_cell = self.n_cell
        gnm_torch.id_cell_ref = self.id_cell_ref
        gnm_torch.device = self.device
        gnm_torch.real_dtype = self.real_dtype
        gnm_torch.complex_dtype = self.complex_dtype
        gnm_torch.crystal = self.crystal # Pass the crystal object/dict
        # Set gamma and neighbors from self.gnm (the NumPy one)
        gnm_torch.gamma = torch.tensor(self.gnm.gamma, dtype=self.real_dtype, device=self.device)
        gnm_torch.asu_neighbors = self.gnm.asu_neighbors

        # 5. Compute Hessian (ensure complex128)
        hessian = self.compute_hessian().to(self.complex_dtype)
        logging.debug(f"  Computed hessian, shape={hessian.shape}, dtype={hessian.dtype}")

        # 6. Compute Kinv for all BZ k-vectors (ensure complex128)
        Kinv_all_bz = gnm_torch.compute_Kinv(hessian, kvec_bz, reshape=False).to(self.complex_dtype)
        logging.debug(f"  Computed Kinv_all_bz, shape={Kinv_all_bz.shape}, dtype={Kinv_all_bz.dtype}")

        # 7. Average Kinv at zero cell offset (r_d=0 => eikr=1)
        avg_Kinv_at_0 = torch.mean(Kinv_all_bz, dim=0) # Average over the BZ points
        logging.debug(f"  Computed avg_Kinv_at_0, shape={avg_Kinv_at_0.shape}, dtype={avg_Kinv_at_0.dtype}")

        # 8. Extract diagonal (ensure complex128)
        diagonal_values = torch.diagonal(avg_Kinv_at_0, dim1=0, dim2=1)
        logging.debug(f"  Extracted diagonal_values, shape={diagonal_values.shape}, dtype={diagonal_values.dtype}")

        # 9. Calculate ADP (projection, sum, scale) - ensure float64
        # Use .real attribute for gradient safety
        ADP_unscaled = diagonal_values.real.to(self.real_dtype)
        logging.debug(f"  ADP_unscaled (real diagonal), shape={ADP_unscaled.shape}, dtype={ADP_unscaled.dtype}")

        # Ensure self.Amat is float64
        Amat_proj = torch.transpose(self.Amat, 0, 1).reshape(self.n_dof_per_asu_actual, self.n_asu * self.n_dof_per_asu).to(self.real_dtype)
        ADP_unscaled = torch.matmul(Amat_proj, ADP_unscaled)
        logging.debug(f"  ADP_unscaled (after Amat), shape={ADP_unscaled.shape}, dtype={ADP_unscaled.dtype}")

        ADP_unscaled = torch.sum(ADP_unscaled.reshape(int(ADP_unscaled.shape[0] / 3), 3), dim=1)
        logging.debug(f"  ADP_unscaled (after spatial sum), shape={ADP_unscaled.shape}, dtype={ADP_unscaled.dtype}")

        # Scaling (match NumPy logic exactly)
        model_adp_tensor = torch.tensor(self.model.adp[0], dtype=self.real_dtype, device=self.device) # Use index 0 for single conformer ADP
        valid_adp = ADP_unscaled[~torch.isnan(ADP_unscaled)]
        if valid_adp.numel() > 0:
            adp_mean = torch.mean(valid_adp)
            if adp_mean.abs() < 1e-9: # Avoid division by near-zero
                 ADP_scale = torch.tensor(1.0, device=self.device, dtype=self.real_dtype)
                 logging.warning("  ADP mean is near zero, using scaling factor 1.0")
            else:
                 ADP_scale = torch.mean(model_adp_tensor) / (8 * torch.pi * torch.pi * adp_mean / 3)
        else:
            ADP_scale = torch.tensor(1.0, device=self.device, dtype=self.real_dtype)
            logging.warning("  All ADP values were NaN, using scaling factor 1.0")

        logging.debug(f"  ADP scaling factor: {ADP_scale.item():.6e}")
        final_ADP = ADP_unscaled * ADP_scale
        logging.debug(f"  Calculated final_ADP, shape={final_ADP.shape}, dtype={final_ADP.dtype}")

        return final_ADP

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
