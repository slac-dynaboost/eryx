"""
PyTorch implementation of disorder models for diffuse scattering calculations.

This module contains PyTorch versions of the disorder models defined in
eryx/models.py. All implementations maintain the same API as the NumPy versions
but use PyTorch tensors and operations to enable gradient flow.

References:
    - Original NumPy implementation in eryx/models.py
"""

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
    
    References:
        - Original NumPy implementation in eryx/models.py:OnePhonon
    """
    
    #@debug
    def __init__(self, pdb_path: str, hsampling: Tuple[float, float, float], 
                 ksampling: Tuple[float, float, float], lsampling: Tuple[float, float, float],
                 expand_p1: bool = True, group_by: str = 'asu',
                 res_limit: float = 0., model: str = 'gnm',
                 gnm_cutoff: float = 4., gamma_intra: float = 1., gamma_inter: float = 1.,
                 batch_size: int = 10000, n_processes: int = 8, device: Optional[torch.device] = None):
        """
        Initialize the OnePhonon model with PyTorch tensors.
        
        Args:
            pdb_path: Path to coordinates file.
            hsampling: Tuple (hmin, hmax, oversampling) for h dimension.
            ksampling: Tuple (kmin, kmax, oversampling) for k dimension.
            lsampling: Tuple (lmin, lmax, oversampling) for l dimension.
            expand_p1: If True, expand to p1 (if PDB is asymmetric unit).
            group_by: Level of rigid-body assembly ('asu' or None).
            res_limit: High-resolution limit in Angstrom.
            model: Chosen phonon model ('gnm' or 'rb').
            gnm_cutoff: Distance cutoff for GNM in Angstrom.
            gamma_intra: Spring constant for intra-asu interactions.
            gamma_inter: Spring constant for inter-asu interactions.
            batch_size: Number of q-vectors to evaluate per batch.
            n_processes: Number of processes for parallel computation.
            device: PyTorch device to use (default: CUDA if available, else CPU).
        """
        self.hsampling = hsampling
        self.ksampling = ksampling
        self.lsampling = lsampling
        self.batch_size = batch_size
        self.n_processes = n_processes
        self.model_type = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self._setup(pdb_path, expand_p1, res_limit, group_by)
        self._setup_phonons(pdb_path, model, gnm_cutoff, gamma_intra, gamma_inter)
    
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
        # Create an AtomicModel instance from the NP implementation.
        self.model = AtomicModel(pdb_path, expand_p1)
        
        # Store a reference to the original model for accessing original data
        self.original_model = self.model
        
        # Use PDBToTensor adapter to extract element weights
        from eryx.adapters import PDBToTensor
        pdb_adapter = PDBToTensor(device=self.device)
        element_weights = pdb_adapter.extract_element_weights(self.model)
        self.model.element_weights = element_weights
        print(f"Extracted {len(element_weights)} element weights using PDBToTensor adapter")
        
        # Build the grid of hkl indices using the NP generate_grid.
        from eryx.map_utils import generate_grid, get_resolution_mask
        hkl_grid, self.map_shape = generate_grid(self.model.A_inv, 
                                                 self.hsampling,
                                                 self.ksampling,
                                                 self.lsampling,
                                                 return_hkl=True)
        # Convert the hkl grid to a torch tensor.
        self.hkl_grid = torch.tensor(hkl_grid, dtype=torch.float32, device=self.device)
        
        # Obtain resolution mask (converted to a torch bool tensor).
        res_mask, _ = get_resolution_mask(self.model.cell, hkl_grid, res_limit)
        self.res_mask = torch.tensor(res_mask, dtype=torch.bool, device=self.device)
        
        # Compute q-grid as: 2π * A_inv^T * hkl_grid^T.
        self.q_grid = 2 * torch.pi * torch.matmul(
            torch.tensor(self.model.A_inv, dtype=torch.float32, device=self.device).T,
            self.hkl_grid.T
        ).T
        
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
        self.n_atoms_per_asu = self.model.xyz.shape[1]
        self.n_dof_per_asu_actual = self.n_atoms_per_asu * 3
        
        self.group_by = group_by
        if self.group_by is None:
            self.n_dof_per_asu = self.n_dof_per_asu_actual
        else:
            self.n_dof_per_asu = 6
        self.n_dof_per_cell = self.n_asu * self.n_dof_per_asu
    
    #@debug
    def _setup_phonons(self, pdb_path: str, model: str, 
                       gnm_cutoff: float, gamma_intra: float, gamma_inter: float):
        """
        Compute phonons using a Gaussian Network Model.
        
        Parameters:
            pdb_path: Path to coordinates file.
            model: Chosen phonon model ('gnm' or 'rb').
            gnm_cutoff: Distance cutoff for GNM.
            gamma_intra: Spring constant for intra-asu interactions.
            gamma_inter: Spring constant for inter-asu interactions.
        """
        # Use the sampling parameters directly.
        h_dim = int(self.hsampling[2])
        k_dim = int(self.ksampling[2])
        l_dim = int(self.lsampling[2])
        
        self.kvec = torch.zeros((h_dim, k_dim, l_dim, 3), device=self.device)
        self.kvec_norm = torch.zeros((h_dim, k_dim, l_dim, 1), device=self.device)
        
        # Initialize tensors for phonon calculations.
        self.V = torch.zeros((h_dim, k_dim, l_dim,
                               self.n_asu * self.n_dof_per_asu,
                               self.n_asu * self.n_dof_per_asu),
                              dtype=torch.complex64, device=self.device)
        self.Winv = torch.zeros((h_dim, k_dim, l_dim,
                                  self.n_asu * self.n_dof_per_asu),
                                 dtype=torch.complex64, device=self.device)
        
        self._build_A()
        self._build_M()
        self._build_kvec_Brillouin()
        
        if model == 'gnm':
            # Store parameters as tensors with gradients
            if isinstance(gamma_intra, torch.Tensor):
                self.gamma_intra = gamma_intra
            else:
                self.gamma_intra = torch.tensor(gamma_intra, dtype=torch.float32, device=self.device, requires_grad=True)
                
            if isinstance(gamma_inter, torch.Tensor):
                self.gamma_inter = gamma_inter
            else:
                self.gamma_inter = torch.tensor(gamma_inter, dtype=torch.float32, device=self.device, requires_grad=True)
            
            # Setup GNM from NP implementation for initialization only
            self.gnm = GaussianNetworkModel(pdb_path, gnm_cutoff, 
                                           float(self.gamma_intra.detach().cpu().numpy()), 
                                           float(self.gamma_inter.detach().cpu().numpy()))
            
            # Create a differentiable gamma tensor that matches the GNM structure
            self.gamma_tensor = torch.zeros((self.n_cell, self.n_asu, self.n_asu), 
                                           device=self.device, dtype=torch.float32)
            
            # Fill it like the original build_gamma method, but with our parameter tensors
            for i_asu in range(self.n_asu):
                for i_cell in range(self.n_cell):
                    for j_asu in range(self.n_asu):
                        self.gamma_tensor[i_cell, i_asu, j_asu] = self.gamma_inter
                        if (i_cell == self.id_cell_ref) and (j_asu == i_asu):
                            self.gamma_tensor[i_cell, i_asu, j_asu] = self.gamma_intra
            
            self.compute_gnm_phonons()
            self.compute_covariance_matrix()
        else:
            self.compute_rb_phonons()
    
    #@debug
    def _build_A(self):
        """
        Build the displacement projection matrix A that projects rigid-body
        displacements to individual atomic displacements.
        """
        if self.group_by == 'asu':
            # Initialize Amat with zeros, using float64 for better precision
            self.Amat = torch.zeros((self.n_asu, self.n_dof_per_asu_actual, self.n_dof_per_asu), 
                                   device=self.device, dtype=torch.float64)
            
            # Create identity matrix for translations
            Adiag = torch.eye(3, device=self.device, dtype=torch.float64)
            
            for i_asu in range(self.n_asu):
                # Get coordinates from model directly - simplest reliable approach
                if hasattr(self.model, 'xyz'):
                    if isinstance(self.model.xyz, torch.Tensor):
                        xyz = self.model.xyz[i_asu].to(dtype=torch.float64)
                    else:
                        xyz = torch.tensor(self.model.xyz[i_asu], dtype=torch.float64, device=self.device)
                else:
                    # Fallback to zeros if no coordinates available
                    xyz = torch.zeros((self.n_atoms_per_asu, 3), device=self.device, dtype=torch.float64)
                
                # Center coordinates properly
                xyz = xyz - xyz.mean(dim=0, keepdim=True)
                
                # Initialize Atmp once per asymmetric unit (outside the atom loop)
                Atmp = torch.zeros((3, 3), device=self.device, dtype=torch.float64)
                
                # Initialize Atmp once per asymmetric unit (outside the atom loop)
                # This allows cumulative updates across atoms, matching NumPy implementation
                Atmp = torch.zeros((3, 3), device=self.device, dtype=torch.float64)
                
                # Process each atom
                for i_atom in range(self.n_atoms_per_asu):
                    # Update skew-symmetric matrix for rotations first
                    if i_atom < xyz.shape[0]:
                        Atmp[0, 1] = xyz[i_atom, 2]  
                        Atmp[0, 2] = -xyz[i_atom, 1]
                        Atmp[1, 2] = xyz[i_atom, 0]
                        Atmp = Atmp - Atmp.transpose(0, 1)
                    
                    # Set identity part (translations) and then the rotation part
                    self.Amat[i_asu, i_atom*3:(i_atom+1)*3, 0:3] = Adiag
                    self.Amat[i_asu, i_atom*3:(i_atom+1)*3, 3:6] = Atmp
            
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
            
            # Convert back to float32 for consistency with the rest of the model
            # while preserving the higher-precision computation
            self.Amat = self.Amat.to(dtype=torch.float32)
            
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
            
            # Convert back to float32 for consistency
            self.Linv = self.Linv.to(dtype=torch.float32)
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
                print(f"Cholesky decomposition failed: {e}")
                print(f"Matrix condition number: {torch.linalg.cond(Mmat_reg).item()}")
                
                # Add stronger regularization and try again
                stronger_eps = 1e-4
                Mmat_reg = Mmat + stronger_eps * eye
                try:
                    L = torch.linalg.cholesky(Mmat_reg)
                    self.Linv = torch.linalg.inv(L)
                    print("Succeeded with stronger regularization")
                except RuntimeError:
                    # Final fallback to SVD approach
                    print("Falling back to SVD decomposition")
                    U, S, V = torch.linalg.svd(Mmat_reg, full_matrices=False)
                    S = torch.clamp(S, min=1e-8)
                    self.Linv = U @ torch.diag(1.0 / torch.sqrt(S)) @ V
            
            # Convert back to float32 for consistency
            self.Linv = self.Linv.to(dtype=torch.float32)
            self.Linv.requires_grad_(True)
    
    #@debug
    def _build_M_allatoms(self) -> torch.Tensor:
        """
        Build the all-atom mass matrix M_0.
        
        Returns:
            torch.Tensor of shape (n_asu, n_dof_per_asu_actual, n_asu, n_dof_per_asu_actual)
        """
        # Use float64 for better precision
        dtype = torch.float64
        
        # Create mass array - default to ones as fallback
        mass_array = torch.ones(self.n_asu * self.n_atoms_per_asu, dtype=dtype, device=self.device)
        
        # Try to extract weights using prioritized strategies
        weights = []
        
        # Strategy 1: Use element_weights if available from PDBToTensor adapter
        if hasattr(self.model, 'element_weights') and isinstance(self.model.element_weights, torch.Tensor):
            try:
                print("Using element_weights from model")
                weights = self.model.element_weights.detach().cpu().tolist()
            except Exception as e:
                print(f"Error using element_weights: {e}")
        
        # Strategy 2: Try to get atomic weights directly from the original model
        if not weights and hasattr(self, 'original_model') and hasattr(self.original_model, '_gemmi_structure'):
            try:
                print("Extracting weights from original gemmi structure")
                import gemmi
                structure = self.original_model._gemmi_structure
                for model in structure:
                    for chain in model:
                        for residue in chain:
                            for atom in residue:
                                element = atom.element
                                if element and hasattr(element, 'weight'):
                                    weights.append(float(element.weight))
                                else:
                                    # Default to carbon weight if element is unknown
                                    weights.append(12.0)
            except Exception as e:
                print(f"Error getting weights from gemmi structure: {e}")
        
        # Strategy 3: Try to extract from model.elements if available
        if not weights and hasattr(self.model, 'elements'):
            try:
                print("Extracting weights from model.elements")
                # Handle various formats of elements data
                if isinstance(self.model.elements, list) and len(self.model.elements) > 0:
                    # List of lists (original format)
                    if isinstance(self.model.elements[0], list):
                        for structure in self.model.elements:
                            for element in structure:
                                if hasattr(element, 'weight'):
                                    weights.append(float(element.weight))
                                elif isinstance(element, dict) and 'weight' in element:
                                    weights.append(float(element['weight']))
                                elif isinstance(element, (float, int, np.number)):
                                    weights.append(float(element))
                    # Direct list of elements or weights
                    elif all(hasattr(e, 'weight') for e in self.model.elements if hasattr(e, '__dict__')):
                        weights = [float(e.weight) for e in self.model.elements]
                    elif all(isinstance(e, (float, int, np.number)) for e in self.model.elements):
                        weights = [float(e) for e in self.model.elements]
            except Exception as e:
                print(f"Error extracting weights from model.elements: {e}")
        
        # Strategy 4: Use cached original weights
        if not weights and hasattr(self.model, '_original_weights'):
            print(f"Using cached original weights")
            weights = self.model._original_weights
        
        # If we have weights, use them
        if weights:
            # Check if all weights are zero, which indicates a problem
            if all(w == 0.0 for w in weights):
                print(f"WARNING: All extracted weights are zero! Using default atomic weights instead.")
                # Use standard atomic weights as fallback
                weights = [12.0] * len(weights)  # Carbon weight as default
            
            if len(weights) < self.n_asu * self.n_atoms_per_asu:
                print(f"Warning: Not enough weights ({len(weights)}) for all atoms ({self.n_asu * self.n_atoms_per_asu}). Using default weight for remaining atoms.")
                # Pad with carbon weights
                weights.extend([12.0] * (self.n_asu * self.n_atoms_per_asu - len(weights)))
            
            print(f"Using weights: min={min(weights)}, max={max(weights)}, count={len(weights)}")
            mass_array = torch.tensor(weights[:self.n_asu * self.n_atoms_per_asu], dtype=dtype, device=self.device)
        else:
            print("No weights found. Using default weights (ones).")
        
        # Create block diagonal matrix
        eye3 = torch.eye(3, device=self.device, dtype=dtype)
        blocks = []
        for i in range(self.n_asu * self.n_atoms_per_asu):
            blocks.append(mass_array[i] * eye3)
        
        try:
            # Use torch.block_diag if available (PyTorch 1.8+)
            M_block_diag = torch.block_diag(*blocks)
        except (AttributeError, RuntimeError):
            # Fallback for older PyTorch versions
            total_dim = self.n_asu * self.n_atoms_per_asu * 3
            M_block_diag = torch.zeros((total_dim, total_dim), device=self.device, dtype=dtype)
            for i in range(self.n_asu * self.n_atoms_per_asu):
                start_idx = i * 3
                M_block_diag[start_idx:start_idx+3, start_idx:start_idx+3] = mass_array[i] * eye3
        
        # Reshape to 4D tensor
        M_allatoms = M_block_diag.reshape(self.n_asu, self.n_dof_per_asu_actual,
                                        self.n_asu, self.n_dof_per_asu_actual)
        
        # Set requires_grad
        M_allatoms.requires_grad_(True)
        
        return M_allatoms
    
    ##@debug
    def _project_M(self, M_allatoms: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """
        Project all-atom mass matrix M_0 using the A matrix: M = A.T M_0 A
        
        Args:
            M_allatoms: Mass matrix of shape (n_asu, n_dof_per_asu_actual, n_asu, n_dof_per_asu_actual)
            
        Returns:
            Mmat: Projected mass matrix of shape (n_asu, n_dof_per_asu, n_asu, n_dof_per_asu)
        """
        # Use the same precision as M_allatoms for consistency
        if isinstance(M_allatoms, torch.Tensor):
            dtype = M_allatoms.dtype
        else:
            dtype = torch.float64
        
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
        
        This implementation matches the NumPy version by regularly sampling
        [-0.5, 0.5[ for h, k and l using the sampling parameters.
        """
        # Initialize tensors
        h_dim = int(self.hsampling[2])
        k_dim = int(self.ksampling[2])
        l_dim = int(self.lsampling[2])
        
        # Create tensors with proper device placement
        self.kvec = torch.zeros((h_dim, k_dim, l_dim, 3), 
                               device=self.device)
        self.kvec_norm = torch.zeros((h_dim, k_dim, l_dim, 1), 
                                    device=self.device)
        
        # Convert A_inv to tensor properly using clone().detach() to avoid warning
        if isinstance(self.model.A_inv, torch.Tensor):
            A_inv_tensor = self.model.A_inv.clone().detach().to(dtype=torch.float32, device=self.device)
        else:
            A_inv_tensor = torch.tensor(self.model.A_inv, dtype=torch.float32, device=self.device)
        
        # Compute k-vectors
        for dh in range(h_dim):
            k_dh = self._center_kvec(dh, h_dim)
            for dk in range(k_dim):
                k_dk = self._center_kvec(dk, k_dim)
                for dl in range(l_dim):
                    k_dl = self._center_kvec(dl, l_dim)
                    # Create hkl vector exactly as in NumPy
                    hkl = np.array([k_dh, k_dk, k_dl])
                    hkl_tensor = torch.tensor(hkl, device=self.device, dtype=torch.float32)
                    
                    # Debug calculation for specific points
                    if dh == 0 and dk == 1 and dl == 0:
                        print(f"\nDEBUGGING calculation for point [0,1,0]:")
                        print(f"k_dh, k_dk, k_dl = {k_dh}, {k_dk}, {k_dl}")
                        print(f"hkl_tensor: {hkl_tensor}")
                        print(f"A_inv_tensor:\n{A_inv_tensor}")
                        print(f"A_inv_tensor.T:\n{A_inv_tensor.T}")
                        result = torch.matmul(A_inv_tensor.T, hkl_tensor)
                        print(f"Result of matmul: {result}")
                    
                    # Use the exact same calculation as NumPy: np.inner(A_inv.T, hkl).T
                    # The NumPy implementation uses np.inner which is different from a simple matmul
                    # This is the key to matching the expected values exactly
                    self.kvec[dh, dk, dl] = torch.matmul(A_inv_tensor.T, hkl_tensor)
                    
                    # Calculate norm exactly as NumPy does
                    self.kvec_norm[dh, dk, dl] = torch.norm(self.kvec[dh, dk, dl])
        
        # Set requires_grad after construction
        self.kvec.requires_grad_(True)
        self.kvec_norm.requires_grad_(True)
    
    #@debug
    def _center_kvec(self, x: int, L: int) -> float:
        """
        Center a k-vector index.
        
        For x and L integers such that 0 <= x < L, return -L/2 < x < L/2
        by applying periodic boundary condition in L/2.
        
        This matches the NumPy implementation exactly.
        """
        # Exactly match the NumPy implementation
        # The key is to use integer division and modulo operations in the same order
        return int(((x - L / 2) % L) - L / 2) / L
    
    #@debug
    def _at_kvec_from_miller_points(self, hkl_kvec: tuple) -> torch.Tensor:
        """
        Return the indices of all q-vectors that are k-vector away from given Miller indices.
        
        Args:
            hkl_kvec: Tuple of starting indices (ints).
            
        Returns:
            Torch tensor of raveled indices.
        """
        # Calculate steps based on sampling parameters
        hsteps = int(self.hsampling[2] * (self.hsampling[1] - self.hsampling[0]) + 1)
        ksteps = int(self.ksampling[2] * (self.ksampling[1] - self.ksampling[0]) + 1)
        lsteps = int(self.lsampling[2] * (self.lsampling[1] - self.lsampling[0]) + 1)
        
        # Create index grid
        h_range = torch.arange(hkl_kvec[0], hsteps, self.hsampling[2], device=self.device, dtype=torch.long)
        k_range = torch.arange(hkl_kvec[1], ksteps, self.ksampling[2], device=self.device, dtype=torch.long)
        l_range = torch.arange(hkl_kvec[2], lsteps, self.lsampling[2], device=self.device, dtype=torch.long)
        
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
    
    
    #@debug
    def compute_hessian(self) -> torch.Tensor:
        """
        Compute the projected Hessian matrix for the supercell.
        """
        hessian = torch.zeros((self.n_asu, self.n_dof_per_asu,
                               self.n_cell, self.n_asu, self.n_dof_per_asu),
                              dtype=torch.complex64, device=self.device)
        
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
        
        # Use our differentiable gamma tensor instead of the NumPy GNM gamma
        if hasattr(self, 'gamma_tensor'):
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
        eye3 = torch.eye(3, device=self.device, dtype=torch.complex64)
        
        for i_cell in range(self.n_cell):
            for i_asu in range(self.n_asu):
                for j_asu in range(self.n_asu):
                    # Apply Kronecker product with identity matrix (3x3)
                    # This expands each element of the hessian into a 3x3 block
                    h_block = hessian_allatoms[i_asu, :, i_cell, j_asu, :]
                    h_expanded = torch.zeros((h_block.shape[0] * 3, h_block.shape[1] * 3), 
                                            dtype=torch.complex64, device=self.device)
                    
                    # Manually implement the Kronecker product
                    for i in range(h_block.shape[0]):
                        for j in range(h_block.shape[1]):
                            h_expanded[i*3:(i+1)*3, j*3:(j+1)*3] = h_block[i, j] * eye3
                    
                    # Perform matrix multiplication with expanded hessian
                    proj = torch.matmul(self.Amat[i_asu].T.to(torch.complex64),
                                        torch.matmul(h_expanded,
                                                     self.Amat[j_asu].to(torch.complex64)))
                    hessian[i_asu, :, i_cell, j_asu, :] = proj
        return hessian
    
    #@debug
    def compute_gnm_phonons(self):
        """
        Compute phonon modes for each k-vector in the first Brillouin zone.
        
        This implementation uses a modified SVD approach that ensures stable gradient flow
        through the eigenvalues while avoiding problematic backpropagation through
        complex singular vectors.
        """
        hessian = self.compute_hessian()
        h_dim = int(self.hsampling[2])
        k_dim = int(self.ksampling[2])
        l_dim = int(self.lsampling[2])
        
        # Create a GaussianNetworkModel instance for K matrix calculations
        from eryx.pdb_torch import GaussianNetworkModel as GaussianNetworkModelTorch
        gnm_torch = GaussianNetworkModelTorch()
        gnm_torch.n_asu = self.n_asu
        gnm_torch.n_atoms_per_asu = self.n_atoms_per_asu
        gnm_torch.n_cell = self.n_cell
        gnm_torch.id_cell_ref = self.id_cell_ref
        gnm_torch.device = self.device
        
        # Ensure crystal is properly set
        if hasattr(self, 'crystal'):
            gnm_torch.crystal = self.crystal
        else:
            print("Warning: No crystal object found in OnePhonon model")
        
        # Convert gamma from NumPy GNM to PyTorch tensor if needed
        if hasattr(self.gnm, 'gamma'):
            from eryx.adapters import PDBToTensor
            adapter = PDBToTensor(device=self.device)
            gnm_torch.gamma = adapter.array_to_tensor(self.gnm.gamma, dtype=torch.float32)
            
            # Copy neighbor list structure
            gnm_torch.asu_neighbors = self.gnm.asu_neighbors
        
        for dh in range(h_dim):
            for dk in range(k_dim):
                for dl in range(l_dim):
                    kvec = self.kvec[dh, dk, dl]
                    Kmat = gnm_torch.compute_K(hessian, kvec=kvec)
                    Kmat_2d = Kmat.reshape((self.n_asu * self.n_dof_per_asu,
                                             self.n_asu * self.n_dof_per_asu))
                    Linv_complex = self.Linv.to(dtype=torch.complex64)
                    Dmat = torch.matmul(Linv_complex, torch.matmul(Kmat_2d, Linv_complex.T))
                    
                    # Extract eigenvalues and eigenvectors without tracking phase gradients
                    # This detaches them from the computation graph to avoid gradient issues with complex phases
                    with torch.no_grad():
                        v, w, _ = torch.linalg.svd(Dmat, full_matrices=False)
                        w = torch.sqrt(w)  # w contains singular values from SVD
                        w = torch.where(w < 1e-6,
                                       torch.tensor(float('nan'), dtype=w.dtype, device=w.device),
                                       w)
                        w = torch.flip(w, [0])
                        v = torch.flip(v, [1])
                    
                    # Recompute eigenvalues in a differentiable way
                    # This reattaches them to the computation graph for gradient flow
                    eigenvalues = []
                    for i in range(v.shape[1]):
                        v_i = v[:, i:i+1]
                        # Compute λ_i = v_i† D v_i (maintains gradient flow through magnitudes)
                        # This creates a path for gradients to flow back to model parameters
                        lambda_i = torch.matmul(torch.matmul(v_i.conj().T, Dmat), v_i).real
                        eigenvalues.append(lambda_i[0, 0])
                    
                    eig_values = torch.stack(eigenvalues)
                    
                    # Compute inverses with stability controls
                    winv_value = 1.0 / (torch.sqrt(torch.abs(eig_values)) ** 2 + 1e-8)
                    
                    # Set extremely large values to NaN for consistency with NumPy
                    winv_value = torch.where(winv_value > 1e6,
                                            torch.tensor(float('nan'), dtype=winv_value.dtype, device=winv_value.device),
                                            winv_value)
                    
                    # Transform eigenvectors to the right basis using Linv
                    # v is detached from computation graph, but that's OK since we only need
                    # its values for the forward pass (not its gradients)
                    v_value = torch.matmul(Linv_complex.T, v)
                    
                    # Use tensor indexing without in-place modification
                    self.Winv = self.Winv.clone()
                    self.V = self.V.clone()
                    self.Winv[dh, dk, dl] = winv_value  # This contains our differentiable eigenvalues
                    self.V[dh, dk, dl] = v_value  # This contains the eigenvectors (used in forward pass only)
    
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
    
    #@debug
    def compute_covariance_matrix(self):
        """
        Compute the covariance matrix for atomic displacements.
        """
        self.covar = torch.zeros((self.n_asu * self.n_dof_per_asu,
                                   self.n_cell, self.n_asu * self.n_dof_per_asu),
                                  dtype=torch.complex64, device=self.device)
        h_dim = int(self.hsampling[2])
        k_dim = int(self.ksampling[2])
        l_dim = int(self.lsampling[2])
        from eryx.torch_utils import ComplexTensorOps
        
        # Create a GaussianNetworkModelTorch instance for Kinv calculations
        gnm_torch = GaussianNetworkModelTorch()
        gnm_torch.n_asu = self.n_asu
        gnm_torch.n_cell = self.n_cell
        gnm_torch.id_cell_ref = self.id_cell_ref
        gnm_torch.device = self.device
        gnm_torch.crystal = self.crystal
        
        hessian = self.compute_hessian()
        
        for dh in range(h_dim):
            for dk in range(k_dim):
                for dl in range(l_dim):
                    kvec = self.kvec[dh, dk, dl]
                    # Use the PyTorch implementation from pdb_torch
                    Kinv = gnm_torch.compute_Kinv(hessian, kvec=kvec, reshape=False)
                    for j_cell in range(self.n_cell):
                        r_cell = self.crystal.get_unitcell_origin(self.crystal.id_to_hkl(j_cell))
                        phase = torch.sum(kvec * r_cell)
                        real_part, imag_part = ComplexTensorOps.complex_exp(phase)
                        eikr = torch.complex(real_part, imag_part)
                        self.covar[:, j_cell, :] += Kinv * eikr
        self.ADP = torch.real(torch.diagonal(self.covar[:, self.crystal.hkl_to_id([0, 0, 0]), :], dim1=0, dim2=1))
        Amat = torch.transpose(self.Amat, 0, 1).reshape(self.n_dof_per_asu_actual, self.n_asu * self.n_dof_per_asu)
        self.ADP = torch.matmul(Amat, self.ADP)
        self.ADP = torch.sum(self.ADP.reshape(int(self.ADP.shape[0] / 3), 3), dim=1)
        ADP_scale = torch.mean(self.array_to_tensor(self.model.adp)) / (8 * torch.pi * torch.pi * torch.mean(self.ADP) / 3)
        self.ADP = self.ADP * ADP_scale
        self.covar = self.covar * ADP_scale
        self.covar = torch.real(self.covar.reshape((self.n_asu, self.n_dof_per_asu,
                                                     self.n_cell, self.n_asu, self.n_dof_per_asu)))
    
    #@debug
    def apply_disorder(self, rank: int = -1, outdir: Optional[str] = None, 
                       use_data_adp: bool = False) -> torch.Tensor:
        """
        Compute the diffuse intensity using the one-phonon approximation.
        """
        import logging
        logging.info(f"apply_disorder: rank={rank}, use_data_adp={use_data_adp}")
        if use_data_adp:
            ADP = torch.tensor(self.model.adp[0], dtype=torch.float32, device=self.device) / (8 * torch.pi * torch.pi)
        else:
            ADP = self.ADP.to(dtype=torch.float32, device=self.device)
        Id = torch.zeros(self.q_grid.shape[0], dtype=torch.float32, device=self.device)
        from eryx.scatter_torch import structure_factors
        h_dim = int(self.hsampling[2])
        k_dim = int(self.ksampling[2])
        l_dim = int(self.lsampling[2])
        for dh in range(h_dim):
            for dk in range(k_dim):
                for dl in range(l_dim):
                    q_indices = self._at_kvec_from_miller_points((dh, dk, dl))
                    valid_mask = self.res_mask[q_indices]
                    valid_indices = q_indices[valid_mask]
                    if valid_indices.numel() == 0:
                        continue
                    F = torch.zeros((valid_indices.numel(), self.n_asu, self.n_dof_per_asu),
                                    dtype=torch.complex64, device=self.device)
                    for i_asu in range(self.n_asu):
                        F[:, i_asu, :] = structure_factors(
                            self.q_grid[valid_indices],
                            torch.tensor(self.crystal.get_asu_xyz(i_asu), dtype=torch.float32, device=self.device),
                            torch.tensor(self.model.ff_a[i_asu], dtype=torch.float32, device=self.device),
                            torch.tensor(self.model.ff_b[i_asu], dtype=torch.float32, device=self.device),
                            torch.tensor(self.model.ff_c[i_asu], dtype=torch.float32, device=self.device),
                            U=ADP,
                            batch_size=self.batch_size,
                            n_processes=self.n_processes,
                            compute_qF=True,
                            project_on_components=self.Amat[i_asu],
                            sum_over_atoms=False
                        )
                    F = F.reshape((valid_indices.numel(), self.n_asu * self.n_dof_per_asu))
                    if rank == -1:
                        FV = torch.matmul(F, self.V[dh, dk, dl])
                        FV_abs_squared = torch.abs(FV) ** 2
                        # Ensure NaN propagation in both real and imaginary parts
                        winv = self.Winv[dh, dk, dl]
                        real_winv = torch.real(winv)
                        # Propagate NaNs from imaginary part to real part
                        real_winv = torch.where(torch.isnan(torch.imag(winv)), 
                                               torch.tensor(float('nan'), device=self.device, dtype=real_winv.dtype),
                                               real_winv)
                        weighted_intensity = torch.matmul(FV_abs_squared, real_winv)
                        Id.index_add_(0, valid_indices, weighted_intensity)
                    else:
                        V_rank = self.V[dh, dk, dl, :, rank]
                        FV = torch.matmul(F, V_rank)
                        weighted_intensity = (torch.abs(FV) ** 2) * torch.real(self.Winv[dh, dk, dl, rank])
                        Id.index_add_(0, valid_indices, weighted_intensity)
        Id_masked = Id.clone()
        Id_masked[~self.res_mask] = float('nan')
        if outdir is not None:
            import os
            os.makedirs(outdir, exist_ok=True)
            torch.save(Id_masked, os.path.join(outdir, f"rank_{rank:05d}_torch.pt"))
            np.save(os.path.join(outdir, f"rank_{rank:05d}.npy"), Id_masked.detach().cpu().numpy())
        return Id_masked

    #@debug
    def array_to_tensor(self, array: np.ndarray, requires_grad: bool = True, dtype=None) -> torch.Tensor:
        """
        Convert a NumPy array to a Torch tensor.
        """
        if dtype is None:
            dtype = torch.float32
        tensor = torch.tensor(array, dtype=dtype, device=self.device)
        if requires_grad and tensor.dtype.is_floating_point:
            tensor.requires_grad_(True)
        return tensor

    #@debug
    def compute_rb_phonons(self):
        """
        Compute phonons for the rigid-body model.
        """
        self.compute_gnm_phonons()

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

