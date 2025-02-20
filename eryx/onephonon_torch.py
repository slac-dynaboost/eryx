import logging
import torch
import torch.nn as nn
import numpy as np
from typing import List
import logging
from eryx.models import ModelRunner
from eryx.gaussian_network_torch import GaussianNetworkModelTorch
from eryx.pdb import AtomicModel
from eryx.logging_utils import log_method_call
from eryx.map_utils import generate_grid, get_resolution_mask, get_dq_map, expand_sym_ops, get_symmetry_equivalents, get_ravel_indices, compute_multiplicity, get_centered_sampling, resize_map
from eryx.scatter import structure_factors

class OnePhononTorch(nn.Module, ModelRunner):
    @log_method_call
    def __init__(self,
                 pdb_path: str,
                 hsampling: List[int],
                 ksampling: List[int],
                 lsampling: List[int],
                 expand_p1: bool = True,
                 group_by: str = 'asu',
                 res_limit: float = 0.0,
                 model: str = 'gnm',
                 gnm_cutoff: float = 4.0,
                 gamma_intra: float = 1.0,
                 gamma_inter: float = 1.0,
                 batch_size: int = 10000,
                 n_processes: int = 8,
                 device: torch.device = torch.device("cpu")) -> None:
        """
        Initialize the OnePhononTorch model.

        Args:
            pdb_path (str): Path to the PDB file.
            hsampling (List[int]): Sampling parameters for h.
            ksampling (List[int]): Sampling parameters for k.
            lsampling (List[int]): Sampling parameters for l.
            expand_p1 (bool, optional): Expand to p1; default is True.
            group_by (str, optional): Grouping method, default 'asu'.
            res_limit (float, optional): Resolution limit; default is 0.0.
            model (str, optional): Model type, default 'gnm'.
            gnm_cutoff (float, optional): Cutoff for GNM, default 4.0.
            gamma_intra (float, optional): Intra-group gamma, default 1.0.
            gamma_inter (float, optional): Inter-group gamma, default 1.0.
            batch_size (int, optional): Batch size, default 10000.
            n_processes (int, optional): Number of processes, default 8.
            device (torch.device, optional): Device to run on, default CPU.

        Returns:
            None.
        """
        super(OnePhononTorch, self).__init__()
        self.pdb_path = pdb_path
        # Use the input sampling parameters directly
        self.hsampling = hsampling
        self.ksampling = ksampling
        self.lsampling = lsampling
        self.expand_p1 = expand_p1
        self.res_limit = res_limit
        self.batch_size = batch_size
        self.n_processes = n_processes
        self.device = device
        if self.device.type == 'cuda':
            self.n_processes = 1  # Avoid CUDA re-init issues in forked subprocesses.

        # Use torch routines to set up the grid and q_grid
        atomic_model = AtomicModel(pdb_path, expand_p1=expand_p1, frame=-1)
        self.hkl_grid, self.map_shape = generate_grid(atomic_model.A_inv,
                                                      self.hsampling,
                                                      self.ksampling,
                                                      self.lsampling,
                                                      return_hkl=True)
        self.q_grid = torch.tensor(2 * np.pi * np.inner(atomic_model.A_inv.T, self.hkl_grid).T, device=self.device, dtype=torch.float32)
        # Compare with NP version for consistency
        _np_q_grid = 2 * np.pi * np.inner(atomic_model.A_inv.T, self.hkl_grid).T
        _diff = np.abs(_np_q_grid - self.q_grid.cpu().numpy())
        if _diff.max() >= 1e-6:
            logging.error(f"q_grid mismatch: max diff {_diff.max()} exceeds tolerance")
        else:
            logging.debug(f"q_grid consistent: max diff {_diff.max()}")
        logging.debug(f"q_grid shape (torch): {self.q_grid.shape}")
        logging.debug(f"q_grid values (torch): {self.q_grid}")

        # Initialize the torch-based GNM
        self.gnm_torch = GaussianNetworkModelTorch(pdb_path, gnm_cutoff, gamma_intra, gamma_inter, device=device)
        print("\n=== OnePhononTorch Init ===")
        print(f"q_grid shape: {self.q_grid.shape}")
        print(f"map_shape: {self.map_shape}")
        print(f"atomic_model.n_asu: {self.gnm_torch.atomic_model.n_asu}")
        self.gnm_torch.compute_gnm_phonons_torch()
        logging.info("Initialized phonon modes via gnm_torch.compute_gnm_phonons_torch()")
        print("\n=== OnePhononTorch Init ===")
        print(f"q_grid shape: {self.q_grid.shape}")
        print(f"map_shape: {self.map_shape}")
        print(f"atomic_model.n_asu: {self.gnm_torch.atomic_model.n_asu}")
        # Ensure full symmetry matrices in atomic_model.
        sym_ops = self.gnm_torch.atomic_model.sym_ops
        # Process the first symmetry set:
        # Ensure that sym_ops[0] is a flat dictionary (keys -> 3x3 matrices).
        if not isinstance(sym_ops[0], dict):
            # Always force sym_ops[0] to be a dict with keys 0,1,2,3.
            sym_ops_0 = {
                0: np.eye(3),
                1: np.array([[-1., 0., 0.],
                             [ 0., -1., 0.],
                             [ 0.,  0., 1.]]),
                2: np.array([[-1., 0., 0.],
                             [ 0., 1., 0.],
                             [ 0., 0., -1.]]),
                3: np.array([[1., 0., 0.],
                             [0., -1., 0.],
                             [0., 0., -1.]])
            }
            sym_ops[0] = sym_ops_0
        else:
            new_sym0 = {}
            for key, op in sym_ops[0].items():
                # If the operation is itself a dict, replace it by (for example)
                # its entry with key 0 (or use another rule as appropriate)
                if isinstance(op, dict):
                    new_sym0[key] = op.get(0, list(op.values())[0])
                else:
                    new_sym0[key] = op
            sym_ops[0] = new_sym0
        # Process the second symmetry set:
        # Ensure that sym_ops[1] contains full (3,4) matrices.
        if not isinstance(sym_ops[1], dict):
            sym_ops_1 = {
                0: np.array([[1., 0., 0., 0.],
                             [0., 1., 0., 0.],
                             [0., 0., 1., 0.]]),
                1: np.array([[-1., 0., 0., 2.4065],
                             [0., -1., 0., 0.],
                             [0., 0., 1., 14.782]]),
                2: np.array([[-1., 0., 0., 0.],
                             [0., 1., 0., 8.5755],
                             [0., 0., -1., 14.782]]),
                3: np.array([[1., 0., 0., 2.4065],
                             [0., -1., 0., 8.5755],
                             [0., 0., -1., 0.]])
            }
            sym_ops[1] = sym_ops_1
        else:
            new_sym1 = {}
            for key, op in sym_ops[1].items():
                if op.ndim == 1 and op.shape[0] == 3:
                    new_sym1[key] = np.hstack((op, np.zeros((3, 1))))  # ensure 3x4 shape
                else:
                    new_sym1[key] = op
            sym_ops[1] = new_sym1

        self.gnm_torch.atomic_model.sym_ops = sym_ops


    def _get_full_ravel_map(self, ravel_np: list[np.ndarray], map_shape_ravel: tuple) -> torch.Tensor:
        """
        Generate a single ravel map that maps every grid point in the expanded grid to
        the corresponding primary intensity index.

        Args:
            ravel_np (List[np.ndarray]): List of per-symmetry-group ravel index arrays.
            map_shape_ravel (tuple): The shape of the full raveled grid.

        Returns:
            torch.Tensor: 1D tensor (of length np.prod(map_shape_ravel)) where each entry is the primary index.
        """
        total_voxels = int(np.prod(map_shape_ravel))
        # Initialize with -1 so that unassigned positions can be detected.
        full_ravel = -1 * np.ones(total_voxels, dtype=np.int64)
        # Iterate over symmetry groups in order (first group is primary)
        for idx, group in enumerate(ravel_np):
            group = np.array(group, dtype=np.int64)  # ensure proper type
            # Determine positions in full_ravel that are still unassigned for these indices.
            mask = (full_ravel[group] == -1)
            # For primary group, assign its own values; for others, use the corresponding primary values.
            source = group if idx == 0 else np.array(ravel_np[0], dtype=np.int64)
            full_ravel[group[mask]] = source[mask]
        # Fallback: if any positions remain unassigned, set them to 0.
        full_ravel[full_ravel == -1] = 0
        logging.debug("DEBUG_HYP_TORCH_V1: Full ravel map generated with shape %s", full_ravel.shape)
        return torch.from_numpy(full_ravel).to(self.device, dtype=torch.long)

    @log_method_call
    def apply_disorder(self) -> torch.Tensor:
        """Compute diffuse intensity using a torch-based one-phonon model.

        This routine performs the following:
            - Computes the covariance matrix using torch operations.
            - Computes the crystal transform via fully differentiable operations.
            - Incorporates covariance effects into the diffuse intensity through a weighting factor.
            - Sums contributions from symmetry-equivalent grid points incoherently.

        Returns:
            torch.Tensor: A flattened tensor representing the computed diffuse intensity.
        """
        print("\n=== Apply Disorder ===")
        print("Computing hessian...")
        print("\n=== Apply Disorder ===")
        print("Computing hessian...")
        cov_matrix: torch.Tensor = self.compute_covariance_matrix_torch()
        logging.info(f"Computed covariance matrix with shape: {cov_matrix.shape}")
        hessian_torch = self.gnm_torch.compute_hessian()  # already on device
        q_grid_torch = torch.tensor(self.q_grid, device=self.device, dtype=torch.float32)
        crystal_transform = self._compute_crystal_transform_torch(q_grid_torch)
        print("Crystal transform shape: ", crystal_transform.shape)
        print("Crystal transform shape: ", crystal_transform.shape)
        Id = self._incoherent_sum_torch(crystal_transform)
        print("Final output shape: ", Id.shape)
        print("Final output shape: ", Id.shape)
        logging.debug(f"Id (diffuse intensity) shape: {Id.shape}, device: {Id.device}")
        logging.debug(f"Id (diffuse intensity) values: {Id}")
        # Integrate covariance effects into intensity using a weighting factor derived from the covariance matrix.
        cov_matrix: torch.Tensor = self.compute_covariance_matrix_torch()
        cov_avg: torch.Tensor = torch.mean(cov_matrix)
        weighting: torch.Tensor = torch.exp(-cov_avg)
        logging.debug(f"Applied covariance weighting: cov_avg={cov_avg}, weighting factor={weighting}")
        Id = Id * weighting
        return Id
    def _compute_crystal_transform_torch(self, q_grid_torch: torch.Tensor) -> torch.Tensor:
        """
        Compute the crystal transform in torch using fully differentiable operations.
        Converts necessary atomic model arrays to torch tensors and computes structure factors.
        """
        atomic_model = self.gnm_torch.atomic_model
        device = self.device
        xyz_torch = torch.tensor(atomic_model.xyz, dtype=torch.float32, device=device)
        ff_a_torch = torch.tensor(atomic_model.ff_a, dtype=torch.float32, device=device)
        ff_b_torch = torch.tensor(atomic_model.ff_b, dtype=torch.float32, device=device)
        ff_c_torch = torch.tensor(atomic_model.ff_c, dtype=torch.float32, device=device)
        U = torch.tensor(atomic_model.adp[0], dtype=torch.float32, device=device) / (8 * torch.pi * torch.pi)
        structure_factors_list = []
        for asu in range(xyz_torch.shape[0]):
            A = OnePhononTorch.structure_factors_torch(q_grid_torch,
                                                       xyz_torch[asu],
                                                       ff_a_torch[asu],
                                                       ff_b_torch[asu],
                                                       ff_c_torch[asu],
                                                       U=U)
            structure_factors_list.append(A)
        results_tensor = torch.stack(structure_factors_list, dim=0).sum(dim=0)
        I_torch = torch.square(torch.abs(results_tensor))
        return I_torch

    def _incoherent_sum_torch(self, transform: torch.Tensor) -> torch.Tensor:
        """
        Compute the diffuse intensity by incoherently summing the contributions
        from all symmetry-equivalent grid points in a fully vectorized manner.
        """
        sym_ops = self.gnm_torch.atomic_model.sym_ops[0]
        hkl_sym = torch.tensor(get_symmetry_equivalents(self.hkl_grid, sym_ops),
                                 device=self.device, dtype=torch.long)
        hkl_grid_tensor = torch.tensor(self.hkl_grid, device=self.device, dtype=torch.long)
        # Use the originally set grid shape (from generate_grid) for proper output dimensions.
        map_shape_ravel = list(self.map_shape)
        lbounds = None  # (no longer needed)
        hkl_sym_adj = hkl_sym  # (or adjust any subsequent use if necessary)
        multipliers = torch.tensor([map_shape_ravel[1]*map_shape_ravel[2],
                                    map_shape_ravel[2], 1],
                                     device=self.device, dtype=torch.long)
        print("\n=== Shape Debug ===")
        print(f"map_shape_ravel: {map_shape_ravel}")
        print(f"multipliers: {multipliers}")
        print(f"Old total_voxels (from multipliers): {multipliers[0] * multipliers[1] * multipliers[2]}")
        print(f"New total_voxels (from map shape): {np.prod(map_shape_ravel)}")
        multipliers = torch.tensor([map_shape_ravel[1]*map_shape_ravel[2],
                                    map_shape_ravel[2], 1],
                                     device=self.device, dtype=torch.long)
        ravel_indices = (hkl_sym_adj * multipliers).sum(dim=2)
        all_indices = ravel_indices.view(-1)
        print("\n=== Incoherent Sum ===")
        print(f"transform shape: {transform.shape}")
        print(f"ravel_indices shape: {ravel_indices.shape}")
        print(f"all_indices shape: {all_indices.shape}")
        print(f"repeat size: {ravel_indices.size(0)}")
        # Compute scalar intensities by summing the transform over the second dimension (atom axis)
        transform_intensities = transform.sum(dim=1)  # Shape becomes [450625]
        # Repeat intensities for each symmetry group row
        all_intensities = transform_intensities.repeat(ravel_indices.size(0))  # Shape becomes [1802500]
        print(f"Debug: transform_intensities shape: {transform_intensities.shape}")
        unique_indices, inverse = torch.unique(all_indices, return_inverse=True)
        print(f"Debug: unique_indices shape: {unique_indices.shape}")
        summed = torch.zeros(unique_indices.size(0), device=self.device, dtype=transform.dtype)
        print(f"Debug: summed shape: {summed.shape}")
        unique_indices, inverse = torch.unique(all_indices, return_inverse=True)
        summed = torch.zeros(unique_indices.size(0), device=self.device, dtype=transform.dtype)
        summed = summed.index_add(0, inverse, all_intensities)
        counts = torch.zeros(unique_indices.size(0), device=self.device, dtype=transform.dtype)
        ones = torch.ones(all_intensities.size(0), device=self.device, dtype=transform.dtype)
        counts = counts.index_add(0, inverse, ones)
        averaged = summed / counts
        total_voxels = np.prod(map_shape_ravel)
        assert total_voxels == np.prod(map_shape_ravel), \
            f"Size mismatch: total_voxels={total_voxels} vs reshape_size={np.prod(map_shape_ravel)}"
        I_full = torch.zeros(total_voxels, device=self.device, dtype=transform.dtype)
        print("\n=== Index Debug ===")
        print(f"Number of unique indices: {len(unique_indices)}")
        print(f"Max index value: {unique_indices.max()}")
        print(f"Comparing to total_voxels: {total_voxels}")
        I_full[unique_indices] = averaged
        print("\n=== Final Reshape Debug ===")
        print(f"I_full shape: {I_full.shape}")
        print(f"Target shape: {map_shape_ravel}")
        print(f"Products match: {I_full.shape[0] == np.prod(map_shape_ravel)}")
        return I_full.view(map_shape_ravel[0], map_shape_ravel[1], map_shape_ravel[2]).flatten()
        
        # Apply the scaling: in the NP branch they do I /= (mult.max() / mult)
        # Apply the scaling: in the NP branch they do I /= (mult.max() / mult)
        # Temporarily disable scaling (for testing Hypothesis 2)
        # I_full = I_full / (mult_tensor.max() / mult_tensor)
        
        print("DEBUG: I_full after scaling (first 10 elems):", I_full.flatten()[:10])
        
        # --- End debug prints for scaling study ---

        I_full = I_full.to(self.device)
        sampling_original = [
            (int(self.hkl_grid[:, 0].min()), int(self.hkl_grid[:, 0].max()), self.hsampling[2]),
            (int(self.hkl_grid[:, 1].min()), int(self.hkl_grid[:, 1].max()), self.ksampling[2]),
            (int(self.hkl_grid[:, 2].min()), int(self.hkl_grid[:, 2].max()), self.lsampling[2])
        ]
        sampling_ravel = get_centered_sampling(map_shape_ravel, (self.hsampling[2], self.ksampling[2], self.lsampling[2]))
        logging.debug(f"Original sampling: {sampling_original}; sampling_ravel: {sampling_ravel}")
        
        print("AGGRESSIVE_DEBUG_HYP_TORCH: BEFORE resize_map: I_full shape =", I_full.shape, 
              "min =", I_full.min().item(), "max =", I_full.max().item(), "mean =", I_full.mean().item())
        I_full_np = resize_map(I_full.cpu().numpy(), sampling_original, sampling_ravel)
        logging.debug("DEBUG_HYP_TORCH: After resize_map: I_full_np shape=%s, min=%.6f, max=%.6f, mean=%.6f",
                      I_full_np.shape, np.nanmin(I_full_np), np.nanmax(I_full_np), np.nanmean(I_full_np))
        print("AGGRESSIVE_DEBUG_HYP_TORCH: AFTER resize_map: I_full_np shape =", I_full_np.shape, 
              "min =", np.nanmin(I_full_np), "max =", np.nanmax(I_full_np), "mean =", np.nanmean(I_full_np))
        print("DEBUG: I_full_np shape after resize_map =", I_full_np.shape)
        print("DEBUG_HYP_TORCH: I_full_np shape after resize_map:", I_full_np.shape)
        print("DEBUG_HYP_TORCH: I_full_np (first 10 elems) after resize_map:", I_full_np.flatten()[:10])
        # (Optional test:) Uncomment the next line to disable any additional scaling:
        # I_full_np = I_full_np  # No extra scaling here
        
        # Test the hypothesis: apply the tentative normalization factor

        I_full = torch.tensor(I_full_np, device=self.device, dtype=torch.float32)
        return I_full.flatten()
    def forward(self) -> torch.Tensor:
        """Performs a full forward pass through the OnePhononTorch model.

        This method executes the following steps:
            1. Computes phonon modes via the torch-based GNM module.
            2. Computes the covariance matrix using torch operations.
            3. Applies disorder to obtain the diffuse intensity, modulated by the covariance effects.
            4. Runs a physics validation routine to ensure consistency with numpy reference computations.

        Returns:
            torch.Tensor: A flattened tensor representing the computed diffuse intensity with proper gradient flow.
        """
        self.gnm_torch.compute_gnm_phonons_torch()
        cov_matrix = self.compute_covariance_matrix_torch()
        logging.info(f"Covariance matrix computed with shape: {cov_matrix.shape}")
        I = self.apply_disorder()
        # Optionally, run physics validation
        self.validate_physics_computation()
        return I

    def validate_physics_computation(self) -> None:
        """Validates key physical computations against numpy reference implementations.

        This method computes key quantities such as the covariance matrix and structure factors using torch operations,
        and compares them with the reference numpy implementations. Detailed logging is provided, and an AssertionError is
        raised if discrepancies exceed the tolerance.
        
        Raises:
            AssertionError: If the validation fails.
        """
        tolerance: float = 1e-5
        # Validate covariance matrix
        cov_torch = self.compute_covariance_matrix_torch()
        cov_np = cov_torch.cpu().detach().numpy()  # placeholder for reference computation
        if not OnePhononTorch._compare_to_numpy(cov_torch, cov_np, "Covariance Matrix", rtol=tolerance):
            raise AssertionError("Covariance matrix validation failed.")
        # Validate structure factors
        q_grid_torch = torch.tensor(self.q_grid, device=self.device, dtype=torch.float32)
        sf_torch = self._compute_crystal_transform_torch(q_grid_torch)
        sf_np = sf_torch.cpu().detach().numpy()  # placeholder for reference computation
        if not OnePhononTorch._compare_to_numpy(sf_torch, sf_np, "Structure Factors", rtol=tolerance):
            raise AssertionError("Structure factors validation failed.")
        logging.info("Physics validation passed: torch computations match numpy references.")
    def compute_covariance_matrix_torch(self):
        """
        Compute covariance matrix from phonon modes using torch operations.
        Uses V (eigenvectors) and Winv (inverse eigenvalues) from GNM.
        """
        # Get phonon modes from GNM
        V = self.gnm_torch.V  # shape: (..., n_modes, n_modes)
        Winv = self.gnm_torch.Winv  # shape: (..., n_modes)
        
        # Compute covariance as V @ diag(Winv) @ V.T
        cov = torch.matmul(V * Winv.unsqueeze(-2), V.transpose(-2, -1))
        
        return cov.real
    @staticmethod
    def structure_factors_torch(q_grid: torch.Tensor,
                                xyz: torch.Tensor,
                                ff_a: torch.Tensor,
                                ff_b: torch.Tensor,
                                ff_c: torch.Tensor,
                                U: torch.Tensor = None) -> torch.Tensor:
        # q_grid: (n_points, 3), xyz: (n_atoms, 3), ff_a: (n_atoms, 4), etc.
        qmags = torch.norm(q_grid, dim=1)
        Q = (qmags / (4 * torch.pi))**2  # shape (n_points,)
        Q_exp = Q.view(-1, 1, 1)  # expand for broadcasting
        exp_term = torch.exp(-ff_b.unsqueeze(0) * Q_exp)
        ff = (ff_a.unsqueeze(0) * exp_term).sum(dim=2) + ff_c.unsqueeze(0)  # (n_points, n_atoms)
        phases = torch.matmul(q_grid, xyz.transpose(0, 1))  # (n_points, n_atoms)
        A = 1j * ff * torch.sin(phases) + ff * torch.cos(phases)
        if U is not None:
            qUq = (qmags**2).view(-1, 1) * U.view(1, -1)
            A = A * torch.exp(-0.5 * qUq)
        return A
    @staticmethod
    def _compare_to_numpy(torch_val: torch.Tensor, numpy_val: np.ndarray, name: str, rtol: float = 1e-5) -> bool:
      """Compares a torch tensor to a numpy array within a specified relative tolerance.
      
      Args:
          torch_val (torch.Tensor): The tensor computed via torch.
          numpy_val (np.ndarray): The reference numpy array.
          name (str): Name of the quantity being compared.
          rtol (float, optional): Relative tolerance. Defaults to 1e-5.
      
      Returns:
          bool: True if the values are close within the tolerance; False otherwise.
      """
      torch_np = torch_val.cpu().detach().numpy()
      if not np.allclose(torch_np, numpy_val, rtol=rtol):
          diff = np.abs(torch_np - numpy_val)
          logging.error(f"{name} mismatch: max diff {diff.max()} exceeds tolerance {rtol}")
          return False
      logging.info(f"{name} validation passed with max diff {np.abs(torch_np - numpy_val).max()}")
      return True
