import torch
import numpy as np
import logging
from eryx.models import ModelRunner
from eryx.gaussian_network_torch import GaussianNetworkModelTorch
from eryx.pdb import AtomicModel
from eryx.logging_utils import log_method_call
from eryx.map_utils import get_resolution_mask, get_dq_map, expand_sym_ops, get_symmetry_equivalents, get_ravel_indices, compute_multiplicity

class OnePhononTorch(ModelRunner):
    @log_method_call
    def __init__(self,
                 pdb_path: str,
                 hsampling: list,
                 ksampling: list,
                 lsampling: list,
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
        Initialize the torch OnePhonon model.
        Uses the torch GNM for phonon computations while keeping data loading in numpy.
        """
        self.pdb_path = pdb_path
        self.hsampling = hsampling
        self.ksampling = ksampling
        self.lsampling = lsampling
        self.expand_p1 = expand_p1
        self.res_limit = res_limit
        self.batch_size = batch_size
        self.n_processes = n_processes
        self.device = device

        # Use numpy routines to set up the grid and q_grid
        atomic_model = AtomicModel(pdb_path, expand_p1)
        from eryx.map_utils import generate_grid  # use existing method
        self.hkl_grid, self.map_shape = generate_grid(atomic_model.A_inv,
                                                      self.hsampling,
                                                      self.ksampling,
                                                      self.lsampling,
                                                      return_hkl=True)
        self.q_grid = 2 * np.pi * np.inner(atomic_model.A_inv.T, self.hkl_grid).T
        logging.debug(f"q_grid shape (numpy): {self.q_grid.shape}")

        # Initialize the torch-based GNM
        self.gnm_torch = GaussianNetworkModelTorch(pdb_path, gnm_cutoff, gamma_intra, gamma_inter, device=device)
        # (Additional initialization may follow as required by the OnePhonon model.)

    @log_method_call
    def apply_disorder(self) -> torch.Tensor:
        """
        Apply disorder computation using torch operations.
        """
        hessian_torch = self.gnm_torch.compute_hessian()  # already on device
        q_grid_torch = torch.tensor(self.q_grid, device=self.device, dtype=torch.float32)
        crystal_transform = self._compute_crystal_transform_torch(q_grid_torch)
        Id = self._incoherent_sum_torch(crystal_transform)
        logging.debug(f"Id (diffuse intensity) shape: {Id.shape}, device: {Id.device}")
        return Id
    def _compute_crystal_transform_torch(self, q_grid_torch: torch.Tensor) -> torch.Tensor:
        """
        Compute the crystal transform in torch in an equivalent way to the NP version.
        """
        mask_np, res_map_np = get_resolution_mask(self.gnm_torch.atomic_model.cell, self.q_grid, self.res_limit)
        dq_map_np = np.around(get_dq_map(self.gnm_torch.atomic_model.A_inv, self.q_grid), 5)
        mask = torch.tensor(mask_np, device=self.device, dtype=torch.bool)
        dq_mask = torch.tensor(np.equal(dq_map_np, 0), device=self.device)
        xyz = torch.tensor(self.gnm_torch.atomic_model.xyz, device=self.device, dtype=torch.float32)
        phi = torch.matmul(q_grid_torch, xyz.T)
        A_real = torch.cos(phi)
        A_imag = torch.sin(phi)
        A = A_real + 1j * A_imag
        I = torch.zeros(q_grid_torch.shape[0], device=self.device, dtype=torch.float32)
        valid = mask & dq_mask
        I[valid] = torch.square(torch.abs(A[valid])).sum(dim=1)
        return I

    def _incoherent_sum_torch(self, transform: torch.Tensor) -> torch.Tensor:
        """
        Compute the diffuse intensity by incoherently summing the contributions
        from each asymmetric unit in a manner equivalent to NP’s incoherent_sum_real().
        """
        sym_ops = expand_sym_ops(self.gnm_torch.atomic_model.sym_ops)
        hkl_sym = get_symmetry_equivalents(self.hkl_grid, sym_ops)
        ravel_np, map_shape_ravel = get_ravel_indices(hkl_sym, (self.hsampling[2], self.ksampling[2], self.lsampling[2]))
        I_np = transform.detach().cpu().numpy()
        I_full = np.zeros(map_shape_ravel).flatten()
        for i in range(ravel_np.shape[0]):
            I_full[ravel_np[i]] += I_np.copy()
        _, mult = compute_multiplicity(self.gnm_torch.atomic_model, 
                                       (-self.hsampling[1], self.hsampling[1], self.hsampling[2]),
                                       (-self.ksampling[1], self.ksampling[1], self.ksampling[2]),
                                       (-self.lsampling[1], self.lsampling[1], self.lsampling[2]))
        I_full = I_full / (mult.max() / mult)
        I_torch = torch.tensor(I_full, device=self.device, dtype=torch.float32)
        return I_torch.flatten()
