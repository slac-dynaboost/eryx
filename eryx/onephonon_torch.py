import torch
import numpy as np
import logging
from eryx.models import ModelRunner
from eryx.gaussian_network_torch import GaussianNetworkModelTorch
from eryx.pdb import AtomicModel
from eryx.logging_utils import log_method_call

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
        (This is a simplified placeholder that shows conversion and torch arithmetic.)
        """
        # Compute the Hessian using our torch-based GNM
        hessian_torch = self.gnm_torch.compute_hessian()  # tensor on device
        # For this example, we assume kvec = 0 (you can extend to sample over q-vectors)
        kvec = torch.zeros(3, device=self.device, dtype=torch.float32)
        # Compute the dynamical matrix and its inverse using torch
        Kmat_torch = self.gnm_torch.compute_K(hessian_torch, kvec)
        Kinv_torch = self.gnm_torch.compute_Kinv(hessian_torch, kvec)

        # Dummy disorder computation – replace with the real formula
        Id = torch.zeros((self.q_grid.shape[0]), device=self.device, dtype=torch.float32)
        for idx in range(self.q_grid.shape[0]):
            # Here you would compute structure factors etc.; below is a placeholder.
            Id[idx] = torch.norm(Kmat_torch.view(-1)).item()  # dummy operation
        logging.debug(f"Id (diffuse intensity) shape: {Id.shape}, device: {Id.device}")
        return Id
