import torch
import numpy as np
import logging
from eryx.pdb import AtomicModel, Crystal
from scipy.spatial import KDTree

class GaussianNetworkModelTorch:
    def __init__(self, pdb_path: str, enm_cutoff: float, gamma_intra: float, gamma_inter: float,
                 device: torch.device = torch.device("cpu")) -> None:
        """
        Initialize the torch-based Gaussian Network Model.
        Loads the atomic model (using existing numpy code) and converts key arrays to torch tensors.
        """
        self.device = device
        # Load and set up the atomic model (using existing numpy routines)
        self.atomic_model = AtomicModel(pdb_path, expand_p1=True)
        self.crystal = Crystal(self.atomic_model)
        self.crystal.supercell_extent(nx=1, ny=1, nz=1)
        self.id_cell_ref = self.crystal.hkl_to_id([0, 0, 0])
        self.n_cell = self.crystal.n_cell
        self.n_asu = self.crystal.model.n_asu
        self.n_atoms_per_asu = self.crystal.get_asu_xyz().shape[0]
        self.n_dof_per_asu_actual = self.n_atoms_per_asu * 3
        self.enm_cutoff = enm_cutoff
        self.gamma_intra = gamma_intra
        self.gamma_inter = gamma_inter

        self.build_gamma()
        self.build_neighbor_list()

    def build_gamma(self) -> None:
        """
        Build the spring constant tensor.
        """
        # Create a torch tensor filled with gamma_inter (shape: [n_cell, n_asu, n_asu])
        self.gamma = torch.full((self.n_cell, self.n_asu, self.n_asu),
                                self.gamma_inter, device=self.device, dtype=torch.float32)
        # In the reference cell (id_cell_ref) set intra interaction gamma
        for i_asu in range(self.n_asu):
            self.gamma[self.id_cell_ref, i_asu, i_asu] = self.gamma_intra

    def build_neighbor_list(self) -> None:
        """
        Build the neighbor list using numpy’s KDTree. (The neighbor lists remain numpy lists.)
        """
        self.asu_neighbors = []
        for i_asu in range(self.n_asu):
            self.asu_neighbors.append([])
            xyz_ref = self.crystal.get_asu_xyz(i_asu, self.crystal.id_to_hkl(self.id_cell_ref))
            kd_tree1 = KDTree(xyz_ref)
            for i_cell in range(self.n_cell):
                self.asu_neighbors[i_asu].append([])
                for j_asu in range(self.n_asu):
                    xyz_neighbor = self.crystal.get_asu_xyz(j_asu, self.crystal.id_to_hkl(i_cell))
                    kd_tree2 = KDTree(xyz_neighbor)
                    neighbors = kd_tree1.query_ball_tree(kd_tree2, r=self.enm_cutoff)
                    self.asu_neighbors[i_asu][-1].append(neighbors)

    def compute_hessian(self) -> torch.Tensor:
        """
        Compute the Hessian matrix using torch operations.
        Returns a tensor with shape:
           (n_asu, n_atoms_per_asu, n_cell, n_asu, n_atoms_per_asu) of dtype complex.
        """
        shape = (self.n_asu, self.n_atoms_per_asu, self.n_cell, self.n_asu, self.n_atoms_per_asu)
        hessian = torch.zeros(shape, dtype=torch.complex64, device=self.device)
        hessian_diag = torch.zeros((self.n_asu, self.n_atoms_per_asu),
                                   dtype=torch.complex64, device=self.device)
        # Loop over ASU and neighbor cells using the neighbor list
        for i_asu in range(self.n_asu):
            for i_cell in range(self.n_cell):
                for j_asu in range(self.n_asu):
                    neighbors_list = self.asu_neighbors[i_asu][i_cell][j_asu]
                    for i_at, neigh_indices in enumerate(neighbors_list):
                        if neigh_indices:
                            gamma_val = self.gamma[i_cell, i_asu, j_asu]
                            # Set off-diagonal entries (neighbors may be a list of indices)
                            idxs = torch.tensor(neigh_indices, device=self.device)
                            hessian[i_asu, i_at, i_cell, j_asu, idxs] = -gamma_val.to(torch.complex64)
                            hessian_diag[i_asu, i_at] -= gamma_val.to(torch.complex64) * float(len(neigh_indices))
        # Set the diagonal (reference cell)
        for i_asu in range(self.n_asu):
            for i_at in range(self.n_atoms_per_asu):
                hessian[i_asu, i_at, self.id_cell_ref, i_asu, i_at] = -hessian_diag[i_asu, i_at] - self.gamma[self.id_cell_ref, i_asu, i_asu].to(torch.complex64)
        logging.debug(f"Hessian shape: {hessian.shape}")
        return hessian

    def compute_K(self, hessian: torch.Tensor, kvec: torch.Tensor = None) -> torch.Tensor:
        """
        Compute the dynamical matrix K(k) using torch.
        """
        if kvec is None:
            kvec = torch.zeros(3, device=self.device, dtype=torch.float32)
        # Start with the reference cell term
        Kmat = hessian[:, :, self.id_cell_ref, :, :].clone()
        # Sum contributions from other cells
        for j_cell in range(self.n_cell):
            if j_cell == self.id_cell_ref:
                continue
            # Get the cell origin (from the numpy Crystal object) and convert to torch tensor
            r_cell_np = self.crystal.get_unitcell_origin(self.crystal.id_to_hkl(j_cell))
            r_cell = torch.tensor(r_cell_np, device=self.device, dtype=torch.float32)
            phase = torch.dot(kvec, r_cell)
            eikr = torch.cos(phase) + 1j * torch.sin(phase)
            Kmat += hessian[:, :, j_cell, :, :] * eikr
        logging.debug(f"Kmat shape: {Kmat.shape}")
        return Kmat

    def compute_Kinv(self, hessian: torch.Tensor, kvec: torch.Tensor = None, reshape: bool = True) -> torch.Tensor:
        """
        Compute the inverse of K(k) using torch.linalg.pinv.
        """
        Kmat = self.compute_K(hessian, kvec)
        shape = Kmat.shape  # (n_asu, n_atoms_per_asu, n_asu, n_atoms_per_asu)
        Kmat_flat = Kmat.reshape(shape[0] * shape[1], shape[2] * shape[3])
        Kinv_flat = torch.linalg.pinv(Kmat_flat)
        if reshape:
            Kinv = Kinv_flat.reshape((shape[0], shape[1], shape[2], shape[3]))
        else:
            Kinv = Kinv_flat
        logging.debug(f"Kinv shape: {Kinv.shape}")
        return Kinv
