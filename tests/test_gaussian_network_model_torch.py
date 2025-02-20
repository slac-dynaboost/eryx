import pytest
import logging
import torch
import numpy as np
from eryx.gaussian_network_torch import GaussianNetworkModelTorch

# Fixture for selecting the device
@pytest.fixture
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fixture for creating the Torch GNM model instance
@pytest.fixture
def gnm_model_torch(device) -> GaussianNetworkModelTorch:
    return GaussianNetworkModelTorch(
        pdb_path="tests/pdbs/5zck.pdb",
        enm_cutoff=4.0,
        gamma_intra=1.0,
        gamma_inter=1.0,
        device=device
    )


def test_neighbor_list_construction_torch(gnm_model_torch):
    """
    Verify that the neighbor lists from the Torch implementation match the reference.
    Ground truth is loaded from tests/test_data/reference/gnm_neighbor_lists.npy.
    """
    ref_neighbors = np.load("tests/test_data/reference/gnm_neighbor_lists.npy", allow_pickle=True)
    assert len(gnm_model_torch.asu_neighbors) == len(ref_neighbors)
    for idx, neighbors in enumerate(gnm_model_torch.asu_neighbors):
        # Convert to numpy if needed (in case any element is a tensor)
        neighbors_np = neighbors.cpu().numpy() if isinstance(neighbors, torch.Tensor) else neighbors
        ref = ref_neighbors[idx]
        assert len(neighbors_np) == len(ref)
        # Optionally, compare the connectivity details element‐by‐element.
        for cell_idx, cell_neighbors in enumerate(neighbors_np):
            assert len(cell_neighbors) == len(ref[cell_idx])


def test_spring_constants_torch(gnm_model_torch):
    """
    Validate that gamma values in the Torch model match the input parameters.
    For the reference cell (gnm_model_torch.id_cell_ref) and when i_asu==j_asu,
    gamma should equal gamma_intra; else, it should be gamma_inter.
    """
    gamma_intra = 1.0
    gamma_inter = 1.0
    for i_asu in range(gnm_model_torch.n_asu):
        for cell in range(gnm_model_torch.n_cell):
            for j_asu in range(gnm_model_torch.n_asu):
                gamma_val = gnm_model_torch.gamma[cell, i_asu, j_asu].item()
                if cell == gnm_model_torch.id_cell_ref and i_asu == j_asu:
                    assert gamma_val == pytest.approx(gamma_intra)
                else:
                    assert gamma_val == pytest.approx(gamma_inter)


def test_hessian_symmetry_torch(gnm_model_torch):
    """
    Compute the Hessian using the Torch implementation and verify that each
    diagonal block (reference cell for an ASU) is symmetric (Hermitian).
    """
    hessian = gnm_model_torch.compute_hessian()
    for i_asu in range(gnm_model_torch.n_asu):
        diag_block = hessian[i_asu, :, gnm_model_torch.id_cell_ref, i_asu, :]
        # Ensure the block equals its conjugate transpose within tolerance.
        assert torch.allclose(diag_block, diag_block.transpose(-2, -1).conj(), rtol=1e-7)


def test_k_matrix_computation_torch(gnm_model_torch, device):
    """
    Test the K-matrix computed by the Torch implementation.
    Compare its shape and values with the reference K matrix.
    """
    kvec = torch.tensor([1.0, 0.0, 0.0], device=device)
    hessian = gnm_model_torch.compute_hessian()
    Kmat = gnm_model_torch.compute_K(hessian, kvec=kvec)
    expected_shape = (
        gnm_model_torch.n_asu,
        gnm_model_torch.n_atoms_per_asu,
        gnm_model_torch.n_asu,
        gnm_model_torch.n_atoms_per_asu,
    )
    assert Kmat.shape == expected_shape
    ref_K = np.load("tests/test_data/reference/gnm_k_matrices.npy")
    assert np.allclose(Kmat.cpu().numpy(), ref_K, rtol=1e-7)


def test_inversion_stability_torch(gnm_model_torch, device):
    """
    Verify that the inversion of the K-matrix is stable.
    Compute the contracted identity and for each block, ensure it equals the identity.
    """
    kvec = torch.tensor([1.0, 0.0, 0.0], device=device)
    hessian = gnm_model_torch.compute_hessian()
    Kmat = gnm_model_torch.compute_K(hessian, kvec=kvec)
    Kinv = gnm_model_torch.compute_Kinv(hessian, kvec=kvec)
    identity_approx = torch.einsum('ijkl,klmn->ijmn', Kmat, Kinv)
    for i in range(gnm_model_torch.n_asu):
        for j in range(gnm_model_torch.n_atoms_per_asu):
            block = identity_approx[i, j, :, :]
            expected = torch.zeros((gnm_model_torch.n_asu, gnm_model_torch.n_atoms_per_asu),
                                   dtype=block.dtype, device=device)
            expected[i, j] = 1.0 + 0j
            diff = torch.abs(block - expected)
            maxdiff = diff.max().item()
            logging.info(f"[INFO test_inversion_stability] For i_asu={i}, atom index={j}, max diff={maxdiff:.8e}")
            assert torch.allclose(block, expected, rtol=1e-7)


def test_device_placement(gnm_model_torch, device):
    """
    Check that key model tensors remain on the appropriate device.
    """
    assert gnm_model_torch.gamma.device.type == device.type
    hessian = gnm_model_torch.compute_hessian()
    logging.debug(f"[DEBUG test_device_placement] hessian.device={hessian.device}, expected device={device}")
    assert hessian.device.type == device.type


def test_cuda_vs_cpu():
    """
    If CUDA is available, verify that results from CPU and CUDA versions match.
    """
    if torch.cuda.is_available():
        device_cuda = torch.device("cuda")
        model_cuda = GaussianNetworkModelTorch(
            pdb_path="tests/pdbs/5zck.pdb",
            enm_cutoff=4.0,
            gamma_intra=1.0,
            gamma_inter=1.0,
            device=device_cuda
        )
        device_cpu = torch.device("cpu")
        model_cpu = GaussianNetworkModelTorch(
            pdb_path="tests/pdbs/5zck.pdb",
            enm_cutoff=4.0,
            gamma_intra=1.0,
            gamma_inter=1.0,
            device=device_cpu
        )
        hessian_cuda = model_cuda.compute_hessian()
        hessian_cpu = model_cpu.compute_hessian()
        assert torch.allclose(hessian_cuda.cpu(), hessian_cpu.cpu(), rtol=1e-7)

def test_hessian_torch(gnm_model_torch):
    """
    Verify that the torch-computed Hessian matches the numpy version within tolerance.
    """
    hessian_torch = gnm_model_torch.compute_hessian_torch()
    hessian_np = gnm_model_torch.compute_hessian().cpu().numpy()
    np.testing.assert_allclose(hessian_torch.cpu().numpy(), hessian_np, rtol=1e-5,
                               err_msg="Hessian mismatch between Torch and NP methods")

def test_k_matrix_torch(gnm_model_torch):
    """
    Check that the K-matrix computed using Torch matches the numpy version.
    """
    hessian_torch = gnm_model_torch.compute_hessian_torch()
    kvec = torch.tensor([1.0, 0.0, 0.0], device=gnm_model_torch.device, dtype=torch.float64)
    Kmat_torch = gnm_model_torch.compute_K_torch(hessian_torch, kvec=kvec)
    Kmat_np = gnm_model_torch.compute_K(hessian_torch, kvec=kvec).cpu().numpy()
    # TODO convert back and forth from block structured to 2d
    np.testing.assert_allclose(Kmat_torch.cpu().numpy(), Kmat_np.reshape(Kmat_torch.cpu().numpy().shape), rtol=1e-5,
                               err_msg="K-matrix mismatch between Torch and NP methods")

def test_phonon_modes_torch(gnm_model_torch):
    """
    Validate the computed phonon modes (frequencies and eigenvectors) using Torch.
    """
    gnm_model_torch.compute_gnm_phonons_torch()
    V = gnm_model_torch.V.cpu().numpy()
    Winv = gnm_model_torch.Winv.cpu().numpy()
    assert not np.isnan(V).any(), "Eigenvectors contain NaN values"
    assert (Winv > 0).all(), "Inverse squared frequencies should be positive"

def test_gradient_flow(gnm_model_torch):
    """
    Verify that gradients propagate through the entire phonon computation.
    """
    gnm_model_torch.gamma.requires_grad_()
    gnm_model_torch.compute_gnm_phonons_torch()
    loss = sum([winv.sum() for winv in gnm_model_torch.Winv])
    loss.backward()
    assert gnm_model_torch.gamma.grad is not None, "Gradients did not propagate to the gamma parameter"
def test_gradient_flow_torch_parameters(gnm_model_torch):
    """
    Test that gradients propagate through learnable gamma parameters in the Torch GNM.
    """
    # Zero out any existing gradients.
    if gnm_model_torch.gamma_intra.grad is not None:
        gnm_model_torch.gamma_intra.grad.zero_()
    if gnm_model_torch.gamma_inter.grad is not None:
        gnm_model_torch.gamma_inter.grad.zero_()
    # Perform a forward pass.
    V, Winv = gnm_model_torch.forward()
    # Define a dummy loss that sums outputs.
    loss = torch.sum(V) + torch.sum(Winv)
    loss.backward()
    assert gnm_model_torch.gamma_intra.grad is not None, "gamma_intra did not receive gradients"
    assert gnm_model_torch.gamma_inter.grad is not None, "gamma_inter did not receive gradients"
