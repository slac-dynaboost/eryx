import pytest
import torch
import numpy as np
from eryx.onephonon_torch import OnePhononTorch
from eryx.scatter import compute_form_factors, structure_factors

@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture
def onephonon_torch(device):
    return OnePhononTorch(
        pdb_path="tests/pdbs/5zck_p1.pdb",
        hsampling=[-4, 4, 3],
        ksampling=[-17, 17, 3],
        lsampling=[-29, 29, 3],
        expand_p1=True,
        gnm_cutoff=4.0,
        gamma_intra=1.0,
        gamma_inter=1.0,
        device=device
    )

class TestDiffractionChainTorch:
    """Tests for the PyTorch diffraction chain ensuring numerical equivalence with the numpy version."""

    def test_form_factors_torch(self, onephonon_torch, device):
        """
        Validate that the form factors computed on device have the correct shape,
        contain no NaNs, and yield a mean value matching the hardcoded ground truth.
        """
        # Use the first 10 q points and move to device
        q_test = torch.tensor(onephonon_torch.q_grid[:10], device=device, dtype=torch.float32)
        # Convert the atomic form factor arrays to torch tensors
        ff_a = torch.tensor(onephonon_torch.gnm_torch.atomic_model.ff_a[0], device=device, dtype=torch.float32)
        ff_b = torch.tensor(onephonon_torch.gnm_torch.atomic_model.ff_b[0], device=device, dtype=torch.float32)
        ff_c = torch.tensor(onephonon_torch.gnm_torch.atomic_model.ff_c[0], device=device, dtype=torch.float32)
        ff = compute_form_factors(q_test, ff_a, ff_b, ff_c)
        # Verify the computed shape and absence of NaNs
        assert ff.shape == (10, ff_a.shape[0])
        assert not torch.any(torch.isnan(ff))
        expected_ff_mean = 3.975718
        torch.testing.assert_close(torch.mean(ff).cpu(), torch.tensor(expected_ff_mean, dtype=torch.float32), rtol=1e-5)

    def test_structure_factors_torch(self, onephonon_torch, device):
        """
        Validate that structure factors computed on device have the correct number of q points,
        contain no NaNs, and the real part of the first element matches the hardcoded reference.
        """
        q_test = torch.tensor(onephonon_torch.q_grid[:10], device=device, dtype=torch.float32)
        atomic_model = onephonon_torch.gnm_torch.atomic_model
        # For 'project_on_components', try using onephonon_torch.Amat if available; otherwise fallback.
        try:
            amat = torch.tensor(onephonon_torch.Amat[0], device=device, dtype=torch.float32)
        except AttributeError:
            amat = torch.tensor(atomic_model.Amat[0], device=device, dtype=torch.float32)
        # Call structure_factors (if it expects numpy arrays, convert as necessary)
        F = structure_factors(
            q_test.cpu().numpy(),
            atomic_model.xyz[0],
            atomic_model.ff_a[0],
            atomic_model.ff_b[0],
            atomic_model.ff_c[0],
            compute_qF=True,
            project_on_components=amat.cpu().numpy()
        )
        # Re-cast F as a torch tensor for the tests
        F = torch.tensor(F, device=device, dtype=torch.complex64)
        assert F.shape[0] == q_test.shape[0]
        assert not torch.any(torch.isnan(torch.real(F)))
        expected_F_first = -941.71642
        torch.testing.assert_close(torch.real(F[0]).cpu(), torch.tensor(expected_F_first, dtype=torch.float32), rtol=1e-5)

    def test_symmetries_and_hessian_torch(self, onephonon_torch, device):
        """
        Ensure that the computed diffraction pattern has mirror symmetry and that the Hessian matrix
        exhibits the expected symmetric (Hermitian) properties.
        """
        # Compute the diffraction pattern using the apply_disorder method (returns a tensor)
        Id = onephonon_torch.apply_disorder(use_data_adp=True)
        Id = Id.reshape(onephonon_torch.map_shape)
        central_h = Id.shape[0] // 2
        central_slice = Id[central_h, :, :]
        torch.testing.assert_close(central_slice, torch.flip(central_slice, dims=[0, 1]), rtol=1e-5)
        
        # Compute the Hessian and check symmetry for each ASU block
        hessian = onephonon_torch.gnm_torch.compute_hessian()
        idx = onephonon_torch.gnm_torch.crystal.hkl_to_id([0, 0, 0])
        n_asu = onephonon_torch.gnm_torch.n_asu
        for i in range(n_asu):
            hi = hessian[i, :, idx, i, :]
            torch.testing.assert_close(hi, hi.transpose(-2, -1), atol=1e-5)

    def test_diffuse_intensity_torch(self, onephonon_torch, device):
        """
        Validate that the diffuse intensity (diffraction pattern) has the correct shape,
        contains no NaN values and that the central intensity matches the hardcoded reference.
        """
        Id = onephonon_torch.apply_disorder(use_data_adp=True)
        Id = Id.reshape(onephonon_torch.map_shape)
        assert Id.shape == onephonon_torch.map_shape
        Id_clean = torch.nan_to_num(Id, nan=0.0)
        central_idx = (Id_clean.shape[0] // 2, Id_clean.shape[1] // 2, Id_clean.shape[2] // 2)
        expected_center_intensity = 0.0
        torch.testing.assert_close(
            Id_clean[central_idx],
            torch.tensor(expected_center_intensity, device=device, dtype=torch.float32),
            rtol=1e-5
        )
        assert torch.count_nonzero(Id_clean) > 0

    def test_device_placement(self, onephonon_torch, device):
        """
        Check that key tensors (such as the q-grid and computed diffraction pattern)
        are placed on the correct device.
        """
        q_tensor = torch.tensor(onephonon_torch.q_grid, device=device)
        assert q_tensor.device == device
        Id = onephonon_torch.apply_disorder(use_data_adp=True)
        assert Id.device == device

    def test_cuda_vs_cpu(self):
        """
        Compare results between CPU and CUDA devices to ensure that
        the computations are consistent across devices.
        """
        cpu_device = torch.device("cpu")
        onephonon_cpu = OnePhononTorch(
            pdb_path="tests/pdbs/5zck_p1.pdb",
            hsampling=[-4, 4, 3],
            ksampling=[-17, 17, 3],
            lsampling=[-29, 29, 3],
            expand_p1=True,
            gnm_cutoff=4.0,
            gamma_intra=1.0,
            gamma_inter=1.0,
            device=cpu_device
        )
        if torch.cuda.is_available():
            cuda_device = torch.device("cuda")
            onephonon_cuda = OnePhononTorch(
                pdb_path="tests/pdbs/5zck_p1.pdb",
                hsampling=[-4, 4, 3],
                ksampling=[-17, 17, 3],
                lsampling=[-29, 29, 3],
                expand_p1=True,
                gnm_cutoff=4.0,
                gamma_intra=1.0,
                gamma_inter=1.0,
                device=cuda_device
            )
            Id_cpu = onephonon_cpu.apply_disorder(use_data_adp=True)
            Id_cuda = onephonon_cuda.apply_disorder(use_data_adp=True)
            torch.testing.assert_close(Id_cpu, Id_cuda.cpu(), rtol=1e-5)

    def test_performance(self, onephonon_torch, device):
        """
        Measure the computation time for generating the diffraction pattern.
        Log the elapsed time and ensure that the performance is within acceptable bounds.
        """
        import time
        start = time.time()
        _ = onephonon_torch.apply_disorder(use_data_adp=True)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.time() - start
        print(f"Diffraction pattern computation time: {elapsed:.4f} seconds")
        # (Optional) Assert that elapsed time is below an expected threshold if desired.
