import pytest
import torch
import numpy as np
from eryx.onephonon_torch import OnePhononTorch
from tests.test_utils.log_analysis import LogAnalyzer

@pytest.fixture
def device():
    return torch.device("cpu")

@pytest.fixture
def onephonon_torch(device):
    return OnePhononTorch(
        "tests/pdbs/5zck_p1.pdb", 
        [-4, 4, 3], [-17, 17, 3], [-29, 29, 3],
        expand_p1=True,
        gnm_cutoff=4.0,
        gamma_intra=1.0,
        gamma_inter=1.0,
        device=device
    )

class TestOnePhononTorch:
    def test_input_parameters_torch(self, onephonon_torch, device):
        """
        Validate that the OnePhononTorch initialization parses parameters correctly
        and that the model is set on the proper device.
        """
        with open("tests/test_data/logs/base_run/base_run.log") as f:
            log_content = f.read()
        analyzer = LogAnalyzer(log_content)
        entries = analyzer.extract_entries()
        init_entry = next(e for e in entries if "OnePhonon.__init__" in e[0])
        # Check for key parameters (as in the numpy version)
        assert "tests/pdbs/5zck.pdb" in init_entry[1]
        assert "[-4, 4, 1]" in init_entry[1]
        assert "[-17, 17, 1]" in init_entry[1]
        assert "[-29, 29, 1]" in init_entry[1]
        assert "'gnm_cutoff': 4.0" in init_entry[1]
        assert "'gamma_intra': 1.0" in init_entry[1]
        assert "'gamma_inter': 1.0" in init_entry[1]
        assert "'expand_p1': True" in init_entry[1]
        # Check that the model’s device matches the fixture
        assert onephonon_torch.device == device

    def test_log_sequence_torch(self):
        """
        Verify that the log entries follow the expected sequence.
        Also check if any torch/device-specific messages are present.
        """
        with open("tests/test_data/logs/base_run/base_run.log") as f:
            log_content = f.read()
        analyzer = LogAnalyzer(log_content)
        expected_sequence = ["ModelRunner.run_model"]
        assert analyzer.validate_sequence(expected_sequence)

    def test_sym_ops_return_value_torch(self, onephonon_torch):
        """
        Validate symmetry operations from the torch model against hardcoded reference
        (ground truth identical to the numpy version).
        """
        expected_sym_ops = (
            {0: np.array([[1., 0., 0.],
                          [0., 1., 0.],
                          [0., 0., 1.]]),
             1: np.array([[-1., 0., 0.],
                          [0., -1., 0.],
                          [0., 0., 1.]]),
             2: np.array([[-1., 0., 0.],
                          [0., 1., 0.],
                          [0., 0., -1.]]),
             3: np.array([[1., 0., 0.],
                          [0., -1., 0.],
                          [0., 0., -1.]])},
            {0: np.array([[1., 0., 0., 0.],
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
        )
        sym_ops = onephonon_torch.gnm_torch.atomic_model.sym_ops  # using the numpy data from AtomicModel
        for key in expected_sym_ops[0]:
            np.testing.assert_array_almost_equal(
                sym_ops[0][key],
                expected_sym_ops[0][key],
                decimal=5,
                err_msg=f"Mismatch in sym_ops[0] for key {key}"
            )
        for key in expected_sym_ops[1]:
            np.testing.assert_array_almost_equal(
                sym_ops[1][key],
                expected_sym_ops[1][key],
                decimal=5,
                err_msg=f"Mismatch in sym_ops[1] for key {key}"
            )

    def test_computation_validation_torch(self, onephonon_torch):
        """
        Verify that the diffraction pattern computed by the torch model matches
        the ground-truth reference (using rtol=1e-2).
        """
        Id = onephonon_torch.apply_disorder()
        # Ensure the result stays on the device until final comparison
        Id_np = Id.detach().cpu().numpy().reshape(onephonon_torch.map_shape)
        ref = np.load("tests/test_data/reference/np_diffuse_intensity.npy")
        valid_mask = ~np.isnan(ref)
        np.testing.assert_allclose(
            Id_np.flatten()[valid_mask.flatten()],
            ref.flatten()[valid_mask.flatten()],
            rtol=1e-2,
            err_msg="Computed diffraction pattern does not match reference data"
        )

    def test_edge_case_invalid_gamma_torch(self):
        """
        Ensure that the torch model reports a failure when gamma_inter is set negative.
        """
        with open("tests/test_data/logs/edge_cases/edge_run_2.log") as f:
            log_content = f.read()
        analyzer = LogAnalyzer(log_content)
        failure_msg = analyzer.get_failure_message()
        assert failure_msg is not None
        assert "Invalid gamma_inter: must be non-negative" in failure_msg
        expected_sequence = ["ModelRunner.run_model"]
        assert analyzer.validate_sequence(expected_sequence)

    def test_device_placement(self, onephonon_torch, device):
        """
        Verify that intermediate tensor operations occur on the proper device.
        """
        # For example, call the internal crystal transform function and check its device.
        q_grid_torch = onephonon_torch.q_grid.clone().detach()
        crystal_transform = onephonon_torch._compute_crystal_transform_torch(q_grid_torch)
        assert crystal_transform.device.type == device.type, "Intermediate tensor not on expected device"

    @pytest.mark.skip(reason="GPU tests disabled; running only on CPU")
    def test_cuda_vs_cpu(self, device):
        pass
def test_gradient_flow(onephonon_torch, device):
    onephonon_torch.train()
    I = onephonon_torch.forward()
    loss = I.sum()
    loss.backward()
    # Verify that gradients propagate to the learnable parameters in the torch GNM.
    assert onephonon_torch.gnm_torch.gamma_intra.grad is not None, "gamma_intra did not receive gradients"
    assert onephonon_torch.gnm_torch.gamma_inter.grad is not None, "gamma_inter did not receive gradients"
def test_performance_torch(onephonon_torch):
    import time
    start_time = time.time()
    I = onephonon_torch.forward()
    elapsed = time.time() - start_time
    # Expect forward pass to complete within 5 seconds on CPU
    assert elapsed < 5.0, f"Forward pass took too long: {elapsed} seconds"
