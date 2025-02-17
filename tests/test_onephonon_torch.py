import os
import numpy as np
import pytest
import torch
from eryx.onephonon_torch import OnePhononTorch

# --- Test 1: Model Initialization and Parameter Handling ---
def test_initialization_torch():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    onephonon = OnePhononTorch(
        pdb_path="tests/pdbs/5zck.pdb",
        hsampling=[-4, 4, 1],
        ksampling=[-17, 17, 1],
        lsampling=[-29, 29, 1],
        expand_p1=True,
        gnm_cutoff=4.0,
        gamma_intra=1.0,
        gamma_inter=1.0,
        device=device  # ensure the model accepts a device argument
    )
    # Check that key attributes are torch tensors and on the correct device.
    # For example, q_grid should be a numpy array from preprocessing,
    # but later computed tensors (if any) should reside on device.
    if hasattr(onephonon, "q_grid") and isinstance(onephonon.q_grid, torch.Tensor):
        assert onephonon.q_grid.device.type == device

# --- Test 2: Disorder Application ---
def test_disorder_application_torch():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    onephonon = OnePhononTorch(
        pdb_path="tests/pdbs/5zck_p1.pdb",  # using p1 version for torch tests
        hsampling=[-4, 4, 1],
        ksampling=[-17, 17, 1],
        lsampling=[-29, 29, 1],
        expand_p1=True,
        gnm_cutoff=4.0,
        gamma_intra=1.0,
        gamma_inter=1.0,
        device=device
    )
    result = onephonon.apply_disorder()
    # Verify output is a torch.Tensor
    assert isinstance(result, torch.Tensor)
    # Verify shape consistency: compare with the pre-set map shape attribute.
    expected_shape = (np.prod(onephonon.map_shape),)
    assert result.flatten().shape[0] == expected_shape[0]
    # Optionally, compare numerical values versus numpy reference.
    computed_np = result.detach().cpu().numpy().flatten()
    ref = np.load("tests/test_data/reference/diffraction_pattern.npy")
    valid = ~np.isnan(ref)
    np.testing.assert_allclose(computed_np[valid], ref[valid], rtol=1e-2)

# --- Test 3: End-to-End Integration and Device Compatibility ---
def test_integration_torch():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    onephonon = OnePhononTorch(
        pdb_path="tests/pdbs/5zck_p1.pdb",
        hsampling=[-4, 4, 1],
        ksampling=[-17, 17, 1],
        lsampling=[-29, 29, 1],
        expand_p1=True,
        gnm_cutoff=4.0,
        gamma_intra=1.0,
        gamma_inter=1.0,
        device=device
    )
    result = onephonon.apply_disorder()
    # Check that result is on the correct device
    assert result.device.type == device
    # Convert to CPU numpy array for comparison
    computed_np = result.detach().cpu().numpy().flatten()
    # Check basic statistics: non-nan values, reasonable range, etc.
    assert np.all(np.isfinite(computed_np))
    # If reference data exists, compare as in previous test.
    ref = np.load("tests/test_data/reference/diffraction_pattern.npy")
    valid = ~np.isnan(ref)
    np.testing.assert_allclose(computed_np[valid], ref[valid], rtol=1e-2)
import torch
