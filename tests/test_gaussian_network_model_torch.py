from eryx.gaussian_network_torch import GaussianNetworkModelTorch

def test_build_gamma_and_neighbors():
    # Set parameters (using test pdb)
    model = GaussianNetworkModelTorch(
        pdb_path="tests/pdbs/5zck.pdb",
        enm_cutoff=4.0,
        gamma_intra=1.0,
        gamma_inter=1.0
    )
    # Ensure gamma is built correctly
    assert hasattr(model, "gamma")
    # Ensure neighbor list is built and non-empty for each asu:
    for neighbors in model.asu_neighbors:
        # neighbors is a list with one entry per cell
        assert any(len(cell_neighbors) > 0 for cell_neighbors in neighbors)
import torch
