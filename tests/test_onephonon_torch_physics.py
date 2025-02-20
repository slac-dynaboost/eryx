import pytest
import numpy as np
import torch
from eryx.onephonon_torch import OnePhononTorch
from eryx.models import OnePhonon  # reference (numpy) implementation

@pytest.fixture
def device():
    return torch.device("cpu")

def test_phonon_modes_equivalence(device):
    pdb_path = "tests/pdbs/5zck.pdb"  # example pdb file
    hsampling = [-1, 2, 10]
    ksampling = [-1, 2, 10]
    lsampling = [-1, 2, 10]
    
    model_torch = OnePhononTorch(pdb_path, hsampling, ksampling, lsampling, device=device)
    I_torch = model_torch.forward().cpu().numpy()
    
    model_np = OnePhonon(pdb_path, hsampling, ksampling, lsampling)
    I_np = model_np.apply_disorder(rank=-1)
    
    assert np.allclose(I_torch, I_np, rtol=1e-5), "Diffuse intensities mismatch between torch and numpy implementations."

def test_structure_factors_match(device):
    pdb_path = "tests/pdbs/5zck.pdb"
    hsampling = [-1, 2, 10]
    ksampling = [-1, 2, 10]
    lsampling = [-1, 2, 10]
    
    model_torch = OnePhononTorch(pdb_path, hsampling, ksampling, lsampling, device=device)
    q_grid = model_torch.q_grid.to(device)
    torch_sf = model_torch._compute_crystal_transform_torch(q_grid).cpu().detach().numpy()
    
    from eryx.scatter import structure_factors
    from eryx.pdb import AtomicModel
    atomic_model = AtomicModel(pdb_path, expand_p1=True)
    np_sf = structure_factors(2 * np.pi * np.inner(atomic_model.A_inv.T, model_torch.hkl_grid).T,
                              atomic_model.xyz[0],
                              atomic_model.ff_a[0],
                              atomic_model.ff_b[0],
                              atomic_model.ff_c[0],
                              U=atomic_model.adp[0]/(8*np.pi*np.pi))
    assert np.allclose(torch.abs(torch_sf), np.abs(np_sf), rtol=1e-5), "Structure factors differ between torch and numpy."
