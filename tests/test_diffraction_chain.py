import pytest
import numpy as np
from eryx.models import OnePhonon
from eryx.scatter import compute_form_factors, structure_factors

def test_diffraction_calculation_chain():
    """Test the full chain of diffraction intensity calculation"""
    # 1. Setup minimal test case
    pdb_path = "tests/pdbs/5zck_p1.pdb"
    sampling = [-2, 2, 1]  # Small grid for quick testing
    
    onephonon = OnePhonon(
        pdb_path,
        sampling, sampling, sampling,
        expand_p1=True,
        gnm_cutoff=4.0,
        gamma_intra=1.0,
        gamma_inter=1.0
    )
    
    # 2. Test form factors
    q_test = onephonon.q_grid[:10]  # Test first few q points
    ff = compute_form_factors(
        q_test,
        onephonon.model.ff_a[0],
        onephonon.model.ff_b[0],
        onephonon.model.ff_c[0]
    )
    assert ff.shape == (10, onephonon.model.ff_a[0].shape[0])
    assert not np.any(np.isnan(ff))
    
    # 3. Test structure factors
    F = structure_factors(
        q_test,
        onephonon.model.xyz[0],
        onephonon.model.ff_a[0],
        onephonon.model.ff_b[0],
        onephonon.model.ff_c[0],
        compute_qF=True,
        project_on_components=onephonon.Amat[0]
    )
    assert F.shape[0] == q_test.shape[0]
    assert not np.any(np.isnan(F))
    
    # 4. Test final diffuse intensity
    Id = onephonon.apply_disorder()
    Id = Id.reshape(onephonon.map_shape)
    assert Id.shape == onephonon.map_shape
    # Ensure that at least one value is not NaN in the diffuse intensity map
    assert np.count_nonzero(~np.isnan(Id)) > 0
    
    # 5. Test expected symmetries
    # Get central slice
    central_h = Id.shape[0] // 2
    central_slice = Id[central_h, :, :]
    # Test inversion symmetry
    np.testing.assert_allclose(
         central_slice,
         np.flip(central_slice),
         rtol=1e-5
    )
