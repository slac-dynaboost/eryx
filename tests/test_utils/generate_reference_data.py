import os
import numpy as np
from eryx.models import OnePhonon
from eryx.scatter import compute_form_factors, structure_factors

def generate_reference_data():
    """Generate reference data for testing diffraction calculations"""
    # Create output directory
    out_dir = "tests/test_data/reference"
    os.makedirs(out_dir, exist_ok=True)

    # 1. Setup test case with small grid
    pdb_path = "tests/pdbs/5zck.pdb"
    sampling = [-2, 2, 1]
    
    onephonon = OnePhonon(
        pdb_path, 
        sampling, sampling, sampling,
        expand_p1=True,
        res_limit=100.0,
        gnm_cutoff=4.0,
        gamma_intra=1.0,
        gamma_inter=1.0
    )
    
    # 2. Generate form factor reference
    q_test = onephonon.q_grid[:10]
    ff = compute_form_factors(
        q_test,
        onephonon.model.ff_a[0],
        onephonon.model.ff_b[0], 
        onephonon.model.ff_c[0]
    )
    np.save(os.path.join(out_dir, "form_factors.npy"), ff)
    
    # 3. Generate structure factor reference  
    F = structure_factors(
        q_test,
        onephonon.model.xyz[0],
        onephonon.model.ff_a[0],
        onephonon.model.ff_b[0],
        onephonon.model.ff_c[0],
        compute_qF=True,
        project_on_components=onephonon.Amat[0]
    )
    np.save(os.path.join(out_dir, "structure_factors.npy"), F)
    
    # 4. Generate full diffraction pattern
    Id = onephonon.apply_disorder()
    np.save(os.path.join(out_dir, "diffraction_pattern.npy"), Id)

    # 5. Save test parameters
    np.savez(os.path.join(out_dir, "test_params.npz"),
             q_test=q_test,
             sampling=sampling)

if __name__ == "__main__":
    generate_reference_data()
