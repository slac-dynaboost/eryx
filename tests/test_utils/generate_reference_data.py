import os
import numpy as np
from eryx.models import OnePhonon
from eryx.scatter import compute_form_factors, structure_factors
from eryx.pdb import GaussianNetworkModel

def generate_reference_data():
    """Generate reference data for testing diffraction calculations"""
    # Create output directory
    out_dir = "tests/test_data/reference"
    os.makedirs(out_dir, exist_ok=True)

    # 1. Setup test case with small grid
    pdb_path = "tests/pdbs/5zck_p1.pdb"
    hsampling = [-4, 4, 3]
    ksampling = [-17, 17, 3]
    lsampling = [-29, 29, 3]
    
    onephonon = OnePhonon(
        pdb_path, 
        hsampling, ksampling, lsampling,
        expand_p1=True,
        res_limit=-1.0,
        gnm_cutoff=4.0,
        gamma_intra=1.0,
        gamma_inter=1.0
    )
    # Re-load atomic model with a valid frame (e.g. frame=-1) to ensure xyz is not None
    from eryx.pdb import AtomicModel
    onephonon.model = AtomicModel(pdb_path, expand_p1=True, frame=-1)
    
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
    non_nan_idx = np.where(~np.isnan(Id))[0]
    first_5_values = Id[non_nan_idx][:5]
    first_5_indices = non_nan_idx[:5]
    print("First 5 non nan values of Id:", first_5_values)
    print("Corresponding indices:", first_5_indices)

    # 5. Generate and save GNM neighbor lists
    gnm = GaussianNetworkModel(pdb_path, enm_cutoff=4.0, gamma_intra=1.0, gamma_inter=1.0)
    np.save(os.path.join(out_dir, "gnm_neighbor_lists.npy"), gnm.asu_neighbors)
    
    # 6. Generate and save GNM K-matrix reference data using a test k-vector
    hessian = gnm.compute_hessian()
    kvec = np.array([1.0, 0.0, 0.0])
    Kmat = gnm.compute_K(hessian, kvec=kvec)
    np.save(os.path.join(out_dir, "gnm_k_matrices.npy"), Kmat)

    # 7. Save test parameters
    np.savez(os.path.join(out_dir, "test_params.npz"),
             q_test=q_test,
             sampling={"hsampling": hsampling, "ksampling": ksampling, "lsampling": lsampling})

if __name__ == "__main__":
    generate_reference_data()
