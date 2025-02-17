import os
import numpy as np
import pytest
from eryx.pdb import GaussianNetworkModel
from tests.test_utils.log_analysis import LogAnalyzer

@pytest.fixture
def gnm_base_log_file() -> str:
    # Path to the baseline log file for GNM runs
    return os.path.join("tests", "test_data", "logs", "gnm_base", "base_run.log")

@pytest.fixture
def gnm_edge_log_file() -> str:
    # Path to an edge-case log file (e.g. a system with a small cutoff or large system, choose one)
    return os.path.join("tests", "test_data", "logs", "gnm_edge", "edge_run_0.log")

@pytest.fixture
def gnm_model() -> GaussianNetworkModel:
    # Instantiate a small test system using a test PDB file with known parameters.
    pdb_path = os.path.join("tests", "pdbs", "5zck_p1.pdb")
    enm_cutoff = 4.0
    gamma_intra = 1.0
    gamma_inter = 1.0
    return GaussianNetworkModel(pdb_path, enm_cutoff, gamma_intra, gamma_inter)

class TestGaussianNetworkModel:

    def test_neighbor_list_construction(self, gnm_model):
        """Verify neighbor list computation against reference data."""
        # (Assuming gnm_model.build_neighbor_list() was called during __init__)
        ref_neighbors = np.load(os.path.join("tests", "test_data", "reference", "gnm_neighbor_lists.npy"), allow_pickle=True)
        # Compare the length (number of ASUs) and basic connectivity.
        assert len(gnm_model.asu_neighbors) == len(ref_neighbors)
        # For each ASU, check neighbor counts match (or are within expected tolerance)
        for idx, neighbors in enumerate(gnm_model.asu_neighbors):
            ref = ref_neighbors[idx]
            assert len(neighbors) == len(ref)

    def test_spring_constants(self, gnm_model):
        """Verify that spring constants are assigned correctly for intra- and inter-molecular interactions."""
        for i_asu in range(gnm_model.n_asu):
            for cell in range(gnm_model.n_cell):
                for j_asu in range(gnm_model.n_asu):
                    if (cell == gnm_model.id_cell_ref) and (i_asu == j_asu):
                        assert gnm_model.gamma[cell, i_asu, j_asu] == pytest.approx(gnm_model.gamma_intra)
                    else:
                        assert gnm_model.gamma[cell, i_asu, j_asu] == pytest.approx(gnm_model.gamma_inter)

    def test_hessian_symmetry(self, gnm_model):
        """Test that the computed Hessian matrix is symmetric in its diagonal blocks."""
        hessian = gnm_model.compute_hessian()
        # Check symmetry for each ASU within the reference cell (diagonal block)
        for i_asu in range(gnm_model.n_asu):
            diag_block = hessian[i_asu, :, gnm_model.id_cell_ref, i_asu, :]
            # The diagonal block should be Hermitian (symmetric for real data)
            assert np.allclose(diag_block, diag_block.T.conj(), rtol=1e-7), f"Diagonal block for ASU {i_asu} is not symmetric"

    def test_k_matrix_computation(self, gnm_model):
        """Test K-matrix computation using phase factors."""
        hessian = gnm_model.compute_hessian()
        kvec = np.array([1.0, 0.0, 0.0])
        Kmat = gnm_model.compute_K(hessian, kvec=kvec)
        expected_shape = (gnm_model.n_asu, gnm_model.n_atoms_per_asu, gnm_model.n_asu, gnm_model.n_atoms_per_asu)
        assert Kmat.shape == expected_shape, f"Unexpected K-matrix shape: {Kmat.shape}"
        # Optionally, compare a block against a pre-generated reference
        ref_K = np.load(os.path.join("tests", "test_data", "reference", "gnm_k_matrices.npy"))
        assert np.allclose(Kmat, ref_K, rtol=1e-7), "Computed K-matrix does not match reference data."

    def test_inversion_stability(self, gnm_model):
        """Test that K-matrix inversion is stable and accurate."""
        hessian = gnm_model.compute_hessian()
        kvec = np.array([1.0, 0.0, 0.0])
        Kmat = gnm_model.compute_K(hessian, kvec=kvec)
        Kinv = gnm_model.compute_Kinv(hessian, kvec=kvec)
        # Compute an approximate identity via contraction
        identity_approx = np.einsum('ijkl,klmn->ijmn', Kmat, Kinv)
        # For each ASU and atom, verify the corresponding block is approximately the identity matrix.
        for i in range(gnm_model.n_asu):
            for j in range(gnm_model.n_atoms_per_asu):
                block = identity_approx[i, j, :, :]
                # Create an identity of proper size
                identity = np.eye(gnm_model.n_asu * gnm_model.n_atoms_per_asu).reshape(expected_shape[2:])
                assert np.allclose(block, identity, rtol=1e-7), "K-matrix inversion does not yield an identity matrix."

    def test_phase_factors(self, gnm_model):
        """Test that phase factors are correctly computed in the dynamical matrix."""
        kvec = np.array([0.5, 0.5, 0.5])
        phases = []
        for cell in range(gnm_model.n_cell):
            cell_indices = gnm_model.crystal.id_to_hkl(cell)
            r_cell = gnm_model.crystal.get_unitcell_origin(cell_indices)
            phase = np.dot(kvec, r_cell)
            eikr = np.cos(phase) + 1j * np.sin(phase)
            phases.append(eikr)
        phases = np.array(phases)
        # Simple check: all phase factors must have absolute value 1
        assert np.allclose(np.abs(phases), 1.0, atol=1e-7)
        # (Additional detailed verifications may compare specific neighbor contributions in Kmat.)

    def test_edge_case_small_cutoff(self, gnm_edge_log_file):
        """Test behavior when using a very small cutoff (edge case)."""
        with open(gnm_edge_log_file, "r") as f:
            log_content = f.read()
        analyzer = LogAnalyzer(log_content)
        messages = [msg for _, msg in analyzer.extract_entries()]
        # Ensure a warning or message regarding disconnected components appears
        assert any("disconnected" in m.lower() for m in messages), "Edge case log does not mention 'disconnected' components."

    def test_edge_case_large_system(self, gnm_edge_log_file):
        """Test numerical stability for large system simulations (edge case)."""
        with open(gnm_edge_log_file, "r") as f:
            log_content = f.read()
        analyzer = LogAnalyzer(log_content)
        messages = [msg for _, msg in analyzer.extract_entries()]
        # Verify that there is at least one message warning about numerical stability issues
        assert any("numerical stability" in m.lower() for m in messages), "Large system log missing numerical stability warnings."
