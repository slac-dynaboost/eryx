import unittest
import torch
import numpy as np
import os

from eryx.models_torch import OnePhonon as TorchOnePhonon
# Assuming TensorComparison utility exists for detailed comparison
try:
    from tests.torch_test_utils import TensorComparison
except ImportError:
    TensorComparison = None # Fallback if not available

class TestModeEquivalence(unittest.TestCase):

    def setUp(self):
        self.pdb_path = 'tests/pdbs/5zck_p1.pdb' # Or your preferred test PDB
        # Use relatively small grid for faster testing
        self.hsampling = [-2, 2, 2]
        self.ksampling = [-2, 2, 2]
        self.lsampling = [-2, 2, 2]
        self.device = torch.device('cpu') # Use CPU for consistency
        self.common_params = {
            'expand_p1': True,
            'group_by': 'asu',
            'res_limit': 0.0,
            'model': 'gnm', # Important for testing computed ADPs
            'gnm_cutoff': 4.0,
            'gamma_intra': 1.0,
            'gamma_inter': 1.0,
            'n_processes': 1
        }
        # Tolerances for comparison (adjust as needed)
        self.rtol = 1e-5
        self.atol = 1e-7

        if not os.path.exists(self.pdb_path):
            self.skipTest(f"Test PDB file not found: {self.pdb_path}")

    def _run_and_compare(self, use_data_adp: bool):
        """Helper function to run both modes and compare results."""
        print(f"\n--- Testing Equivalence (use_data_adp={use_data_adp}) ---")

        # --- 1. Grid Mode Run ---
        print("Initializing grid model...")
        model_grid = TorchOnePhonon(
            pdb_path=self.pdb_path,
            hsampling=self.hsampling,
            ksampling=self.ksampling,
            lsampling=self.lsampling,
            device=self.device,
            **self.common_params
        )
        print("Running apply_disorder (grid)...")
        intensity_grid = model_grid.apply_disorder(use_data_adp=use_data_adp)
        q_vectors_from_grid = model_grid.q_grid.clone().detach() # Extract q-vectors
        print(f"Grid intensity shape: {intensity_grid.shape}")

        # --- 2. Arbitrary-Q Mode Run ---
        print("\nInitializing arbitrary-q model...")
        model_q = TorchOnePhonon(
            pdb_path=self.pdb_path,
            q_vectors=q_vectors_from_grid, # Use extracted q-vectors
            # *** IMPORTANT: Pass sampling params even in arbitrary-q mode ***
            # They might be needed internally (e.g., for compute_covariance_matrix)
            hsampling=self.hsampling,
            ksampling=self.ksampling,
            lsampling=self.lsampling,
            device=self.device,
            **self.common_params
        )
        print("Running apply_disorder (arbitrary-q)...")
        intensity_q = model_q.apply_disorder(use_data_adp=use_data_adp)
        print(f"Arbitrary-q intensity shape: {intensity_q.shape}")

        # --- 3. Comparison ---
        print("\nComparing results...")
        self.assertEqual(intensity_grid.shape, intensity_q.shape, "Output shapes do not match")

        # Convert to NumPy for comparison
        intensity_grid_np = intensity_grid.detach().cpu().numpy()
        intensity_q_np = intensity_q.detach().cpu().numpy()

        # Compare using allclose, handling NaNs
        are_close = np.allclose(intensity_grid_np, intensity_q_np,
                                rtol=self.rtol, atol=self.atol, equal_nan=True)

        if not are_close:
            mask = ~np.isnan(intensity_grid_np) & ~np.isnan(intensity_q_np)
            if np.any(mask):
                 max_diff = np.max(np.abs(intensity_grid_np[mask] - intensity_q_np[mask]))
                 print(f"Max difference: {max_diff}")
                 max_idx = np.unravel_index(np.argmax(np.abs(intensity_grid_np[mask] - intensity_q_np[mask])), intensity_grid_np[mask].shape) # Find index within the masked array
                 # Map back to original index if needed (more complex)
                 print(f"Example mismatch at an index (within valid mask): grid={intensity_grid_np[mask][max_idx]}, q={intensity_q_np[mask][max_idx]}")
            else:
                 print("Comparison skipped - all values are NaN.")


        self.assertTrue(are_close, f"Intensity results differ between modes (use_data_adp={use_data_adp})")
        print(f"Equivalence test PASSED (use_data_adp={use_data_adp})")

    def test_equivalence_with_data_adp(self):
        """Test equivalence using ADPs from PDB data."""
        self._run_and_compare(use_data_adp=True)

    def test_equivalence_with_computed_adp(self):
        """Test equivalence using internally computed ADPs (requires model='gnm')."""
        if self.common_params.get('model') != 'gnm':
            self.skipTest("Skipping computed ADP test because model is not 'gnm'")
        self._run_and_compare(use_data_adp=False)

if __name__ == '__main__':
    unittest.main()
