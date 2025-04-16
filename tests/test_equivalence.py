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
        print(f"Grid model V shape: {model_grid.V.shape}, Winv shape: {model_grid.Winv.shape}")
        print(f"Grid model ADP[0]: {model_grid.ADP[0].detach().item():.6e}" if hasattr(model_grid,'ADP') and model_grid.ADP is not None and model_grid.ADP.numel() > 0 else "No ADP or ADP empty")
        # Find target_q_idx for BZ point (0,0,1) in 2x2x2 grid
        target_dh, target_dk, target_dl = 0, 0, 1
        target_bz_idx = model_grid._3d_to_flat_indices_bz(torch.tensor([target_dh]), torch.tensor([target_dk]), torch.tensor([target_dl])).item()
        q_indices_for_target_bz = model_grid._at_kvec_from_miller_points((target_dh, target_dk, target_dl))
        target_q_idx = q_indices_for_target_bz[0].item() if q_indices_for_target_bz is not None and q_indices_for_target_bz.numel() > 0 else -1
        print(f"Grid model q_grid[{target_q_idx}]: {model_grid.q_grid[target_q_idx].detach().cpu().numpy() if target_q_idx != -1 else 'N/A'}")
        print(f"Grid model V[{target_bz_idx},0,0] abs: {torch.abs(model_grid.V[target_bz_idx,0,0]).detach().item():.6e}") # Check BZ idx 1 (target_bz_idx)
        print("Running apply_disorder (grid)...")
        intensity_grid = model_grid.apply_disorder(use_data_adp=use_data_adp)
        q_vectors_from_grid = model_grid.q_grid.clone().detach() # Extract q-vectors
        print(f"Grid intensity shape: {intensity_grid.shape}")
        print(f"Grid intensity[{target_q_idx}]: {intensity_grid[target_q_idx].detach().item():.6e}" if target_q_idx != -1 else 'N/A')

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
        # Check consistency after init
        print(f"Arb model V shape: {model_q.V.shape}, Winv shape: {model_q.Winv.shape}")
        print(f"Arb model ADP[0]: {model_q.ADP[0].detach().item():.6e}" if hasattr(model_q,'ADP') and model_q.ADP is not None and model_q.ADP.numel() > 0 else "No ADP or ADP empty")
        print(f"Arb model q_grid[{target_q_idx}]: {model_q.q_grid[target_q_idx].detach().cpu().numpy() if target_q_idx != -1 else 'N/A'}")
        print(f"Arb model V[{target_q_idx},0,0] abs: {torch.abs(model_q.V[target_q_idx,0,0]).detach().item():.6e}") # Check q_idx 9 (target_q_idx)

        # Verify q_grids match (already exists, ensure tolerance is tight)
        self.assertTrue(torch.allclose(model_grid.q_grid, model_q.q_grid, atol=1e-12, rtol=1e-9), "q_grids do not match")

        # Verify hkl_grids match (Crucial!)
        self.assertTrue(torch.allclose(model_grid.hkl_grid, model_q.hkl_grid, atol=1e-12, rtol=1e-9), "hkl_grids do not match")

        # Verify resolution masks match (Crucial!)
        self.assertTrue(torch.equal(model_grid.res_mask, model_q.res_mask), "Resolution masks (res_mask) do not match")

        # Verify ADPs match (if computed) (already exists)
        if not use_data_adp and hasattr(model_grid, 'ADP') and hasattr(model_q, 'ADP'):
             # Use detach() for comparison if ADP requires grad
             adp_grid = model_grid.ADP.detach() if model_grid.ADP.requires_grad else model_grid.ADP
             adp_q = model_q.ADP.detach() if model_q.ADP.requires_grad else model_q.ADP
             self.assertTrue(torch.allclose(adp_grid, adp_q, atol=1e-8, rtol=1e-5), "Computed ADPs do not match")

        # Verify V/Winv match point-by-point after expansion (Optional but informative)
        # Can be slow, maybe check only a few points like target_q_idx
        if target_q_idx != -1:
             self.assertTrue(torch.allclose(model_grid.V[target_bz_idx], model_q.V[target_q_idx], atol=1e-9), f"V mismatch for q_idx {target_q_idx}")
             self.assertTrue(torch.allclose(model_grid.Winv[target_bz_idx], model_q.Winv[target_q_idx], atol=1e-9), f"Winv mismatch for q_idx {target_q_idx}")

        print("Running apply_disorder (arbitrary-q)...")
        intensity_q = model_q.apply_disorder(use_data_adp=use_data_adp)
        print(f"Arbitrary-q intensity shape: {intensity_q.shape}")
        print(f"Arbitrary-q intensity[{target_q_idx}]: {intensity_q[target_q_idx].detach().item():.6e}" if target_q_idx != -1 else 'N/A')

        # --- 3. Comparison ---
        print("\nComparing results...")
        self.assertEqual(intensity_grid.shape, intensity_q.shape, "Output shapes do not match")

        # Convert to NumPy for comparison
        intensity_grid_np = intensity_grid.detach().cpu().numpy()
        intensity_q_np = intensity_q.detach().cpu().numpy()

        # Compare using allclose, handling NaNs
        are_close = np.allclose(intensity_grid_np, intensity_q_np,
                                rtol=self.rtol, atol=self.atol, equal_nan=True)

        # Inside _run_and_compare, after are_close is calculated:
        if not are_close:
            print("\n--- DETAILED COMPARISON (Equivalence Failed) ---")
            # Ensure NaNs are handled consistently for calculations
            nan_mask = np.isnan(intensity_grid_np) | np.isnan(intensity_q_np)
            valid_mask = ~nan_mask

            if np.sum(valid_mask) == 0:
                print("Comparison failed, and NO valid (non-NaN) points found to compare.")
            else:
                # Calculate differences only on valid points
                grid_valid = intensity_grid_np[valid_mask]
                q_valid = intensity_q_np[valid_mask]
                abs_diff_valid = np.abs(grid_valid - q_valid)
                # Add epsilon to avoid division by zero for relative diff
                rel_diff_valid = abs_diff_valid / (np.abs(grid_valid) + 1e-9)

                # --- Overall Statistics ---
                max_abs_diff = np.max(abs_diff_valid)
                mean_abs_diff = np.mean(abs_diff_valid)
                max_rel_diff = np.max(rel_diff_valid)
                mean_rel_diff = np.mean(rel_diff_valid)
                print(f"Overall Max Abs Diff:  {max_abs_diff:.6e}")
                print(f"Overall Mean Abs Diff: {mean_abs_diff:.6e}")
                print(f"Overall Max Rel Diff:  {max_rel_diff:.6e}")
                print(f"Overall Mean Rel Diff: {mean_rel_diff:.6e}")

                # --- Find Problematic Indices ---
                # Define thresholds (adjust if needed)
                rel_diff_threshold = 0.001 # 0.1% relative difference
                # Ignore points where grid intensity is very small compared to max
                grid_max_valid = np.max(grid_valid)
                abs_intensity_threshold = 1e-5 * grid_max_valid

                problematic_indices_valid = np.where(
                    (rel_diff_valid > rel_diff_threshold) &
                    (np.abs(grid_valid) > abs_intensity_threshold)
                )[0]

                # Map valid indices back to original flat indices
                original_indices = np.arange(intensity_grid_np.size)
                valid_original_indices = original_indices[valid_mask]
                problematic_original_indices = valid_original_indices[problematic_indices_valid]

                print(f"\nFound {len(problematic_original_indices)} points with RelDiff > {rel_diff_threshold:.1e} and Grid Intensity > {abs_intensity_threshold:.1e}")

                # Print details for a few problematic points
                num_to_print = min(10, len(problematic_original_indices))
                if num_to_print > 0:
                    print(f"\nDetails for first {num_to_print} problematic points (Original Index):")
                    print("Idx      Grid Val       Q Val          Abs Diff       Rel Diff")
                    print("--------------------------------------------------------------------")
                    for i in range(num_to_print):
                        orig_idx = problematic_original_indices[i]
                        valid_idx = problematic_indices_valid[i] # Index within the _valid arrays
                        g_val = grid_valid[valid_idx]
                        q_val = q_valid[valid_idx]
                        a_diff = abs_diff_valid[valid_idx]
                        r_diff = rel_diff_valid[valid_idx]
                        print(f"{orig_idx:<7d} {g_val:<14.6e} {q_val:<14.6e} {a_diff:<14.6e} {r_diff:<14.6e}")

            # Raise the original assertion failure
            self.assertTrue(are_close, f"Intensity results differ between modes (use_data_adp={use_data_adp})")
        else:
            print(f"\nEquivalence test PASSED (use_data_adp={use_data_adp})")

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
