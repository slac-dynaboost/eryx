import unittest
import numpy as np
import torch
import tempfile
import os
from tests.torch_test_base import TorchComponentTestCase

class TestTorchDisorder(TorchComponentTestCase):
    """Test suite for PyTorch apply_disorder method."""
    
    def setUp(self):
        """Set up test environment with default configuration."""
        super().setUp()
        # Use smaller test parameters for faster testing
        self.test_params = {
            'pdb_path': 'tests/pdbs/5zck_p1.pdb',
            'hsampling': [-1, 1, 2],  # Very small grid for faster testing
            'ksampling': [-1, 1, 2],
            'lsampling': [-1, 1, 2],
            'expand_p1': True,
            'res_limit': 0.0,
            'gnm_cutoff': 4.0,
            'gamma_intra': 1.0,
            'gamma_inter': 1.0
        }
    
    def test_apply_disorder_output_shape(self):
        """Test that apply_disorder returns output with correct shape."""
        # Create models with minimal parameters for speed
        self.create_models()
        
        # Execute apply_disorder on both models
        np_intensity = self.np_model.apply_disorder(use_data_adp=True)
        torch_intensity = self.torch_model.apply_disorder(use_data_adp=True)
        
        # Check shapes match
        np_shape = np_intensity.shape
        torch_shape = tuple(torch_intensity.shape)
        
        self.assertEqual(
            np_shape, torch_shape,
            f"Output shapes don't match: NP={np_shape}, Torch={torch_shape}"
        )
    
    def test_apply_disorder_values(self):
        """Test that apply_disorder returns similar values in NumPy and PyTorch."""
        # Create models with minimal parameters for speed
        self.create_models()
        
        # Execute apply_disorder on both models
        np_intensity = self.np_model.apply_disorder(use_data_adp=True)
        torch_intensity = self.torch_model.apply_disorder(use_data_adp=True)
        
        # Convert PyTorch tensor to NumPy array
        torch_intensity_np = torch_intensity.detach().cpu().numpy()
        
        # Create mask for valid (non-NaN) values
        mask = ~np.isnan(np_intensity) & ~np.isnan(torch_intensity_np)
        self.assertTrue(np.any(mask), "All values are NaN in at least one intensity map")
        
        # Calculate statistics
        np_mean = np.mean(np_intensity[mask])
        torch_mean = np.mean(torch_intensity_np[mask])
        np_std = np.std(np_intensity[mask])
        torch_std = np.std(torch_intensity_np[mask])
        
        # Print statistics for debugging
        print(f"NP mean: {np_mean}, Torch mean: {torch_mean}")
        print(f"NP std: {np_std}, Torch std: {torch_std}")
        
        # Calculate correlation
        correlation = np.corrcoef(np_intensity[mask], torch_intensity_np[mask])[0, 1]
        print(f"Correlation: {correlation}")
        
        # Check mean is similar (within 5%)
        self.assertAlmostEqual(
            np_mean, torch_mean,
            delta=abs(np_mean * 0.05),
            msg=f"Mean intensity differs significantly: NP={np_mean}, Torch={torch_mean}"
        )
        
        # Check standard deviation is similar (within 10%)
        self.assertAlmostEqual(
            np_std, torch_std,
            delta=abs(np_std * 0.1),
            msg=f"Intensity std dev differs significantly: NP={np_std}, Torch={torch_std}"
        )
        
        # Check correlation is high (> 0.9)
        self.assertGreater(
            correlation, 0.9,
            f"Correlation too low: {correlation}"
        )
    
    def test_apply_disorder_with_output(self):
        """Test apply_disorder saves output files correctly."""
        # Create models with minimal parameters
        self.create_models()
        
        # Create temporary directory for outputs
        with tempfile.TemporaryDirectory() as temp_dir:
            # Execute apply_disorder with output directory
            rank = -1  # Default case
            self.np_model.apply_disorder(use_data_adp=True, outdir=temp_dir, rank=rank)
            self.torch_model.apply_disorder(use_data_adp=True, outdir=temp_dir, rank=rank)
            
            # Check output files exist
            np_file = os.path.join(temp_dir, f"rank_{rank:05d}.npy")
            torch_file = os.path.join(temp_dir, f"rank_{rank:05d}_torch.pt")
            
            self.assertTrue(
                os.path.exists(np_file),
                f"NumPy output file {np_file} not found"
            )
            
            self.assertTrue(
                os.path.exists(torch_file),
                f"PyTorch output file {torch_file} not found"
            )
            
            # Load and compare outputs
            np_intensity = np.load(np_file)
            torch_intensity = torch.load(torch_file).detach().cpu().numpy()
            
            # Create mask for valid values
            mask = ~np.isnan(np_intensity) & ~np.isnan(torch_intensity)
            self.assertTrue(np.any(mask), "All values are NaN in saved outputs")
            
            # Calculate correlation
            correlation = np.corrcoef(np_intensity[mask], torch_intensity[mask])[0, 1]
            
            # Check correlation is high (> 0.9)
            self.assertGreater(
                correlation, 0.9,
                f"Correlation too low for saved outputs: {correlation}"
            )
    
    def test_apply_disorder_gradient_flow(self):
        """Test gradient flow through apply_disorder."""
        # Skip this test if not running on a device that supports autograd
        if not torch.cuda.is_available() and not hasattr(torch.backends, 'mps') and not torch.backends.mps.is_available():
            self.skipTest("Skipping gradient test on CPU for performance reasons")
        
        # Create PyTorch model with parameters that require gradients
        from eryx.models_torch import OnePhonon as TorchOnePhonon
        
        gamma_intra = torch.tensor(1.0, requires_grad=True)
        gamma_inter = torch.tensor(1.0, requires_grad=True)
        
        # Create model with gradient-enabled parameters
        torch_model = TorchOnePhonon(
            **self.test_params,
            gamma_intra=gamma_intra,
            gamma_inter=gamma_inter,
            device=self.device
        )
        
        # Forward pass
        intensity = torch_model.apply_disorder(use_data_adp=True)
        
        # Create simple loss function (using valid values only)
        mask = ~torch.isnan(intensity)
        if torch.any(mask):
            loss = torch.mean(intensity[mask])
            
            # Backward pass
            loss.backward()
            
            # Check gradients exist and are non-zero
            self.assertIsNotNone(gamma_intra.grad)
            self.assertIsNotNone(gamma_inter.grad)
            
            # Check gradient magnitudes
            gamma_intra_grad_norm = torch.norm(gamma_intra.grad).item()
            gamma_inter_grad_norm = torch.norm(gamma_inter.grad).item()
            
            print(f"gamma_intra gradient norm: {gamma_intra_grad_norm}")
            print(f"gamma_inter gradient norm: {gamma_inter_grad_norm}")
            
            # Gradients should be non-zero
            self.assertGreater(gamma_intra_grad_norm, 1e-10)
            self.assertGreater(gamma_inter_grad_norm, 1e-10)
        else:
            self.skipTest("No valid intensity values for gradient test")

if __name__ == '__main__':
    unittest.main()
