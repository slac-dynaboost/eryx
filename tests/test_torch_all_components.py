import unittest
import numpy as np
import torch
from tests.torch_test_base import TorchComponentTestCase

class TestTorchAllComponents(TorchComponentTestCase):
    """Test suite that runs basic tests for all components to identify failing components."""
    
    def setUp(self):
        """Set up test environment with very minimal parameters for quick testing."""
        super().setUp()
        
        # Initialize test parameters if not already done by parent class
        if not hasattr(self, 'test_params'):
            self.test_params = self.default_test_params.copy()
        
        # Override with minimal test parameters
        self.test_params.update({
            'hsampling': [-1, 1, 2],
            'ksampling': [-1, 1, 2],
            'lsampling': [-1, 1, 2],
        })
    
    def test_10_kvector_construction(self):
        """Basic test of k-vector construction (runs first)."""
        self.create_models()
        
        # Run _build_kvec_Brillouin
        self.np_model._build_kvec_Brillouin()
        self.torch_model._build_kvec_Brillouin()
        
        # Check kvec dimensions
        np_shape = self.np_model.kvec.shape
        torch_shape = tuple(self.torch_model.kvec.shape)
        
        self.assertEqual(
            np_shape, torch_shape,
            f"kvec shapes don't match: NP={np_shape}, Torch={torch_shape}"
        )
        
        # Check sample k-vector values
        dh, dk, dl = 0, 0, 0
        self.assert_tensors_equal(
            self.np_model.kvec[dh, dk, dl],
            self.torch_model.kvec[dh, dk, dl],
            msg=f"kvec[{dh},{dk},{dl}] values don't match"
        )
    
    def test_20_hessian_calculation(self):
        """Basic test of hessian calculation (runs second)."""
        self.create_models()
        
        # Run compute_hessian
        np_hessian = self.np_model.compute_hessian()
        torch_hessian = self.torch_model.compute_hessian()
        
        # Check hessian dimensions
        np_shape = np_hessian.shape
        torch_shape = tuple(torch_hessian.shape)
        
        self.assertEqual(
            np_shape, torch_shape,
            f"Hessian shapes don't match: NP={np_shape}, Torch={torch_shape}"
        )
        
        # Critical check: k-vector dimension
        self.assertEqual(
            np_shape[2], torch_shape[2],
            f"Hessian k-vector dimensions don't match: NP={np_shape[2]}, Torch={torch_shape[2]}"
        )
    
    def test_30_phonon_calculation(self):
        """Basic test of phonon calculation (runs third)."""
        self.create_models()
        
        # Run compute_gnm_phonons
        self.np_model.compute_gnm_phonons()
        self.torch_model.compute_gnm_phonons()
        
        # Check eigenvector and eigenvalue dimensions
        np_v_shape = self.np_model.V.shape
        torch_v_shape = tuple(self.torch_model.V.shape)
        
        self.assertEqual(
            np_v_shape, torch_v_shape,
            f"Eigenvector (V) shapes don't match: NP={np_v_shape}, Torch={torch_v_shape}"
        )
        
        np_winv_shape = self.np_model.Winv.shape
        torch_winv_shape = tuple(self.torch_model.Winv.shape)
        
        self.assertEqual(
            np_winv_shape, torch_winv_shape,
            f"Eigenvalue (Winv) shapes don't match: NP={np_winv_shape}, Torch={torch_winv_shape}"
        )
        
        # Check NaN pattern in first k-vector eigenvalues
        np_nan_count = np.sum(np.isnan(self.np_model.Winv[0, 0, 0]))
        torch_nan_count = torch.sum(torch.isnan(self.torch_model.Winv[0, 0, 0])).item()
        
        self.assertEqual(
            np_nan_count, torch_nan_count,
            f"NaN counts don't match: NP={np_nan_count}, Torch={torch_nan_count}"
        )
    
    def test_40_apply_disorder(self):
        """Basic test of apply_disorder (runs fourth)."""
        self.create_models()
        
        # Run apply_disorder
        np_intensity = self.np_model.apply_disorder(use_data_adp=True)
        torch_intensity = self.torch_model.apply_disorder(use_data_adp=True)
        
        # Check shapes
        np_shape = np_intensity.shape
        torch_shape = tuple(torch_intensity.shape)
        
        self.assertEqual(
            np_shape, torch_shape,
            f"Intensity shapes don't match: NP={np_shape}, Torch={torch_shape}"
        )
        
        # Check basic statistics
        np_nan_count = np.sum(np.isnan(np_intensity))
        torch_nan_count = torch.sum(torch.isnan(torch_intensity)).item()
        
        self.assertEqual(
            np_nan_count, torch_nan_count,
            f"NaN counts don't match: NP={np_nan_count}, Torch={torch_nan_count}"
        )

if __name__ == '__main__':
    unittest.main()
