import unittest
import torch
import numpy as np
from unittest.mock import MagicMock, patch
from eryx.models_torch import OnePhonon

class TestOnePhononInitialization(unittest.TestCase):
    def setUp(self):
        """Set up test environment and data."""
        # Set up test data
        self.device = torch.device('cpu')  # Use CPU for consistent testing
        self.pdb_path = "tests/pdbs/5zck_p1.pdb"
        self.hsampling = (-4, 4, 3)
        self.ksampling = (-17, 17, 3)
        self.lsampling = (-29, 29, 3)
        
    @patch('eryx.models_torch.OnePhonon._setup')
    @patch('eryx.models_torch.OnePhonon._setup_phonons')
    def test_init_parameters(self, mock_setup_phonons, mock_setup):
        """Test parameter initialization in __init__."""
        # Create model with mocked setup methods
        model = OnePhonon(
            self.pdb_path,
            self.hsampling,
            self.ksampling,
            self.lsampling,
            expand_p1=True,
            res_limit=0.0,
            gnm_cutoff=4.0,
            gamma_intra=1.0,
            gamma_inter=1.0,
            device=self.device
        )
        
        # Verify parameters were set correctly
        self.assertEqual(model.hsampling, self.hsampling)
        self.assertEqual(model.ksampling, self.ksampling)
        self.assertEqual(model.lsampling, self.lsampling)
        self.assertEqual(model.device, self.device)
        
        # Verify _setup and _setup_phonons were called with correct parameters
        mock_setup.assert_called_once_with(self.pdb_path, True, 0.0, 'asu')
        mock_setup_phonons.assert_called_once_with(self.pdb_path, 'gnm', 4.0, 1.0, 1.0)
    
    def test_device_selection(self):
        """Test device selection logic."""
        # Test default device selection
        with patch('torch.cuda.is_available', return_value=True):
            with patch('eryx.models_torch.OnePhonon._setup'), \
                 patch('eryx.models_torch.OnePhonon._setup_phonons'):
                model = OnePhonon(
                    self.pdb_path,
                    self.hsampling,
                    self.ksampling,
                    self.lsampling
                )
                self.assertEqual(model.device, torch.device('cuda'))
        
        # Test explicit device selection
        with patch('eryx.models_torch.OnePhonon._setup'), \
             patch('eryx.models_torch.OnePhonon._setup_phonons'):
            model = OnePhonon(
                self.pdb_path,
                self.hsampling,
                self.ksampling,
                self.lsampling,
                device=torch.device('cpu')
            )
            self.assertEqual(model.device, torch.device('cpu'))

if __name__ == '__main__':
    unittest.main()
