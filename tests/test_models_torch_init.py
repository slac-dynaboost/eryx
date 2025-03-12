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
    
    @patch('eryx.pdb.AtomicModel')
    @patch('eryx.pdb.Crystal')
    @patch('eryx.adapters.PDBToTensor')
    @patch('eryx.map_utils_torch.generate_grid')
    @patch('eryx.map_utils_torch.get_resolution_mask')
    def test_setup_method(self, mock_get_resolution_mask, mock_generate_grid, 
                         mock_pdb_adapter, mock_crystal, mock_atomic_model):
        """Test _setup method with mocked dependencies."""
        # Configure mocks
        mock_atomic_model.return_value = MagicMock()
        mock_crystal.return_value = MagicMock()
        mock_pdb_adapter.return_value = MagicMock()
        mock_pdb_adapter.return_value.convert_atomic_model.return_value = {
            'A_inv': torch.eye(3, device=self.device),
            'cell': torch.ones(6, device=self.device)
        }
        mock_pdb_adapter.return_value.convert_crystal.return_value = {
            'n_cell': 27,
            'n_asu': 4,
            'n_atoms_per_asu': 100,
            'hkl_to_id': lambda x: 0,
            'id_to_hkl': lambda x: [0, 0, 0]
        }
        mock_generate_grid.return_value = (
            torch.ones((100, 3), device=self.device),
            (10, 10, 10)
        )
        mock_get_resolution_mask.return_value = (
            torch.ones(100, dtype=torch.bool, device=self.device),
            torch.ones(100, device=self.device)
        )
        
        # Create model
        model = OnePhonon.__new__(OnePhonon)
        model.device = self.device
        model.hsampling = self.hsampling
        model.ksampling = self.ksampling
        model.lsampling = self.lsampling
        
        # Call _setup
        model._setup(self.pdb_path, True, 0.0, 'asu')
        
        # Verify method calls
        mock_atomic_model.assert_called_once_with(self.pdb_path, True)
        mock_crystal.assert_called_once()
        mock_pdb_adapter.return_value.convert_atomic_model.assert_called_once()
        mock_pdb_adapter.return_value.convert_crystal.assert_called_once()
        mock_generate_grid.assert_called_once()
        mock_get_resolution_mask.assert_called_once()
        
        # Verify attributes
        self.assertEqual(model.n_asu, 4)
        self.assertEqual(model.n_atoms_per_asu, 100)
        self.assertEqual(model.n_dof_per_asu, 6)  # Because group_by='asu'
        self.assertEqual(model.n_dof_per_cell, 24)  # 4 ASUs * 6 DOF per ASU
    
    @patch('eryx.pdb.GaussianNetworkModel')
    @patch('eryx.adapters.PDBToTensor')
    def test_setup_phonons(self, mock_pdb_adapter, mock_gnm):
        """Test _setup_phonons method with mocked dependencies."""
        # Configure mocks
        mock_gnm.return_value = MagicMock()
        mock_pdb_adapter.return_value = MagicMock()
        mock_pdb_adapter.return_value.convert_gnm.return_value = {
            'enm_cutoff': 4.0,
            'gamma_intra': 1.0,
            'gamma_inter': 1.0
        }
        
        # Create model with necessary attributes
        model = OnePhonon.__new__(OnePhonon)
        model.device = self.device
        model.hsampling = self.hsampling
        model.ksampling = self.ksampling
        model.lsampling = self.lsampling
        model.n_asu = 4
        model.n_dof_per_asu = 6
        model._build_A = MagicMock()
        model._build_M = MagicMock()
        model._build_kvec_Brillouin = MagicMock()
        model.compute_gnm_phonons = MagicMock()
        model.compute_covariance_matrix = MagicMock()
        
        # Call _setup_phonons
        model._setup_phonons(self.pdb_path, 'gnm', 4.0, 1.0, 1.0)
        
        # Verify tensor initialization
        self.assertEqual(model.kvec.shape, (model.hsampling[2], model.ksampling[2], model.lsampling[2], 3))
        self.assertEqual(model.kvec_norm.shape, (model.hsampling[2], model.ksampling[2], model.lsampling[2], 1))
        self.assertEqual(model.V.shape, (model.hsampling[2], model.ksampling[2], model.lsampling[2], 
                                        model.n_asu * model.n_dof_per_asu, model.n_asu * model.n_dof_per_asu))
        self.assertEqual(model.Winv.shape, (model.hsampling[2], model.ksampling[2], model.lsampling[2],
                                           model.n_asu * model.n_dof_per_asu))
        
        # Verify method calls
        model._build_A.assert_called_once()
        model._build_M.assert_called_once()
        model._build_kvec_Brillouin.assert_called_once()
        mock_gnm.assert_called_once_with(self.pdb_path, 4.0, 1.0, 1.0)
        mock_pdb_adapter.return_value.convert_gnm.assert_called_once()
        model.compute_gnm_phonons.assert_called_once()
        model.compute_covariance_matrix.assert_called_once()
    
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
