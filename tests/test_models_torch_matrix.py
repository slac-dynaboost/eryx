import unittest
import os
import torch
import numpy as np
from eryx.models_torch import OnePhonon
from eryx.autotest.torch_testing import TorchTesting
from eryx.autotest.logger import Logger
from eryx.autotest.functionmapping import FunctionMapping
from unittest.mock import patch, MagicMock

class TestOnePhononMatrixConstruction(unittest.TestCase):
    def setUp(self):
        # Set up the testing framework
        self.logger = Logger()
        self.function_mapping = FunctionMapping()
        self.torch_testing = TorchTesting(self.logger, self.function_mapping, rtol=1e-5, atol=1e-8)
        
        # Set device to CPU for consistent testing
        self.device = torch.device('cpu')
        
        # Log file prefixes for ground truth data
        self.build_A_log = "logs/eryx.models._build_A"
        self.build_M_log = "logs/eryx.models._build_M"
        self.build_M_allatoms_log = "logs/eryx.models._build_M_allatoms"
        self.project_M_log = "logs/eryx.models._project_M"
        
        # Ensure log files exist
        for log_file in [self.build_A_log, self.build_M_log, self.build_M_allatoms_log, self.project_M_log]:
            log_file_path = f"{log_file}.log"
            self.assertTrue(os.path.exists(log_file_path), f"Log file {log_file_path} not found")
        
        # Create a minimal instance of OnePhonon for testing individual methods
        # This requires special initialization since we're testing internal methods
        # In a real test, we would use mocks or create a properly initialized instance
        self.model = self._create_test_model()
    
    def _create_test_model(self):
        """Create a minimal test model with necessary attributes for matrix methods."""
        model = OnePhonon.__new__(OnePhonon)  # Create instance without calling __init__
        
        # Set necessary attributes for the matrix methods
        model.device = self.device
        model.group_by = 'asu'
        model.n_asu = 2
        model.n_atoms_per_asu = 3
        model.n_dof_per_asu_actual = model.n_atoms_per_asu * 3
        model.n_dof_per_asu = 6  # For 'asu' group_by
        
        return model
    
    def test_build_A(self):
        """Test _build_A method with ground truth data."""
        # Load log data for this method
        logs = self.logger.loadLog(f"{self.build_A_log}.log")
        
        # Process each input/output pair from logs
        for i in range(len(logs) // 2):
            # Get input data and expected output
            instance_data = self.logger.serializer.deserialize(logs[2*i]['args'])[0]
            expected_output = self.logger.serializer.deserialize(logs[2*i+1]['result'])
            
            # Create a partially initialized OnePhonon instance
            model = OnePhonon.__new__(OnePhonon)
            
            # Set necessary attributes from instance_data
            model.device = self.device
            model.group_by = instance_data.group_by
            model.n_asu = instance_data.n_asu
            model.n_atoms_per_asu = instance_data.n_atoms_per_asu
            model.n_dof_per_asu_actual = instance_data.n_dof_per_asu_actual
            model.n_dof_per_asu = instance_data.n_dof_per_asu
            
            # Mock the crystal attribute and get_asu_xyz method
            model.crystal = MagicMock()
            
            # Configure the mock to return tensor data when get_asu_xyz is called
            def get_asu_xyz_side_effect(asu_id, unit_cell=None):
                # Convert numpy array from instance_data to tensor
                xyz_data = instance_data.crystal.get_asu_xyz(asu_id)
                return torch.tensor(xyz_data, device=self.device)
            
            model.crystal.get_asu_xyz.side_effect = get_asu_xyz_side_effect
            
            # Call the method
            model._build_A()
            
            # Convert result to numpy for comparison
            actual_output = model.Amat.cpu().detach().numpy()
            
            # Compare with expected output
            self.assertTrue(np.allclose(actual_output, expected_output, rtol=1e-5, atol=1e-8),
                           "Results don't match ground truth")
    
    def test_build_M_allatoms(self):
        """Test _build_M_allatoms method with ground truth data."""
        # Load log data for this method
        logs = self.logger.loadLog(f"{self.build_M_allatoms_log}.log")
        
        # Process each input/output pair from logs
        for i in range(len(logs) // 2):
            # Get input data and expected output
            instance_data = self.logger.serializer.deserialize(logs[2*i]['args'])[0]
            expected_output = self.logger.serializer.deserialize(logs[2*i+1]['result'])
            
            # Create a partially initialized OnePhonon instance
            model = OnePhonon.__new__(OnePhonon)
            
            # Set necessary attributes from instance_data
            model.device = self.device
            model.n_asu = instance_data.n_asu
            model.n_atoms_per_asu = instance_data.n_atoms_per_asu
            model.n_dof_per_asu_actual = instance_data.n_dof_per_asu_actual
            
            # Mock the crystal.model.elements attribute
            model.crystal = MagicMock()
            model.crystal.model = MagicMock()
            
            # Create mock Element objects with weight attributes
            class MockElement:
                def __init__(self, weight):
                    self.weight = weight
            
            # Extract element weights from instance_data
            element_weights = []
            for structure in instance_data.crystal.model.elements:
                structure_weights = [elem.weight for elem in structure]
                element_weights.append([MockElement(w) for w in structure_weights])
            
            model.crystal.model.elements = element_weights
            
            # Call the method
            actual_output = model._build_M_allatoms()
            
            # Convert result to numpy for comparison
            actual_output_np = actual_output.cpu().detach().numpy()
            
            # Compare with expected output
            self.assertTrue(np.allclose(actual_output_np, expected_output, rtol=1e-5, atol=1e-8),
                           "Results don't match ground truth")
    
    def test_project_M(self):
        """Test _project_M method with ground truth data."""
        # Load log data for this method
        logs = self.logger.loadLog(f"{self.project_M_log}.log")
        
        # Process each input/output pair from logs
        for i in range(len(logs) // 2):
            # Get input data and expected output
            args = self.logger.serializer.deserialize(logs[2*i]['args'])
            instance_data = args[0]
            M_allatoms_data = args[1]
            expected_output = self.logger.serializer.deserialize(logs[2*i+1]['result'])
            
            # Create a partially initialized OnePhonon instance
            model = OnePhonon.__new__(OnePhonon)
            
            # Set necessary attributes from instance_data
            model.device = self.device
            model.n_asu = instance_data.n_asu
            model.n_dof_per_asu = instance_data.n_dof_per_asu
            
            # Set up the Amat attribute using the instance data
            model.Amat = torch.tensor(instance_data.Amat, device=self.device)
            
            # Convert input M_allatoms to tensor
            M_allatoms = torch.tensor(M_allatoms_data, device=self.device)
            
            # Call the method
            actual_output = model._project_M(M_allatoms)
            
            # Convert result to numpy for comparison
            actual_output_np = actual_output.cpu().detach().numpy()
            
            # Compare with expected output
            self.assertTrue(np.allclose(actual_output_np, expected_output, rtol=1e-5, atol=1e-8),
                           "Results don't match ground truth")
    
    def test_build_M(self):
        """Test _build_M method with ground truth data."""
        # Load log data for this method
        logs = self.logger.loadLog(f"{self.build_M_log}.log")
        
        # Process each input/output pair from logs
        for i in range(len(logs) // 2):
            # Get input data and expected output
            instance_data = self.logger.serializer.deserialize(logs[2*i]['args'])[0]
            expected_output = self.logger.serializer.deserialize(logs[2*i+1]['result'])
            
            # Create a partially initialized OnePhonon instance
            model = OnePhonon.__new__(OnePhonon)
            
            # Set necessary attributes from instance_data
            model.device = self.device
            model.group_by = instance_data.group_by
            model.n_asu = instance_data.n_asu
            model.n_atoms_per_asu = instance_data.n_atoms_per_asu
            model.n_dof_per_asu_actual = instance_data.n_dof_per_asu_actual
            model.n_dof_per_asu = instance_data.n_dof_per_asu
            
            if hasattr(instance_data, 'Amat'):
                model.Amat = torch.tensor(instance_data.Amat, device=self.device)
            
            # Mock _build_M_allatoms and _project_M methods
            original_build_M_allatoms = model._build_M_allatoms if hasattr(model, '_build_M_allatoms') else None
            original_project_M = model._project_M if hasattr(model, '_project_M') else None
            
            def mock_build_M_allatoms():
                M_allatoms_data = instance_data._build_M_allatoms() if hasattr(instance_data, '_build_M_allatoms') else np.zeros((model.n_asu, model.n_dof_per_asu_actual, model.n_asu, model.n_dof_per_asu_actual))
                return torch.tensor(M_allatoms_data, device=model.device)
            
            def mock_project_M(M_allatoms):
                projected_data = instance_data._project_M(M_allatoms.cpu().numpy()) if hasattr(instance_data, '_project_M') else np.zeros((model.n_asu, model.n_dof_per_asu, model.n_asu, model.n_dof_per_asu))
                return torch.tensor(projected_data, device=model.device)
            
            model._build_M_allatoms = MagicMock(side_effect=mock_build_M_allatoms)
            model._project_M = MagicMock(side_effect=mock_project_M)
            
            # Call the method
            model._build_M()
            
            # Restore original methods if they existed
            if original_build_M_allatoms:
                model._build_M_allatoms = original_build_M_allatoms
            if original_project_M:
                model._project_M = original_project_M
            
            # Convert result to numpy for comparison
            actual_output = model.Linv.cpu().detach().numpy()
            
            # Compare with expected output
            self.assertTrue(np.allclose(actual_output, expected_output, rtol=1e-5, atol=1e-8),
                           "Results don't match ground truth")
    
    def test_gradient_flow(self):
        """Test gradient flow through matrix operations."""
        # Create a test model with attributes that support gradients
        model = OnePhonon.__new__(OnePhonon)
        model.device = self.device
        model.group_by = 'asu'
        model.n_asu = 2
        model.n_atoms_per_asu = 3
        model.n_dof_per_asu_actual = 9  # 3 atoms * 3 coordinates
        model.n_dof_per_asu = 6  # 3 translations + 3 rotations
        
        # Create mock crystal with get_asu_xyz method that returns tensors requiring gradients
        model.crystal = MagicMock()
        
        # Store the created tensors to check their gradients later
        model._coords_tensors = []
        
        # Mock coordinates with requires_grad=True
        def get_asu_xyz_side_effect(asu_id, unit_cell=None):
            # Create coordinates that will produce non-zero gradients
            # Use a pattern that ensures the skew-symmetric matrix has non-zero elements
            coords = torch.ones(model.n_atoms_per_asu, 3, device=model.device) * (asu_id + 1.0)
            # Add some variation to ensure unique gradients
            coords = coords + torch.randn(model.n_atoms_per_asu, 3, device=model.device) * 0.1
            coords.requires_grad_(True)
            model._coords_tensors.append(coords)
            return coords
        
        model.crystal.get_asu_xyz.side_effect = get_asu_xyz_side_effect
        
        # Test gradient flow through _build_A
        model._build_A()
        self.assertIsNotNone(model.Amat)
        
        # Create a loss function based on the output that will produce meaningful gradients
        # Use a more complex function than just sum() to ensure non-uniform gradients
        loss = (model.Amat * torch.randn_like(model.Amat)).sum()
        loss.backward()
        
        # Check that gradients flowed back to the inputs
        for coords in model._coords_tensors:
            self.assertIsNotNone(coords.grad)
            self.assertFalse(torch.allclose(coords.grad, torch.zeros_like(coords.grad)))
        
        # Test gradient flow through the other matrix operations
        # First, clear gradients
        for coords in model._coords_tensors:
            coords.grad = None
        
        # Mock the crystal.model.elements attribute for _build_M_allatoms
        model.crystal.model = MagicMock()
        
        class MockElement:
            def __init__(self, weight):
                self.weight = weight
        
        # Create elements with random weights
        element_weights = []
        for _ in range(model.n_asu):
            structure_weights = [MockElement(float(i)+1.0) for i in range(model.n_atoms_per_asu)]
            element_weights.append(structure_weights)
        
        model.crystal.model.elements = element_weights
        
        # Call _build_M_allatoms and verify gradients can flow
        M_allatoms = model._build_M_allatoms()
        
        # Verify tensor was created and has correct shape
        self.assertIsNotNone(M_allatoms)
        self.assertEqual(M_allatoms.shape, (model.n_asu, model.n_dof_per_asu_actual, 
                                            model.n_asu, model.n_dof_per_asu_actual))
        
        # Test gradient flow through _project_M
        # Create a tensor for Amat that requires gradients
        model.Amat = torch.randn((model.n_asu, model.n_dof_per_asu_actual, model.n_dof_per_asu), 
                                device=model.device, requires_grad=True)
        
        # Call _project_M
        Mmat = model._project_M(M_allatoms)
        
        # Verify tensor was created and has correct shape
        self.assertIsNotNone(Mmat)
        self.assertEqual(Mmat.shape, (model.n_asu, model.n_dof_per_asu, 
                                     model.n_asu, model.n_dof_per_asu))
        
        # Create a loss based on the output with non-uniform gradients
        # Multiply by random tensor to ensure non-zero gradients throughout
        random_weights = torch.randn_like(Mmat)
        loss = (Mmat * random_weights).sum()
        loss.backward()
        
        # Check that gradients flowed back to the inputs
        self.assertIsNotNone(model.Amat.grad)
        self.assertFalse(torch.allclose(model.Amat.grad, torch.zeros_like(model.Amat.grad)))

if __name__ == '__main__':
    unittest.main()
