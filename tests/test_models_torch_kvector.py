import unittest
import os
import torch
import numpy as np
from eryx.models_torch import OnePhonon
from eryx.autotest.torch_testing import TorchTesting
from eryx.autotest.logger import Logger
from eryx.autotest.functionmapping import FunctionMapping
from unittest.mock import patch, MagicMock

class TestOnePhononKvector(unittest.TestCase):
    def setUp(self):
        # Set up the testing framework
        self.logger = Logger()
        self.function_mapping = FunctionMapping()
        self.torch_testing = TorchTesting(self.logger, self.function_mapping, rtol=1e-5, atol=1e-8)
        
        # Set device to CPU for consistent testing
        self.device = torch.device('cpu')
        
        # Log file prefixes for ground truth data
        self.build_kvec_brillouin_log = "logs/eryx.models._build_kvec_Brillouin"
        self.center_kvec_log = "logs/eryx.models._center_kvec"
        self.at_kvec_miller_log = "logs/eryx.models._at_kvec_from_miller_points"
        self.compute_covariance_matrix_log = "logs/eryx.models.compute_covariance_matrix"
        
        # Ensure log files exist
        for log_file in [self.build_kvec_brillouin_log, self.center_kvec_log, 
                         self.at_kvec_miller_log, self.compute_covariance_matrix_log]:
            log_file_path = f"{log_file}.log"
            self.assertTrue(os.path.exists(log_file_path), f"Log file {log_file_path} not found")
        
        # Create minimal test model
        self.model = self._create_test_model()
    
    def _create_test_model(self):
        """Create a minimal test model with necessary attributes for k-vector methods."""
        model = OnePhonon.__new__(OnePhonon)  # Create instance without calling __init__
        
        # Set necessary attributes
        model.device = self.device
        model.hsampling = (0, 5, 2)  # Sample values for testing
        model.ksampling = (0, 5, 2)
        model.lsampling = (0, 5, 2)
        model.map_shape = (6, 6, 6)  # Sample map shape
        
        # Create a sample A_inv matrix
        model.A_inv = torch.eye(3, device=self.device, requires_grad=True)
        
        return model
    
    def test_center_kvec(self):
        """Test _center_kvec method with ground truth data."""
        # Load log data for this method
        logs = self.logger.loadLog(f"{self.center_kvec_log}.log")
        
        # Process each log entry
        for i in range(0, len(logs), 2):
            # The log format might have the function call data and result
            if 'args' in logs[i]:
                args = self.logger.serializer.deserialize(logs[i]['args'])
                expected_output = self.logger.serializer.deserialize(logs[i+1]['result'])
                
                # Call the method
                actual_output = self.model._center_kvec(*args)
                
                # Compare with expected output
                self.assertEqual(actual_output, expected_output,
                               f"Results don't match ground truth for input {args}")
    
    def test_build_kvec_brillouin(self):
        """Test _build_kvec_Brillouin method with ground truth data."""
        # Load log data for this method
        logs = self.logger.loadLog(f"{self.build_kvec_brillouin_log}.log")
        
        # Process each input/output pair from logs
        for i in range(len(logs) // 2):
            # Get input data and expected output
            instance_data = self.logger.serializer.deserialize(logs[2*i]['args'])[0]
            expected_kvec = self.logger.serializer.deserialize(logs[2*i+1]['result'][0])
            expected_kvec_norm = self.logger.serializer.deserialize(logs[2*i+1]['result'][1])
            
            # Create a partially initialized OnePhonon instance
            model = OnePhonon.__new__(OnePhonon)
            
            # Set necessary attributes from instance_data
            model.device = self.device
            model.hsampling = instance_data.hsampling
            model.ksampling = instance_data.ksampling
            model.lsampling = instance_data.lsampling
            
            # Set A_inv tensor with gradient tracking
            model.A_inv = torch.tensor(instance_data.model.A_inv, 
                                      device=self.device, 
                                      requires_grad=True)
            
            # Call the method
            model._build_kvec_Brillouin()
            
            # Convert results to numpy for comparison
            actual_kvec = model.kvec.cpu().detach().numpy()
            actual_kvec_norm = model.kvec_norm.cpu().detach().numpy()
            
            # Compare with expected outputs
            self.assertTrue(np.allclose(actual_kvec, expected_kvec, rtol=1e-5, atol=1e-8),
                           "kvec doesn't match ground truth")
            self.assertTrue(np.allclose(actual_kvec_norm, expected_kvec_norm, rtol=1e-5, atol=1e-8),
                           "kvec_norm doesn't match ground truth")
    
    def test_at_kvec_from_miller_points(self):
        """Test _at_kvec_from_miller_points method with ground truth data."""
        # Load log data for this method
        logs = self.logger.loadLog(f"{self.at_kvec_miller_log}.log")
        
        # Process each log entry
        for i in range(0, len(logs), 2):
            # The log format might have the function call data and result
            if 'args' in logs[i]:
                args = self.logger.serializer.deserialize(logs[i]['args'])
                instance_data = args[0]
                hkl_kvec = args[1]
                expected_output = self.logger.serializer.deserialize(logs[i+1]['result'])
                
                # Create a partially initialized OnePhonon instance
                model = OnePhonon.__new__(OnePhonon)
                
                # Set necessary attributes from instance_data
                model.device = self.device
                model.hsampling = instance_data.hsampling
                model.ksampling = instance_data.ksampling
                model.lsampling = instance_data.lsampling
                model.map_shape = instance_data.map_shape
                
                # Call the method
                actual_output = model._at_kvec_from_miller_points(hkl_kvec)
                
                # Convert to numpy for comparison
                actual_output_np = actual_output.cpu().numpy()
                
                # Compare indices
                # For indices, we want exact matching
                self.assertTrue(np.array_equal(actual_output_np, expected_output),
                               "Indices don't match ground truth")
    
    def test_gradient_flow(self):
        """Test gradient flow through k-vector operations."""
        # Enable anomaly detection to help debug gradient issues
        torch.autograd.set_detect_anomaly(True)
        
        # Test gradient flow through _build_kvec_Brillouin
        model = self._create_test_model()
        
        # Ensure A_inv requires gradients
        model.A_inv.requires_grad_(True)
        
        # Call the method
        model._build_kvec_Brillouin()
        
        # Create a scalar output dependent on the results
        # Use .clone() to avoid in-place operations that break gradient flow
        output = model.kvec.clone().sum() + model.kvec_norm.clone().sum()
        
        # Compute gradients
        output.backward()
        
        # Check that gradients flowed back to A_inv
        self.assertIsNotNone(model.A_inv.grad)
        self.assertFalse(torch.allclose(model.A_inv.grad, torch.zeros_like(model.A_inv.grad)),
                        "No gradient flow to A_inv")
        
        # Reset grads and test _at_kvec_from_miller_points
        # This is mostly for ensuring the method runs without error in backward pass
        # rather than checking specific gradient values, as it's primarily an indexing operation
        model.A_inv.grad = None
        hkl_kvec = (0, 0, 0)
        
        # Get indices
        indices = model._at_kvec_from_miller_points(hkl_kvec)
        
        # Create dummy data with enough space for all indices
        max_index = torch.max(indices).item()
        dummy_data = torch.ones((max_index + 1,), device=model.device, requires_grad=True)
        
        # Index into the dummy data using the indices
        # Use clone() to avoid in-place operations
        selected = dummy_data[indices].clone()
        
        # Compute a scalar output and gradient
        output = selected.sum()
        output.backward()
        
        # Check that gradient flowed to dummy_data
        self.assertIsNotNone(dummy_data.grad)
        # Only the indexed positions should have gradients
        indices_np = indices.cpu().numpy()
        for i in range(len(dummy_data)):
            if i in indices_np:
                self.assertEqual(dummy_data.grad[i].item(), 1.0,
                               f"Expected gradient 1.0 at index {i}, got {dummy_data.grad[i].item()}")
            else:
                self.assertEqual(dummy_data.grad[i].item(), 0.0,
                               f"Expected gradient 0.0 at index {i}, got {dummy_data.grad[i].item()}")
        
        # Disable anomaly detection after test
        torch.autograd.set_detect_anomaly(False)

if __name__ == '__main__':
    unittest.main()
