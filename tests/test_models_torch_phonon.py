import unittest
import os
import torch
import numpy as np
from eryx.models_torch import OnePhonon
from eryx.autotest.torch_testing import TorchTesting
from eryx.autotest.logger import Logger
from eryx.autotest.functionmapping import FunctionMapping
from unittest.mock import patch, MagicMock

class TestOnePhononPhonon(unittest.TestCase):
    def setUp(self):
        # Set up the testing framework
        self.logger = Logger()
        self.function_mapping = FunctionMapping()
        self.torch_testing = TorchTesting(self.logger, self.function_mapping, rtol=1e-4, atol=1e-6)
        
        # Set device to CPU for consistent testing
        self.device = torch.device('cpu')
        
        # Log file prefixes for ground truth data
        self.compute_gnm_phonons_log = "logs/eryx.models.compute_gnm_phonons"
        self.compute_hessian_log = "logs/eryx.models.compute_hessian"
        self.gnm_compute_hessian_log = "logs/eryx.pdb.compute_hessian"
        self.gnm_compute_K_log = "logs/eryx.pdb.compute_K"
        self.gnm_compute_Kinv_log = "logs/eryx.pdb.compute_Kinv"
        
        # Ensure log files exist
        for log_file in [self.compute_gnm_phonons_log, self.compute_hessian_log, 
                         self.gnm_compute_hessian_log, self.gnm_compute_K_log, 
                         self.gnm_compute_Kinv_log]:
            log_file_path = f"{log_file}.log"
            self.assertTrue(os.path.exists(log_file_path), f"Log file {log_file_path} not found")
        
        # Create minimal test model
        self.model = self._create_test_model()
    
    def _create_test_model(self):
        """Create a minimal test model with necessary attributes for phonon methods."""
        model = OnePhonon.__new__(OnePhonon)  # Create instance without calling __init__
        
        # Set necessary attributes
        model.device = self.device
        model.n_asu = 2
        model.n_cell = 3
        model.n_atoms_per_asu = 4
        model.n_dof_per_asu = 6
        model.n_dof_per_asu_actual = 12  # n_atoms_per_asu * 3
        model.n_dof_per_cell = 12  # n_asu * n_dof_per_asu
        model.id_cell_ref = 0
        
        # Create other necessary attributes
        model.hsampling = (0, 5, 2)
        model.ksampling = (0, 5, 2)
        model.lsampling = (0, 5, 2)
        
        # Create sample matrices
        model.Amat = torch.zeros((model.n_asu, model.n_dof_per_asu_actual, model.n_dof_per_asu), 
                                device=self.device)
        model.Linv = torch.eye(model.n_dof_per_cell, device=self.device)
        
        # Create k-vectors for testing
        model.kvec = torch.zeros((model.hsampling[2], model.ksampling[2], model.lsampling[2], 3), 
                                device=self.device)
        model.kvec_norm = torch.zeros((model.hsampling[2], model.ksampling[2], model.lsampling[2], 1), 
                                     device=self.device)
        
        # Sample methods that would be called
        model.id_to_hkl = lambda cell_id: [0, 0, 0]
        model.get_unitcell_origin = lambda unit_cell: torch.tensor([0.0, 0.0, 0.0], device=self.device)
        
        return model
    
    def test_compute_gnm_hessian(self):
        """Test compute_gnm_hessian method (former GaussianNetworkModel.compute_hessian)."""
        # Load log data for this method
        logs = self.logger.loadLog(f"{self.gnm_compute_hessian_log}.log")
        
        # Process each input/output pair from logs
        for i in range(0, len(logs), 2):
            if 'args' in logs[i]:
                # Get input data and expected output
                args = self.logger.serializer.deserialize(logs[i]['args'])
                instance_data = args[0]
                expected_output = self.logger.serializer.deserialize(logs[i+1]['result'])
                
                # Create a partially initialized OnePhonon instance
                model = OnePhonon.__new__(OnePhonon)
                model.device = self.device
                
                # Set necessary attributes from instance_data
                model.n_asu = instance_data.n_asu
                model.n_atoms_per_asu = instance_data.n_atoms_per_asu
                model.n_cell = instance_data.n_cell
                model.id_cell_ref = instance_data.id_cell_ref
                
                # Mock the asu_neighbors attribute to return neighbors from instance_data
                model._get_atom_neighbors = lambda i_asu, i_cell, j_asu, i_at: instance_data.asu_neighbors[i_asu][i_cell][j_asu][i_at]
                
                # Mock the gamma values to return from instance_data
                model._get_gamma = lambda i_asu, i_cell, j_asu: torch.tensor(instance_data.gamma[i_cell, i_asu, j_asu], 
                                                                            device=self.device, 
                                                                            dtype=torch.complex64)
                
                # Call the method
                result = model.compute_gnm_hessian()
                
                # Convert result to numpy for comparison
                result_np = result.cpu().detach().numpy()
                
                # Compare with expected output
                self.assertTrue(np.allclose(result_np, expected_output, rtol=1e-5, atol=1e-8),
                              "compute_gnm_hessian doesn't match ground truth")
                break  # Test first example for now
    
    def test_compute_gnm_K(self):
        """Test compute_gnm_K method (former GaussianNetworkModel.compute_K)."""
        # Load log data for this method
        logs = self.logger.loadLog(f"{self.gnm_compute_K_log}.log")
        
        # Process each input/output pair from logs
        for i in range(0, len(logs), 2):
            if 'args' in logs[i]:
                # Get input data and expected output
                args = self.logger.serializer.deserialize(logs[i]['args'])
                instance_data = args[0]  # GNM instance
                hessian_data = args[1]   # Hessian matrix
                kvec_data = args[2]      # k-vector
                expected_output = self.logger.serializer.deserialize(logs[i+1]['result'])
                
                # Create a partially initialized OnePhonon instance
                model = OnePhonon.__new__(OnePhonon)
                model.device = self.device
                
                # Set necessary attributes
                model.n_asu = instance_data.n_asu
                model.n_cell = instance_data.n_cell
                model.id_cell_ref = instance_data.id_cell_ref
                
                # Mock the id_to_hkl and get_unitcell_origin methods
                model.id_to_hkl = lambda cell_id: instance_data.crystal.id_to_hkl(cell_id)
                model.get_unitcell_origin = lambda unit_cell: torch.tensor(
                    instance_data.crystal.get_unitcell_origin(unit_cell), 
                    device=self.device)
                
                # Convert numpy arrays to PyTorch tensors
                hessian = torch.tensor(hessian_data, device=self.device)
                kvec = torch.tensor(kvec_data, device=self.device)
                
                # Call the method
                result = model.compute_gnm_K(hessian, kvec)
                
                # Convert result to numpy for comparison
                result_np = result.cpu().detach().numpy()
                
                # Compare with expected output
                self.assertTrue(np.allclose(result_np, expected_output, rtol=1e-5, atol=1e-8),
                              "compute_gnm_K doesn't match ground truth")
                break  # Test first example for now
    
    def test_compute_gnm_Kinv(self):
        """Test compute_gnm_Kinv method (former GaussianNetworkModel.compute_Kinv)."""
        # Load log data for this method
        logs = self.logger.loadLog(f"{self.gnm_compute_Kinv_log}.log")
        
        # Process each input/output pair from logs
        for i in range(0, len(logs), 2):
            if 'args' in logs[i]:
                # Get input data and expected output
                args = self.logger.serializer.deserialize(logs[i]['args'])
                instance_data = args[0]  # GNM instance
                hessian_data = args[1]   # Hessian matrix
                kvec_data = args[2] if len(args) > 2 else None  # k-vector
                reshape_flag = args[3] if len(args) > 3 else True  # reshape flag
                expected_output = self.logger.serializer.deserialize(logs[i+1]['result'])
                
                # Create a partially initialized OnePhonon instance
                model = OnePhonon.__new__(OnePhonon)
                model.device = self.device
                
                # Set necessary attributes
                model.n_asu = instance_data.n_asu
                model.n_cell = instance_data.n_cell
                model.id_cell_ref = instance_data.id_cell_ref
                
                # Mock the methods needed by compute_gnm_K
                model.id_to_hkl = lambda cell_id: instance_data.crystal.id_to_hkl(cell_id)
                model.get_unitcell_origin = lambda unit_cell: torch.tensor(
                    instance_data.crystal.get_unitcell_origin(unit_cell), 
                    device=self.device)
                
                # Mock compute_gnm_K method to avoid duplicate testing
                original_compute_gnm_K = model.compute_gnm_K
                model.compute_gnm_K = lambda hessian, kvec=None: torch.tensor(
                    instance_data.compute_K(hessian_data, kvec_data), 
                    device=self.device)
                
                # Convert numpy arrays to PyTorch tensors
                hessian = torch.tensor(hessian_data, device=self.device)
                kvec = torch.tensor(kvec_data, device=self.device) if kvec_data is not None else None
                
                # Call the method
                result = model.compute_gnm_Kinv(hessian, kvec, reshape_flag)
                
                # Restore original method
                model.compute_gnm_K = original_compute_gnm_K
                
                # Convert result to numpy for comparison
                result_np = result.cpu().detach().numpy()
                
                # Compare with expected output
                self.assertTrue(np.allclose(result_np, expected_output, rtol=1e-5, atol=1e-8),
                              "compute_gnm_Kinv doesn't match ground truth")
                break  # Test first example for now
    
    def test_compute_hessian(self):
        """Test compute_hessian method."""
        # Load log data for this method
        logs = self.logger.loadLog(f"{self.compute_hessian_log}.log")
        
        # Process each input/output pair from logs
        for i in range(0, len(logs), 2):
            if 'args' in logs[i]:
                # Get input data and expected output
                args = self.logger.serializer.deserialize(logs[i]['args'])
                instance_data = args[0]  # OnePhonon instance
                expected_output = self.logger.serializer.deserialize(logs[i+1]['result'])
                
                # Create a partially initialized OnePhonon instance
                model = OnePhonon.__new__(OnePhonon)
                model.device = self.device
                
                # Set necessary attributes
                model.n_asu = instance_data.n_asu
                model.n_atoms_per_asu = instance_data.n_atoms_per_asu
                model.n_dof_per_asu = instance_data.n_dof_per_asu
                model.n_dof_per_asu_actual = instance_data.n_dof_per_asu_actual
                model.n_cell = instance_data.n_cell
                model.id_cell_ref = instance_data.id_cell_ref
                
                # Set Amat tensor
                model.Amat = torch.tensor(instance_data.Amat, device=self.device)
                
                # Mock compute_gnm_hessian method
                model.compute_gnm_hessian = lambda: torch.tensor(
                    instance_data.gnm.compute_hessian(), 
                    device=self.device)
                
                # Call the method
                result = model.compute_hessian()
                
                # Convert result to numpy for comparison
                result_np = result.cpu().detach().numpy()
                
                # Compare with expected output
                self.assertTrue(np.allclose(result_np, expected_output, rtol=1e-5, atol=1e-8),
                              "compute_hessian doesn't match ground truth")
                break  # Test first example for now
    
    def test_compute_gnm_phonons(self):
        """Test compute_gnm_phonons method."""
        # Load log data for this method
        logs = self.logger.loadLog(f"{self.compute_gnm_phonons_log}.log")
        
        # Process each input/output pair from logs
        for i in range(0, len(logs), 2):
            if 'args' in logs[i]:
                # Get input data and expected output
                args = self.logger.serializer.deserialize(logs[i]['args'])
                instance_data = args[0]  # OnePhonon instance
                expected_output = self.logger.serializer.deserialize(logs[i+1]['result'])
                
                # Expected V and Winv
                expected_V = expected_output[0]
                expected_Winv = expected_output[1]
                
                # Create a partially initialized OnePhonon instance
                model = OnePhonon.__new__(OnePhonon)
                model.device = self.device
                
                # Set necessary attributes
                model.n_asu = instance_data.n_asu
                model.n_dof_per_asu = instance_data.n_dof_per_asu
                model.hsampling = instance_data.hsampling
                model.ksampling = instance_data.ksampling
                model.lsampling = instance_data.lsampling
                
                # Set k-vectors
                model.kvec = torch.tensor(instance_data.kvec, device=self.device)
                
                # Set Linv tensor
                model.Linv = torch.tensor(instance_data.Linv, device=self.device)
                
                # Mock compute_hessian method
                model.compute_hessian = lambda: torch.tensor(
                    instance_data.compute_hessian(), 
                    device=self.device)
                
                # Mock compute_gnm_K method
                original_compute_gnm_K = model.compute_gnm_K
                model.compute_gnm_K = lambda hessian, kvec=None: torch.tensor(
                    instance_data.gnm.compute_K(hessian.cpu().numpy(), 
                                              kvec.cpu().numpy() if kvec is not None else None), 
                    device=self.device)
                
                # Call the method
                model.compute_gnm_phonons()
                
                # Restore original method
                model.compute_gnm_K = original_compute_gnm_K
                
                # Convert results to numpy for comparison
                V_np = model.V.cpu().detach().numpy()
                Winv_np = model.Winv.cpu().detach().numpy()
                
                # Compare with expected outputs - use lower tolerance for eigendecomposition
                rtol = 1e-4
                atol = 1e-6
                
                # Check shapes first
                self.assertEqual(V_np.shape, expected_V.shape,
                               f"V shape mismatch: {V_np.shape} vs {expected_V.shape}")
                self.assertEqual(Winv_np.shape, expected_Winv.shape,
                               f"Winv shape mismatch: {Winv_np.shape} vs {expected_Winv.shape}")
                
                # Check values
                # Note: eigenvectors may differ by a phase factor, so check their absolute values
                V_match = np.allclose(np.abs(V_np), np.abs(expected_V), rtol=rtol, atol=atol)
                Winv_match = np.allclose(Winv_np, expected_Winv, rtol=rtol, atol=atol, 
                                       equal_nan=True)  # Handle NaN values
                
                self.assertTrue(V_match, "Eigenvectors V don't match ground truth")
                self.assertTrue(Winv_match, "Eigenvalues Winv don't match ground truth")
                break  # Test first example for now
    
    def test_gradient_flow(self):
        """Test gradient flow through phonon calculation methods."""
        # Enable anomaly detection to help debug gradient issues
        torch.autograd.set_detect_anomaly(True)
        
        # Create test model
        model = self._create_test_model()
        
        # Test gradient flow through compute_gnm_K
        # Create a simple hessian matrix with gradient tracking
        hessian = torch.ones((model.n_asu, model.n_atoms_per_asu,
                             model.n_cell, model.n_asu, model.n_atoms_per_asu),
                            dtype=torch.complex64, device=self.device, requires_grad=True)
        
        # Create a simple k-vector with gradient tracking
        kvec = torch.ones(3, device=self.device, requires_grad=True)
        
        # Mock required methods
        model.id_to_hkl = lambda cell_id: [cell_id, 0, 0]
        model.get_unitcell_origin = lambda unit_cell: torch.tensor(
            [float(unit_cell[0]), 0.0, 0.0], device=self.device, requires_grad=True)
        
        # Call compute_gnm_K
        Kmat = model.compute_gnm_K(hessian, kvec)
        
        # Create a loss function
        loss = torch.abs(Kmat).sum()
        
        # Compute gradients
        loss.backward()
        
        # Check that gradients flowed back to inputs
        self.assertIsNotNone(hessian.grad)
        self.assertIsNotNone(kvec.grad)
        self.assertFalse(torch.allclose(hessian.grad, torch.zeros_like(hessian.grad)),
                        "No gradient flow to hessian in compute_gnm_K")
        self.assertFalse(torch.allclose(kvec.grad, torch.zeros_like(kvec.grad)),
                        "No gradient flow to kvec in compute_gnm_K")
        
        # Reset gradients
        hessian.grad = None
        kvec.grad = None
        
        # Test gradient flow through compute_gnm_Kinv
        Kinv = model.compute_gnm_Kinv(hessian, kvec)
        
        # Create a loss function
        loss = torch.abs(Kinv).sum()
        
        # Compute gradients
        loss.backward()
        
        # Check that gradients flowed back to inputs
        self.assertIsNotNone(hessian.grad)
        self.assertIsNotNone(kvec.grad)
        self.assertFalse(torch.allclose(hessian.grad, torch.zeros_like(hessian.grad)),
                        "No gradient flow to hessian in compute_gnm_Kinv")
        self.assertFalse(torch.allclose(kvec.grad, torch.zeros_like(kvec.grad)),
                        "No gradient flow to kvec in compute_gnm_Kinv")
        
        # Disable anomaly detection after test
        torch.autograd.set_detect_anomaly(False)

if __name__ == '__main__':
    unittest.main()
