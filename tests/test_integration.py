import unittest
import numpy as np
import torch
import tempfile
import shutil
from pathlib import Path
from eryx.models import OnePhonon as NumpyOnePhonon
from eryx.models_torch import OnePhonon as TorchOnePhonon

class TestOnePhononIntegration(unittest.TestCase):
    def setUp(self):
        """Set up test environment with temporary directory."""
        # Suppress specific Gemmi warnings that don't affect test functionality
        import warnings
        warnings.filterwarnings("ignore", message="remove_ligands_and_waters.*missing entity_type.*")
        
        # Add test parameter for ignoring known warnings
        self.ignore_warnings = True
        
        # Create temporary directory for test outputs
        self.temp_dir = tempfile.mkdtemp()
        self.pdb_path = "tests/pdbs/5zck_p1.pdb"
        self.device = torch.device('cpu')  # Use CPU for consistent testing
        
        # Smaller test parameters for faster testing
        self.test_params = {
            'pdb_path': self.pdb_path,
            'hsampling': [-2, 2, 2],  # Smaller grid
            'ksampling': [-2, 2, 2],
            'lsampling': [-2, 2, 2],
            'expand_p1': True,
            'res_limit': 0.0,
            'gnm_cutoff': 4.0,
            'gamma_intra': 1.0,
            'gamma_inter': 1.0
        }
    
    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)
    
    def test_end_to_end_match(self):
        """Test that PyTorch implementation matches NumPy end-to-end."""
        # Define file paths in temp directory
        np_file = Path(self.temp_dir) / "np_result.npy"
        torch_file = Path(self.temp_dir) / "torch_result.npy"
        
        if self.ignore_warnings:
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="remove_ligands_and_waters.*")
                # Run NumPy implementation
                np_model = NumpyOnePhonon(**self.test_params)
                Id_np = np_model.apply_disorder(use_data_adp=True)
                np.save(np_file, Id_np)
                
                # Run PyTorch implementation
                torch_model = TorchOnePhonon(**self.test_params, device=self.device)
                Id_torch = torch_model.apply_disorder(use_data_adp=True)
                np.save(torch_file, Id_torch.detach().cpu().numpy())
        else:
            # Run NumPy implementation
            np_model = NumpyOnePhonon(**self.test_params)
            Id_np = np_model.apply_disorder(use_data_adp=True)
            np.save(np_file, Id_np)
            
            # Run PyTorch implementation
            torch_model = TorchOnePhonon(**self.test_params, device=self.device)
            Id_torch = torch_model.apply_disorder(use_data_adp=True)
            np.save(torch_file, Id_torch.detach().cpu().numpy())
        
        # Load and compare results
        np_result = np.load(np_file)
        torch_result = np.load(torch_file)
        
        # Get non-NaN mask
        mask = ~np.isnan(np_result) & ~np.isnan(torch_result)
        self.assertTrue(np.any(mask), "All values are NaN")
        
        # Calculate metrics
        mse = np.mean((np_result[mask] - torch_result[mask])**2)
        correlation = np.corrcoef(np_result[mask], torch_result[mask])[0, 1]
        max_diff = np.max(np.abs(np_result[mask] - torch_result[mask]))
        
        # Log detailed comparison info for debugging
        print(f"MSE: {mse}")
        print(f"Correlation: {correlation}")
        print(f"Max difference: {max_diff}")
        print(f"NumPy min/max: {np.min(np_result[mask])}/{np.max(np_result[mask])}")
        print(f"PyTorch min/max: {np.min(torch_result[mask])}/{np.max(torch_result[mask])}")
        
        # Use more appropriate tolerances for floating point calculations
        # MSE should be proportional to the magnitude of the values
        max_magnitude = max(np.max(np.abs(np_result[mask])), np.max(np.abs(torch_result[mask])))
        relative_mse = mse / (max_magnitude**2) if max_magnitude > 0 else mse
        
        # Verify results are close enough with relative tolerances
        self.assertLess(relative_mse, 1e-4, f"Relative MSE too high: {relative_mse}")
        self.assertGreater(correlation, 0.99, f"Correlation too low: {correlation}")
        self.assertLess(max_diff / max_magnitude if max_magnitude > 0 else max_diff, 
                      1e-2, f"Relative max difference too high: {max_diff / max_magnitude}")
    
    def test_parameter_gradients(self):
        """Test gradient flow through model parameters."""
        if self.ignore_warnings:
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="remove_ligands_and_waters.*")
                # Create model with parameters that require gradients
                gamma_intra = torch.tensor(1.0, requires_grad=True)
                gamma_inter = torch.tensor(1.0, requires_grad=True)
                
                # Create OnePhonon model with these parameters
                torch_model = TorchOnePhonon(
                    pdb_path=self.pdb_path,
                    hsampling=self.test_params['hsampling'],
                    ksampling=self.test_params['ksampling'],
                    lsampling=self.test_params['lsampling'],
                    expand_p1=self.test_params['expand_p1'],
                    res_limit=self.test_params['res_limit'],
                    gnm_cutoff=self.test_params['gnm_cutoff'],
                    gamma_intra=gamma_intra,
                    gamma_inter=gamma_inter,
                    device=self.device
                )
        else:
            # Create model with parameters that require gradients
            gamma_intra = torch.tensor(1.0, requires_grad=True)
            gamma_inter = torch.tensor(1.0, requires_grad=True)
            
            # Create OnePhonon model with these parameters
            torch_model = TorchOnePhonon(
                pdb_path=self.pdb_path,
                hsampling=self.test_params['hsampling'],
                ksampling=self.test_params['ksampling'],
                lsampling=self.test_params['lsampling'],
                expand_p1=self.test_params['expand_p1'],
                res_limit=self.test_params['res_limit'],
                gnm_cutoff=self.test_params['gnm_cutoff'],
                gamma_intra=gamma_intra,
                gamma_inter=gamma_inter,
                device=self.device
            )
        
        # Forward pass
        Id_torch = torch_model.apply_disorder(use_data_adp=True)
        
        # Create a masked loss that properly handles NaNs
        mask = ~torch.isnan(Id_torch)
        # Ensure we're working with real tensors for the loss
        valid_intensity = torch.where(mask, Id_torch, torch.zeros_like(Id_torch))
        # Convert to float32 to ensure consistent gradient flow
        valid_intensity = valid_intensity.to(dtype=torch.float32)
        loss = torch.sum(valid_intensity)  # Sum of valid intensities
        
        # Backward pass
        loss.backward()
        
        # Verify gradients exist and are non-zero
        self.assertIsNotNone(gamma_intra.grad, "No gradient for gamma_intra")
        self.assertIsNotNone(gamma_inter.grad, "No gradient for gamma_inter")
        
        # Print gradient values for debugging
        print(f"gamma_intra gradient: {gamma_intra.grad.item()}")
        print(f"gamma_inter gradient: {gamma_inter.grad.item()}")
        
        # Test gradient magnitude relative to parameter magnitude
        gamma_intra_rel_grad = torch.norm(gamma_intra.grad) / torch.norm(gamma_intra)
        gamma_inter_rel_grad = torch.norm(gamma_inter.grad) / torch.norm(gamma_inter)
        
        print(f"Relative gradient magnitudes - gamma_intra: {gamma_intra_rel_grad.item()}, gamma_inter: {gamma_inter_rel_grad.item()}")
        
        # Check that gradients are non-zero
        self.assertGreater(torch.norm(gamma_intra.grad).item(), 1e-10, 
                          "Gradient for gamma_intra too small")
        self.assertGreater(torch.norm(gamma_inter.grad).item(), 1e-10,
                          "Gradient for gamma_inter too small")
    
    def test_device_compatibility(self):
        """Test model works on different devices if available."""
        if self.ignore_warnings:
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="remove_ligands_and_waters.*")
                # Test on CPU
                try:
                    model_cpu = TorchOnePhonon(**self.test_params, device=torch.device('cpu'))
                    result_cpu = model_cpu.apply_disorder(use_data_adp=True)
                    self.assertIsNotNone(result_cpu, "CPU execution failed")
                    
                    # Test on CUDA if available
                    if torch.cuda.is_available():
                        model_cuda = TorchOnePhonon(**self.test_params, device=torch.device('cuda'))
                        result_cuda = model_cuda.apply_disorder(use_data_adp=True)
                        self.assertIsNotNone(result_cuda, "CUDA execution failed")
                        
                        # Verify results match between devices
                        cpu_array = result_cpu.detach().cpu().numpy()
                        cuda_array = result_cuda.detach().cpu().numpy()
                        
                        mask = ~np.isnan(cpu_array) & ~np.isnan(cuda_array)
                        self.assertTrue(np.any(mask), "All values are NaN")
                        
                        correlation = np.corrcoef(cpu_array[mask], cuda_array[mask])[0, 1]
                        self.assertGreater(correlation, 0.99, f"CPU/CUDA correlation too low: {correlation}")
                except RuntimeError as e:
                    if "CUDA" in str(e) and not torch.cuda.is_available():
                        self.skipTest("CUDA test skipped - not available")
                    else:
                        raise
        else:
            # Test on CPU
            try:
                model_cpu = TorchOnePhonon(**self.test_params, device=torch.device('cpu'))
                result_cpu = model_cpu.apply_disorder(use_data_adp=True)
                self.assertIsNotNone(result_cpu, "CPU execution failed")
                
                # Test on CUDA if available
                if torch.cuda.is_available():
                    model_cuda = TorchOnePhonon(**self.test_params, device=torch.device('cuda'))
                    result_cuda = model_cuda.apply_disorder(use_data_adp=True)
                    self.assertIsNotNone(result_cuda, "CUDA execution failed")
                
                # Verify results match between devices
                cpu_array = result_cpu.detach().cpu().numpy()
                cuda_array = result_cuda.detach().cpu().numpy()
                
                mask = ~np.isnan(cpu_array) & ~np.isnan(cuda_array)
                self.assertTrue(np.any(mask), "All values are NaN")
                
                correlation = np.corrcoef(cpu_array[mask], cuda_array[mask])[0, 1]
                self.assertGreater(correlation, 0.99, f"CPU/CUDA correlation too low: {correlation}")
            except RuntimeError as e:
                if "CUDA" in str(e) and not torch.cuda.is_available():
                    self.skipTest("CUDA test skipped - not available")
                else:
                    raise
    
    def test_crystal_adapter(self):
        """Test that the Crystal adapter properly converts method return values to tensors."""
        if self.ignore_warnings:
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="remove_ligands_and_waters.*")
                # Create model with basic parameters
                model = TorchOnePhonon(**self.test_params, device=self.device)
                
                # Test key crystal methods return tensors
                # Test get_unitcell_origin returns a tensor
                unit_cell = [0, 0, 0]
                origin = model.crystal.get_unitcell_origin(unit_cell)
                self.assertIsInstance(origin, torch.Tensor)
                self.assertEqual(origin.device, self.device)
                
                # Test get_asu_xyz returns a tensor
                xyz = model.crystal.get_asu_xyz(0, unit_cell)
                self.assertIsInstance(xyz, torch.Tensor)
                self.assertEqual(xyz.device, self.device)
                
                # Test that end-to-end operations work with the dictionary pattern
                # Apply a basic operation that uses crystal methods
                Id = model.apply_disorder(use_data_adp=True)
                self.assertIsInstance(Id, torch.Tensor)
                
                # Verify tensor has gradients enabled
                loss = torch.sum(torch.where(torch.isnan(Id), torch.tensor(0.0, device=self.device), Id))
                loss.backward()
        else:
            # Same as above without warning suppression
            model = TorchOnePhonon(**self.test_params, device=self.device)
            
            # Test get_unitcell_origin returns a tensor
            unit_cell = [0, 0, 0]
            origin = model.crystal.get_unitcell_origin(unit_cell)
            self.assertIsInstance(origin, torch.Tensor)
            self.assertEqual(origin.device, self.device)
            
            # Test get_asu_xyz returns a tensor
            xyz = model.crystal.get_asu_xyz(0, unit_cell)
            self.assertIsInstance(xyz, torch.Tensor)
            self.assertEqual(xyz.device, self.device)
            
            # Test that end-to-end operations work with the dictionary pattern
            # Apply a basic operation that uses crystal methods
            Id = model.apply_disorder(use_data_adp=True)
            self.assertIsInstance(Id, torch.Tensor)
            
            # Verify tensor has gradients enabled
            mask = ~torch.isnan(Id)
            valid_intensity = torch.where(mask, Id, torch.zeros_like(Id))
            loss = torch.sum(valid_intensity)
            loss.backward()

if __name__ == '__main__':
    unittest.main()
