"""
Tests for the adapter components that bridge NumPy and PyTorch implementations.

This module contains tests for the adapter classes in eryx.adapters:
- PDBToTensor: Converts atomic model data to PyTorch tensors
- GridToTensor: Converts reciprocal space grids to PyTorch tensors
- TensorToNumpy: Converts PyTorch tensors back to NumPy arrays
- ModelAdapters: Provides higher-level integration between adapter components
"""

import unittest
import numpy as np
import torch
from typing import Dict, Any, Tuple, Optional

from eryx.adapters import PDBToTensor, GridToTensor, TensorToNumpy, ModelAdapters

class TestPDBToTensor(unittest.TestCase):
    """Tests for the PDBToTensor adapter class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.adapter = PDBToTensor()
        # Create a device that's guaranteed to work for testing
        self.device = torch.device('cpu')
        self.adapter_with_device = PDBToTensor(device=self.device)
        
    def test_state_dict_conversion(self):
        """Test full state dictionary conversion."""
        # Create a complex nested state dictionary similar to what Logger.loadStateLog returns
        nested_state = {
            'simple_array': np.random.rand(10, 3),
            'nested': {
                'array1': np.random.rand(5, 2),
                'array2': np.random.rand(3, 4)
            },
            'scalar': 42.0,
            'string': 'test',
            'array_list': [np.random.rand(2, 2), np.random.rand(3, 3)],
            'complex_value': np.complex128(1+2j),
            'boolean_array': np.array([True, False, True])
        }
        
        # Convert to tensors
        tensor_state = self.adapter.convert_state_dict(nested_state)
        
        # Verify structure preservation
        self.assertIsInstance(tensor_state['simple_array'], torch.Tensor)
        self.assertIsInstance(tensor_state['nested'], dict)
        self.assertIsInstance(tensor_state['nested']['array1'], torch.Tensor)
        self.assertIsInstance(tensor_state['array_list'], list)
        self.assertIsInstance(tensor_state['array_list'][0], torch.Tensor)
        self.assertEqual(tensor_state['scalar'], 42.0)
        self.assertEqual(tensor_state['string'], 'test')
        
        # Verify special type handling
        self.assertTrue(torch.is_complex(tensor_state['complex_value']))
        self.assertEqual(tensor_state['boolean_array'].dtype, torch.bool)
        
        # Convert back to NumPy
        numpy_adapter = TensorToNumpy()
        numpy_state = numpy_adapter.convert_state_to_numpy(tensor_state)
        
        # Verify round-trip conversion
        self.assertIsInstance(numpy_state['simple_array'], np.ndarray)
        self.assertTrue(np.allclose(numpy_state['simple_array'], nested_state['simple_array']))
        self.assertTrue(np.allclose(numpy_state['nested']['array1'], nested_state['nested']['array1']))
        self.assertTrue(np.allclose(numpy_state['array_list'][0], nested_state['array_list'][0]))
        self.assertEqual(numpy_state['scalar'], 42.0)
        self.assertEqual(numpy_state['string'], 'test')
        self.assertEqual(numpy_state['complex_value'], nested_state['complex_value'])
        self.assertTrue(np.array_equal(numpy_state['boolean_array'], nested_state['boolean_array']))
    
    def test_array_to_tensor_conversion(self):
        """Test basic NumPy array to tensor conversion functionality."""
        # Test with a simple 2D array
        array_2d = np.array([[1.0, 2.0], [3.0, 4.0]])
        tensor_2d = self.adapter.array_to_tensor(array_2d)
        
        self.assertIsInstance(tensor_2d, torch.Tensor)
        self.assertEqual(tensor_2d.shape, torch.Size([2, 2]))
        self.assertTrue(tensor_2d.requires_grad)
        self.assertTrue(torch.allclose(tensor_2d.cpu(), torch.tensor(array_2d, dtype=torch.float32)))
        
        # Test with a 3D array
        array_3d = np.random.rand(2, 3, 4)
        tensor_3d = self.adapter.array_to_tensor(array_3d)
        
        self.assertIsInstance(tensor_3d, torch.Tensor)
        self.assertEqual(tensor_3d.shape, torch.Size([2, 3, 4]))
        self.assertTrue(tensor_3d.requires_grad)
        self.assertTrue(torch.allclose(tensor_3d.cpu(), torch.tensor(array_3d, dtype=torch.float32)))
        
        # Test with requires_grad=False
        tensor_no_grad = self.adapter.array_to_tensor(array_2d, requires_grad=False)
        self.assertFalse(tensor_no_grad.requires_grad)
        
        # Test with None input
        self.assertIsNone(self.adapter.array_to_tensor(None))
        
        # Test with empty array
        empty_array = np.array([])
        tensor_empty = self.adapter.array_to_tensor(empty_array)
        self.assertEqual(tensor_empty.numel(), 0)
        
        # Test with specified device
        tensor_with_device = self.adapter_with_device.array_to_tensor(array_2d)
        self.assertEqual(tensor_with_device.device, self.device)
    
    def test_pdb_to_tensor_atomic_model(self):
        """Test conversion of AtomicModel to tensor dictionary."""
        # Create a dummy AtomicModel as a dictionary with minimum required attributes
        dummy_model = type('DummyAtomicModel', (), {
            'xyz': np.random.rand(1, 10, 3),
            'ff_a': np.random.rand(1, 10, 4),
            'ff_b': np.random.rand(1, 10, 4),
            'ff_c': np.random.rand(1, 10),
            'adp': np.random.rand(1, 10),
            'cell': np.array([10.0, 10.0, 10.0, 90.0, 90.0, 90.0]),
            'A_inv': np.random.rand(3, 3),
            'unit_cell_axes': np.random.rand(3, 3),
            'space_group': 'P1',
            'n_asu': 1,
            'n_conf': 1,
            'sym_ops': {0: np.eye(3)},
            'transformations': {0: np.eye(4)},
            'elements': [['C'] * 10]
        })
        
        # Convert the dummy model
        tensor_dict = self.adapter.convert_atomic_model(dummy_model)
        
        # Verify tensor conversion of numeric attributes
        self.assertIsInstance(tensor_dict['xyz'], torch.Tensor)
        self.assertIsInstance(tensor_dict['ff_a'], torch.Tensor)
        self.assertIsInstance(tensor_dict['ff_b'], torch.Tensor)
        self.assertIsInstance(tensor_dict['ff_c'], torch.Tensor)
        self.assertIsInstance(tensor_dict['adp'], torch.Tensor)
        self.assertIsInstance(tensor_dict['cell'], torch.Tensor)
        self.assertIsInstance(tensor_dict['A_inv'], torch.Tensor)
        self.assertIsInstance(tensor_dict['unit_cell_axes'], torch.Tensor)
        
        # Verify shapes are preserved
        self.assertEqual(tensor_dict['xyz'].shape, torch.Size([1, 10, 3]))
        self.assertEqual(tensor_dict['ff_a'].shape, torch.Size([1, 10, 4]))
        self.assertEqual(tensor_dict['ff_b'].shape, torch.Size([1, 10, 4]))
        self.assertEqual(tensor_dict['ff_c'].shape, torch.Size([1, 10]))
        self.assertEqual(tensor_dict['adp'].shape, torch.Size([1, 10]))
        self.assertEqual(tensor_dict['cell'].shape, torch.Size([6]))
        self.assertEqual(tensor_dict['A_inv'].shape, torch.Size([3, 3]))
        self.assertEqual(tensor_dict['unit_cell_axes'].shape, torch.Size([3, 3]))
        
        # Verify non-tensor attributes are preserved as-is
        self.assertEqual(tensor_dict['space_group'], 'P1')
        self.assertEqual(tensor_dict['n_asu'], 1)
        self.assertEqual(tensor_dict['n_conf'], 1)
        self.assertEqual(tensor_dict['sym_ops'], dummy_model.sym_ops)
        self.assertEqual(tensor_dict['transformations'], dummy_model.transformations)
        self.assertEqual(tensor_dict['elements'], dummy_model.elements)
        
        # Test with None model
        with self.assertRaises(ValueError):
            self.adapter.convert_atomic_model(None)

class TestGridToTensor(unittest.TestCase):
    """Tests for the GridToTensor adapter class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.adapter = GridToTensor()
        # Create a device that's guaranteed to work for testing
        self.device = torch.device('cpu')
        self.adapter_with_device = GridToTensor(device=self.device)
    
    def test_grid_to_tensor_conversion(self):
        """Test conversion of grid data to tensors."""
        # Create sample q_grid and map_shape
        q_grid = np.random.rand(100, 3)
        map_shape = (10, 10, 10)
        
        # Convert using convert_grid
        q_grid_tensor, returned_map_shape = self.adapter.convert_grid(q_grid, map_shape)
        
        # Verify tensor properties
        self.assertIsInstance(q_grid_tensor, torch.Tensor)
        self.assertEqual(q_grid_tensor.shape, torch.Size([100, 3]))
        self.assertTrue(q_grid_tensor.requires_grad)
        self.assertTrue(torch.allclose(q_grid_tensor.cpu().detach(), torch.tensor(q_grid, dtype=torch.float32)))
        
        # Verify map_shape is unchanged
        self.assertEqual(returned_map_shape, map_shape)
        
        # Test with requires_grad=False
        q_grid_tensor, _ = self.adapter.convert_grid(q_grid, map_shape, requires_grad=False)
        self.assertFalse(q_grid_tensor.requires_grad)
        
        # Test with different map shapes
        map_shape_2 = (5, 5, 5)
        _, returned_map_shape_2 = self.adapter.convert_grid(q_grid, map_shape_2)
        self.assertEqual(returned_map_shape_2, map_shape_2)
        
        # Test with specified device
        q_grid_tensor, _ = self.adapter_with_device.convert_grid(q_grid, map_shape)
        self.assertEqual(q_grid_tensor.device, self.device)
        
        # Test with None q_grid
        with self.assertRaises(ValueError):
            self.adapter.convert_grid(None, map_shape)
        
        # Test with empty q_grid
        empty_grid = np.zeros((0, 3))
        q_grid_tensor, _ = self.adapter.convert_grid(empty_grid, map_shape)
        self.assertEqual(q_grid_tensor.shape, torch.Size([0, 3]))
    
    def test_convert_mask(self):
        """Test conversion of boolean mask to tensor."""
        # Create boolean mask
        mask = np.random.choice([True, False], size=(10, 10))
        
        # Convert using convert_mask
        mask_tensor = self.adapter.convert_mask(mask)
        
        # Verify tensor properties
        self.assertIsInstance(mask_tensor, torch.Tensor)
        self.assertEqual(mask_tensor.shape, torch.Size([10, 10]))
        self.assertEqual(mask_tensor.dtype, torch.bool)
        self.assertFalse(mask_tensor.requires_grad)
        
        # Compare values
        mask_np = mask_tensor.cpu().numpy()
        self.assertTrue(np.array_equal(mask_np, mask))
        
        # Test with requires_grad=True (unusual but supported)
        mask_tensor = self.adapter.convert_mask(mask, requires_grad=True)
        self.assertTrue(mask_tensor.requires_grad)
        
        # Test with None mask
        self.assertIsNone(self.adapter.convert_mask(None))
        
        # Test with empty mask
        empty_mask = np.array([], dtype=bool)
        mask_tensor = self.adapter.convert_mask(empty_mask)
        self.assertEqual(mask_tensor.numel(), 0)
        self.assertEqual(mask_tensor.dtype, torch.bool)

class TestTensorToNumpy(unittest.TestCase):
    """Tests for the TensorToNumpy adapter class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.adapter = TensorToNumpy()
        
    def test_state_dict_conversion(self):
        """Test full state dictionary conversion."""
        # Create a complex nested state dictionary with tensors
        nested_state = {
            'simple_tensor': torch.rand(10, 3),
            'nested': {
                'tensor1': torch.rand(5, 2),
                'tensor2': torch.rand(3, 4)
            },
            'scalar': 42.0,
            'string': 'test',
            'tensor_list': [torch.rand(2, 2), torch.rand(3, 3)],
            'complex_value': torch.complex(torch.tensor(1.0), torch.tensor(2.0)),
            'boolean_tensor': torch.tensor([True, False, True])
        }
        
        # Convert to NumPy
        numpy_state = self.adapter.convert_state_to_numpy(nested_state)
        
        # Verify structure preservation
        self.assertIsInstance(numpy_state['simple_tensor'], np.ndarray)
        self.assertIsInstance(numpy_state['nested'], dict)
        self.assertIsInstance(numpy_state['nested']['tensor1'], np.ndarray)
        self.assertIsInstance(numpy_state['tensor_list'], list)
        self.assertIsInstance(numpy_state['tensor_list'][0], np.ndarray)
        self.assertEqual(numpy_state['scalar'], 42.0)
        self.assertEqual(numpy_state['string'], 'test')
        
        # Verify special type handling
        self.assertTrue(isinstance(numpy_state['complex_value'], np.complex64) or 
                       isinstance(numpy_state['complex_value'], np.complex128))
        self.assertEqual(numpy_state['boolean_tensor'].dtype, np.bool_)
        
        # Verify tensor values were correctly converted
        self.assertTrue(np.allclose(numpy_state['simple_tensor'], 
                                   nested_state['simple_tensor'].numpy()))
        self.assertTrue(np.allclose(numpy_state['nested']['tensor1'], 
                                   nested_state['nested']['tensor1'].numpy()))
    
    def test_tensor_to_array_conversion(self):
        """Test conversion of PyTorch tensors back to NumPy arrays."""
        # Create tensor without gradients
        tensor_no_grad = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        array_no_grad = self.adapter.tensor_to_array(tensor_no_grad)
        
        self.assertIsInstance(array_no_grad, np.ndarray)
        self.assertEqual(array_no_grad.shape, (2, 2))
        self.assertTrue(np.allclose(array_no_grad, tensor_no_grad.numpy()))
        
        # Create tensor with gradients
        tensor_with_grad = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        array_with_grad = self.adapter.tensor_to_array(tensor_with_grad)
        
        self.assertIsInstance(array_with_grad, np.ndarray)
        self.assertEqual(array_with_grad.shape, (2, 2))
        self.assertTrue(np.allclose(array_with_grad, tensor_with_grad.detach().numpy()))
        
        # Test with None tensor
        self.assertIsNone(self.adapter.tensor_to_array(None))
        
        # Test with tensor on non-CPU device (if available)
        if torch.cuda.is_available():
            tensor_gpu = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).cuda()
            array_from_gpu = self.adapter.tensor_to_array(tensor_gpu)
            self.assertIsInstance(array_from_gpu, np.ndarray)
            self.assertEqual(array_from_gpu.shape, (2, 2))
    
    def test_convert_intensity_map(self):
        """Test conversion of intensity tensor to shaped NumPy array."""
        # Create flat intensity tensor
        map_shape = (2, 3, 4)
        total_points = 2 * 3 * 4
        flat_intensity = torch.rand(total_points)
        
        # Convert using convert_intensity_map
        intensity_map = self.adapter.convert_intensity_map(flat_intensity, map_shape)
        
        # Verify shape and values
        self.assertIsInstance(intensity_map, np.ndarray)
        self.assertEqual(intensity_map.shape, map_shape)
        self.assertTrue(np.allclose(intensity_map.flatten(), flat_intensity.numpy()))
        
        # Test with already shaped intensity tensor
        shaped_intensity = torch.rand(map_shape)
        intensity_map = self.adapter.convert_intensity_map(shaped_intensity, map_shape)
        self.assertEqual(intensity_map.shape, map_shape)
        
        # Test with tensor requiring gradients
        flat_intensity_grad = torch.rand(total_points, requires_grad=True)
        intensity_map = self.adapter.convert_intensity_map(flat_intensity_grad, map_shape)
        self.assertEqual(intensity_map.shape, map_shape)
        
        # Test with None intensity
        self.assertIsNone(self.adapter.convert_intensity_map(None, map_shape))
        
        # Test with mismatched sizes
        wrong_size_tensor = torch.rand(total_points + 1)
        with self.assertRaises(ValueError):
            self.adapter.convert_intensity_map(wrong_size_tensor, map_shape)
        
        # Test with already shaped but wrong shape
        wrong_shape_tensor = torch.rand((3, 2, 4))
        with self.assertRaises(ValueError):
            self.adapter.convert_intensity_map(wrong_shape_tensor, map_shape)

class TestGradientFlow(unittest.TestCase):
    """Tests for gradient flow through the adapters."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.pdb_adapter = PDBToTensor()
        self.numpy_adapter = TensorToNumpy()
    
    def test_gradient_flow_through_adapters(self):
        """Test that gradients flow properly through the adapter conversion cycle."""
        # Create NumPy array with coordinates
        coords_np = np.random.rand(10, 3)
        
        # Convert to tensor with PDBToTensor
        coords_tensor = self.pdb_adapter.array_to_tensor(coords_np, requires_grad=True)
        
        # Perform simple computation (sum of squared values)
        result = torch.sum(coords_tensor ** 2)
        
        # Compute gradient with backward()
        result.backward()
        
        # Verify gradients are computed correctly
        self.assertIsNotNone(coords_tensor.grad)
        
        # Expected gradient is 2 * coords_tensor
        expected_grad = 2 * coords_np
        actual_grad = self.numpy_adapter.tensor_to_array(coords_tensor.grad)
        
        # Verify gradient values
        self.assertTrue(np.allclose(actual_grad, expected_grad))
        
        # Test more complex computation
        coords_np = np.random.rand(5, 3)
        coords_tensor = self.pdb_adapter.array_to_tensor(coords_np, requires_grad=True)
        
        # Compute distance from origin for each point
        distances = torch.sqrt(torch.sum(coords_tensor ** 2, dim=1))
        mean_distance = torch.mean(distances)
        
        # Compute gradient
        mean_distance.backward()
        
        # Convert gradients back to NumPy
        grad_np = self.numpy_adapter.tensor_to_array(coords_tensor.grad)
        
        # Verify gradient shape
        self.assertEqual(grad_np.shape, coords_np.shape)
        
        # Manually compute expected gradients for verification
        # For mean of distances, gradient is coords_np[i] / (distances[i] * n_points)
        distances_np = np.sqrt(np.sum(coords_np ** 2, axis=1))
        expected_grad = np.zeros_like(coords_np)
        for i in range(len(coords_np)):
            expected_grad[i] = coords_np[i] / (distances_np[i] * len(coords_np))
        
        # Verify gradient values (with some tolerance for numerical precision)
        self.assertTrue(np.allclose(grad_np, expected_grad, rtol=1e-5, atol=1e-5))

class TestDictConversion(unittest.TestCase):
    """Tests for dictionary conversion methods."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.pdb_adapter = PDBToTensor()
        self.numpy_adapter = TensorToNumpy()
    
    def test_convert_dict_of_arrays(self):
        """Test conversion of dictionary of arrays to tensors."""
        # Create dictionary with NumPy arrays
        arrays_dict = {
            'coords': np.random.rand(10, 3),
            'values': np.random.rand(10),
            'mask': np.random.choice([True, False], size=10)
        }
        
        # Convert using convert_dict_of_arrays
        tensors_dict = self.pdb_adapter.convert_dict_of_arrays(arrays_dict)
        
        # Verify all arrays converted to tensors
        for key, tensor in tensors_dict.items():
            self.assertIsInstance(tensor, torch.Tensor)
            self.assertTrue(tensor.requires_grad)
            self.assertTrue(torch.allclose(tensor.cpu().detach(), 
                                          torch.tensor(arrays_dict[key], dtype=torch.float32)))
        
        # Test with requires_grad=False
        tensors_dict = self.pdb_adapter.convert_dict_of_arrays(arrays_dict, requires_grad=False)
        for tensor in tensors_dict.values():
            self.assertFalse(tensor.requires_grad)
        
        # Test conversion back to NumPy
        numpy_dict = self.numpy_adapter.convert_dict_of_tensors(tensors_dict)
        for key, array in numpy_dict.items():
            self.assertIsInstance(array, np.ndarray)
            self.assertTrue(np.allclose(array, arrays_dict[key]))

class TestModelAdapters(unittest.TestCase):
    """Tests for the ModelAdapters class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.adapter = ModelAdapters()
        # Create a device that's guaranteed to work for testing
        self.device = torch.device('cpu')
        self.adapter_with_device = ModelAdapters(device=self.device)
        
    def test_initialize_from_state(self):
        """Test initializing a model from state dictionary."""
        # Create a mock state dictionary for a simple model
        mock_state = {
            'scalar_param': 5.0,
            'tensor_attr': np.random.rand(10, 3),
            'complex_attr': np.array([1+2j, 3+4j]),
            'nested_dict': {'values': np.random.rand(5)}
        }
        
        # Define a simple model class
        class MockModel:
            def __init__(self):
                pass
        
        # Initialize model from state
        model = self.adapter.initialize_from_state(MockModel, mock_state)
        
        # Verify attribute setting
        self.assertEqual(model.scalar_param, 5.0)
        self.assertIsInstance(model.tensor_attr, torch.Tensor)
        self.assertTrue(model.tensor_attr.requires_grad)
        self.assertIsInstance(model.nested_dict, dict)
        self.assertIsInstance(model.nested_dict['values'], torch.Tensor)
        
        # Verify gradients work
        output = model.tensor_attr.sum()
        output.backward()
        self.assertIsNotNone(model.tensor_attr.grad)
        
    def test_initialize_one_phonon_from_state(self):
        """Test initializing a OnePhonon model from state dictionary."""
        # Create a mock state dictionary mimicking OnePhonon structure
        mock_phonon_state = {
            'kvec': np.random.rand(2, 2, 2, 3),
            'kvec_norm': np.random.rand(2, 2, 2, 1),
            'V': np.random.rand(2, 2, 2, 6, 6),  # Real part only initially
            'Winv': np.random.rand(2, 2, 2, 6),  # Real part only initially
            'n_asu': 2,
            'n_dof_per_asu': 3
        }
        
        # Define a mock OnePhonon class
        class MockOnePhonon:
            def __init__(self):
                pass
        
        # Initialize model from state
        model = self.adapter.initialize_one_phonon_from_state(MockOnePhonon, mock_phonon_state)
        
        # Verify attribute setting
        self.assertEqual(model.n_asu, 2)
        self.assertEqual(model.n_dof_per_asu, 3)
        
        # Verify complex tensor handling
        self.assertTrue(torch.is_complex(model.V))
        self.assertTrue(torch.is_complex(model.Winv))
        
        # Verify gradients work
        output = torch.real(model.V).sum()
        output.backward()
        self.assertIsNotNone(model.V.grad)
        
        # Verify the real part matches the input data
        v_real = torch.real(model.V).detach().cpu().numpy()
        self.assertTrue(np.allclose(v_real, mock_phonon_state['V']))
        
        # Verify the imaginary part is zero
        v_imag = torch.imag(model.V).detach().cpu().numpy()
        self.assertTrue(np.allclose(v_imag, np.zeros_like(v_imag)))
        
    def test_compatibility_with_logger(self):
        """Test compatibility with Logger's state format."""
        # Create a mock Logger
        from eryx.autotest.logger import Logger
        logger = Logger()
        
        # Create a simple object with NumPy arrays
        class SimpleObject:
            def __init__(self):
                self.data = np.random.rand(5, 3)
                self.values = np.random.rand(10)
                self.name = "test_object"
        
        # Create object and capture state
        obj = SimpleObject()
        state = logger.captureState(obj)
        
        # Mock saving and loading the state
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.log', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # Save and load the state
            logger.saveStateLog(temp_path, state)
            loaded_state = logger.loadStateLog(temp_path)
            
            # Test conversion of loaded state
            tensor_state = self.adapter.pdb_to_tensor.convert_state_dict(loaded_state)
            
            # Verify conversion worked correctly
            self.assertIsInstance(tensor_state['data'], torch.Tensor)
            self.assertIsInstance(tensor_state['values'], torch.Tensor)
            self.assertEqual(tensor_state['name'], "test_object")
            
            # Test model initialization from loaded state
            class MockModel:
                def __init__(self):
                    pass
            
            model = self.adapter.initialize_from_state(MockModel, loaded_state)
            self.assertIsInstance(model.data, torch.Tensor)
            self.assertIsInstance(model.values, torch.Tensor)
            
        finally:
            # Clean up temp file
            import os
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def test_model_adapters_initialization(self):
        """Test initialization of ModelAdapters."""
        # Verify sub-adapters are correctly initialized
        self.assertIsInstance(self.adapter.pdb_to_tensor, PDBToTensor)
        self.assertIsInstance(self.adapter.grid_to_tensor, GridToTensor)
        self.assertIsInstance(self.adapter.tensor_to_numpy, TensorToNumpy)
        
        # Check device propagation
        self.assertEqual(self.adapter_with_device.device, self.device)
        self.assertEqual(self.adapter_with_device.pdb_to_tensor.device, self.device)
        self.assertEqual(self.adapter_with_device.grid_to_tensor.device, self.device)
    
    def test_model_adapters_integration(self):
        """Test integration of adapter components in ModelAdapters."""
        # Create a simple mock model
        mock_model = type('MockOnePhonon', (), {
            'model': type('MockAtomicModel', (), {
                'xyz': np.random.rand(1, 10, 3),
                'ff_a': np.random.rand(1, 10, 4),
                'ff_b': np.random.rand(1, 10, 4),
                'ff_c': np.random.rand(1, 10),
                'cell': np.array([10.0, 10.0, 10.0, 90.0, 90.0, 90.0]),
                'A_inv': np.random.rand(3, 3)
            }),
            'q_grid': np.random.rand(100, 3),
            'map_shape': (10, 10, 10),
            'hsampling': (0, 10, 1),
            'ksampling': (0, 10, 1),
            'lsampling': (0, 10, 1),
            'gnm': type('MockGNM', (), {
                'enm_cutoff': 4.0,
                'gamma_intra': 1.0,
                'gamma_inter': 0.5
            }),
            'res_limit': 2.0,
            'expand_p1': True,
            'group_by': 'asu',
            'model_type': 'gnm'
        })
        
        # Test adapt_one_phonon_inputs
        inputs_dict = self.adapter.adapt_one_phonon_inputs(mock_model)
        
        # Verify key components are converted
        self.assertIn('atomic_model', inputs_dict)
        self.assertIn('q_grid', inputs_dict)
        self.assertIn('map_shape', inputs_dict)
        self.assertIn('gnm_params', inputs_dict)
        
        # Verify tensor conversion
        self.assertIsInstance(inputs_dict['atomic_model']['xyz'], torch.Tensor)
        self.assertIsInstance(inputs_dict['q_grid'], torch.Tensor)
        
        # Verify scalar parameters are preserved
        self.assertEqual(inputs_dict['res_limit'], 2.0)
        self.assertEqual(inputs_dict['expand_p1'], True)
        self.assertEqual(inputs_dict['group_by'], 'asu')
        self.assertEqual(inputs_dict['model_type'], 'gnm')
        
        # Verify GNM parameters
        self.assertEqual(inputs_dict['gnm_params']['enm_cutoff'], 4.0)
        self.assertEqual(inputs_dict['gnm_params']['gamma_intra'], 1.0)
        self.assertEqual(inputs_dict['gnm_params']['gamma_inter'], 0.5)
        
        # Create mock output data
        mock_output = {
            'intensity': torch.rand(1000),
            'map_shape': (10, 10, 10)
        }
        
        # Test adapt_one_phonon_outputs
        intensity_map = self.adapter.adapt_one_phonon_outputs(mock_output)
        
        # Verify output conversion
        self.assertIsInstance(intensity_map, np.ndarray)
        self.assertEqual(intensity_map.shape, (10, 10, 10))
        
        # Test with already shaped intensity
        mock_output_shaped = {
            'intensity': torch.rand((10, 10, 10))
        }
        intensity_map = self.adapter.adapt_one_phonon_outputs(mock_output_shaped)
        self.assertEqual(intensity_map.shape, (10, 10, 10))
        
        # Test with None outputs
        with self.assertRaises(ValueError):
            self.adapter.adapt_one_phonon_outputs(None)
        
        # Test with missing intensity
        with self.assertRaises(ValueError):
            self.adapter.adapt_one_phonon_outputs({})
        
        # Test with missing map_shape for flat intensity
        with self.assertRaises(ValueError):
            self.adapter.adapt_one_phonon_outputs({'intensity': torch.rand(100)})


if __name__ == '__main__':
    unittest.main()
