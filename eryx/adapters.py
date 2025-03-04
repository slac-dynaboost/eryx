"""
Adapter components to bridge NumPy and PyTorch implementations.

This module contains adapter classes to convert between NumPy arrays and PyTorch tensors,
as well as domain-specific adapters for the diffuse scattering calculations.
All adapters preserve the computational graph for gradient backpropagation.
"""

import numpy as np
import torch
from typing import Tuple, List, Dict, Optional, Union, Any
import gemmi  # For crystallographic data structures

class PDBToTensor:
    """
    Adapter to convert PDB data from NumPy arrays to PyTorch tensors.
    
    This class handles the conversion of AtomicModel and related classes from
    the NumPy implementation to PyTorch tensors suitable for gradient-based calculations.
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize the adapter for converting PDB data to PyTorch tensors.
        
        Args:
            device: The PyTorch device to place tensors on. If None, uses CUDA if 
                   available, otherwise CPU.
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def convert_atomic_model(self, model: Any) -> Dict[str, Any]:
        """
        Convert an AtomicModel to PyTorch tensors.
        
        Args:
            model: AtomicModel instance from eryx.pdb
            
        Returns:
            Dictionary containing PyTorch tensor versions of the model attributes
            with gradient support for optimization.
            
        Note:
            Converts key array attributes to tensors while preserving non-array attributes.
            The returned dictionary can be used directly with PyTorch implementations.
        """
        if model is None:
            raise ValueError("Cannot convert None model")
            
        result = {}
        
        # Convert key array attributes to tensors
        tensor_attributes = {
            'xyz': True,           # Atomic coordinates (n_conf, n_atoms, 3)
            'ff_a': True,          # Form factor coefficients (n_conf, n_atoms, 4)
            'ff_b': True,          # Form factor coefficients (n_conf, n_atoms, 4)
            'ff_c': True,          # Form factor coefficients (n_conf, n_atoms)
            'adp': True,           # Atomic displacement parameters
            'cell': True,          # Unit cell parameters (6,)
            'A_inv': True,         # Fractional cell matrix (3, 3)
            'unit_cell_axes': True # Unit cell axes (3, 3)
        }
        
        # Process each attribute
        for attr_name, requires_grad in tensor_attributes.items():
            if hasattr(model, attr_name):
                attr_value = getattr(model, attr_name)
                if attr_value is not None:
                    result[attr_name] = self.array_to_tensor(attr_value, requires_grad=requires_grad)
        
        # Preserve non-array attributes
        non_tensor_attributes = [
            'space_group', 'n_asu', 'n_conf', 'sym_ops', 'transformations', 'elements'
        ]
        
        for attr_name in non_tensor_attributes:
            if hasattr(model, attr_name):
                result[attr_name] = getattr(model, attr_name)
        
        return result
    
    def convert_crystal(self, crystal: Any) -> Dict[str, Any]:
        """
        Convert a Crystal object to PyTorch tensors.
        
        Args:
            crystal: Crystal instance from eryx.pdb
            
        Returns:
            Dictionary containing PyTorch tensor versions of the crystal attributes
        """
        # TODO: Convert relevant attributes to PyTorch tensors
        # TODO: Handle unit cell information
        # TODO: Preserve crystallographic metadata
        
        raise NotImplementedError("convert_crystal not implemented")
    
    def convert_gnm(self, gnm: Any) -> Dict[str, Any]:
        """
        Convert a GaussianNetworkModel to PyTorch tensors.
        
        Args:
            gnm: GaussianNetworkModel instance from eryx.pdb
            
        Returns:
            Dictionary containing PyTorch tensor versions of the GNM attributes
        """
        # TODO: Convert gamma matrix to tensor
        # TODO: Convert neighbor lists to tensor format
        # TODO: Preserve parameter information
        
        raise NotImplementedError("convert_gnm not implemented")
    
    def array_to_tensor(self, array: np.ndarray, requires_grad: bool = True) -> torch.Tensor:
        """
        Convert a NumPy array to a PyTorch tensor with gradient support.
        
        Args:
            array: NumPy array to convert. Can be of any shape or dtype.
            requires_grad: Whether the tensor requires gradients for backpropagation.
                           Only applied to floating point tensors.
            
        Returns:
            PyTorch tensor with the same data, on the specified device with requires_grad set.
            Returns None if input is None.
            
        Examples:
            >>> adapter = PDBToTensor()
            >>> x_np = np.array([[1.0, 2.0], [3.0, 4.0]])
            >>> x_tensor = adapter.array_to_tensor(x_np, requires_grad=True)
            >>> x_tensor.shape
            torch.Size([2, 2])
            >>> x_tensor.requires_grad
            True
        """
        if array is None:
            return None
            
        if array.size == 0:  # Handle empty arrays
            tensor = torch.from_numpy(array.copy()).to(self.device)
        else:
            # Use clone to avoid memory sharing issues with NumPy
            tensor = torch.from_numpy(array.copy()).to(self.device)
        
        # Only set requires_grad for floating point tensors
        if requires_grad and tensor.dtype.is_floating_point:
            tensor.requires_grad_(True)
            
        return tensor
    
    def convert_dict_of_arrays(self, dict_arrays: Dict[Any, np.ndarray], 
                              requires_grad: bool = True) -> Dict[Any, torch.Tensor]:
        """
        Convert a dictionary of NumPy arrays to PyTorch tensors.
        
        Args:
            dict_arrays: Dictionary mapping keys to NumPy arrays
            requires_grad: Whether tensors require gradients (only applied to floating point tensors)
            
        Returns:
            Dictionary mapping the same keys to PyTorch tensors
        """
        result = {}
        for k, v in dict_arrays.items():
            # Handle different dtypes appropriately
            if v.dtype == bool:
                # Convert boolean arrays to float tensors if requires_grad is True
                if requires_grad:
                    tensor = torch.tensor(v, dtype=torch.float32, device=self.device)
                    tensor.requires_grad_(True)
                else:
                    tensor = torch.tensor(v, dtype=torch.bool, device=self.device)
            else:
                tensor = self.array_to_tensor(v, requires_grad=requires_grad)
            result[k] = tensor
        return result

class GridToTensor:
    """
    Adapter to convert grid data from NumPy arrays to PyTorch tensors.
    
    This class handles the conversion of reciprocal space grids and related data
    from the NumPy implementation to PyTorch tensors.
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize the adapter for converting grid data to PyTorch tensors.
        
        Args:
            device: The PyTorch device to place tensors on. If None, uses CUDA if 
                   available, otherwise CPU.
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def convert_grid(self, q_grid: np.ndarray, map_shape: Tuple[int, int, int], 
                    requires_grad: bool = True) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        """
        Convert a grid of q-vectors to PyTorch tensor with gradient support.
        
        Args:
            q_grid: NumPy array of shape (n_points, 3) with q-vectors
            map_shape: Tuple with 3D map shape (dim_h, dim_k, dim_l)
            requires_grad: Whether the tensor requires gradients for backpropagation
            
        Returns:
            Tuple containing:
                - PyTorch tensor of q-vectors with shape (n_points, 3)
                - Tuple with map shape (unchanged)
                
        Note:
            The q-grid tensor will have requires_grad=True by default, as it's
            typically used in gradient-based optimization.
        """
        if q_grid is None:
            raise ValueError("q_grid cannot be None")
            
        if q_grid.size == 0:
            # Handle empty grid
            q_grid_tensor = torch.zeros((0, 3), dtype=torch.float32, device=self.device)
            q_grid_tensor.requires_grad_(requires_grad)
            return q_grid_tensor, map_shape
            
        # Convert to tensor with gradient support
        # Ensure we use the same dtype as the input for consistency
        input_dtype = torch.get_default_dtype() if q_grid.dtype == np.float64 else torch.float32
        q_grid_tensor = torch.tensor(q_grid, dtype=input_dtype, device=self.device)
        
        # Only set requires_grad for floating point tensors
        if requires_grad:
            q_grid_tensor.requires_grad_(True)
        
        return q_grid_tensor, map_shape
    
    def convert_mask(self, mask: np.ndarray, requires_grad: bool = False) -> torch.Tensor:
        """
        Convert a boolean mask to PyTorch tensor.
        
        Args:
            mask: NumPy boolean array of any shape
            requires_grad: Whether the tensor requires gradients (defaults to False for masks)
                           Note: Boolean tensors cannot require gradients, this will be ignored
                           for boolean masks.
            
        Returns:
            PyTorch boolean tensor on the specified device
            
        Examples:
            >>> adapter = GridToTensor()
            >>> mask_np = np.array([True, False, True])
            >>> mask_tensor = adapter.convert_mask(mask_np)
            >>> mask_tensor.dtype
            torch.bool
        """
        if mask is None:
            return None
            
        if mask.size == 0:
            # Handle empty mask
            mask_tensor = torch.zeros(mask.shape, dtype=torch.bool, device=self.device)
        else:
            # Ensure boolean dtype is preserved
            mask_tensor = torch.tensor(mask, dtype=torch.bool, device=self.device)
        
        # Boolean tensors cannot require gradients, so we ignore requires_grad
        # If requires_grad is True and we need gradients, convert to float
        if requires_grad:
            # Convert to float tensor that can have gradients
            return mask_tensor.to(torch.float32).requires_grad_(True)
        
        return mask_tensor
    
    def convert_symmetry_ops(self, sym_ops: Dict[int, np.ndarray]) -> Dict[int, torch.Tensor]:
        """
        Convert symmetry operations to PyTorch tensors.
        
        Args:
            sym_ops: Dictionary mapping IDs to rotation matrices
            
        Returns:
            Dictionary mapping IDs to tensor rotation matrices
        """
        # TODO: Convert each symmetry operation matrix to tensor
        # TODO: Maintain dictionary structure
        
        raise NotImplementedError("convert_symmetry_ops not implemented")

class TensorToNumpy:
    """
    Adapter to convert PyTorch tensors back to NumPy arrays.
    
    This class handles the conversion of PyTorch tensors to NumPy arrays
    for visualization, saving, or compatibility with existing code.
    """
    
    def __init__(self):
        """
        Initialize the adapter for converting PyTorch tensors back to NumPy arrays.
        
        This adapter handles proper detachment of gradients and device transfer
        to ensure safe conversion back to NumPy for visualization, saving, or
        compatibility with existing code.
        """
        pass
    
    def tensor_to_array(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Convert a PyTorch tensor to a NumPy array.
        
        Args:
            tensor: PyTorch tensor to convert, can be of any shape or dtype
            
        Returns:
            NumPy array with the same data
            
        Note:
            This method properly detaches gradients and moves the tensor to CPU
            before conversion, ensuring safe conversion regardless of the tensor's
            original device or gradient status.
        """
        if tensor is None:
            return None
            
        # Detach from computation graph if it requires gradients
        if tensor.requires_grad:
            tensor = tensor.detach()
        
        # Move to CPU if on another device
        if tensor.device.type != 'cpu':
            tensor = tensor.cpu()
        
        return tensor.numpy()
    
    def convert_dict_of_tensors(self, dict_tensors: Dict[Any, torch.Tensor]) -> Dict[Any, np.ndarray]:
        """
        Convert a dictionary of PyTorch tensors to NumPy arrays.
        
        Args:
            dict_tensors: Dictionary mapping keys to PyTorch tensors
            
        Returns:
            Dictionary mapping the same keys to NumPy arrays
        """
        return {k: self.tensor_to_array(v) for k, v in dict_tensors.items()}
    
    def convert_intensity_map(self, intensity: torch.Tensor, map_shape: Tuple[int, int, int]) -> np.ndarray:
        """
        Convert an intensity map tensor to a NumPy array with proper shape.
        
        Args:
            intensity: PyTorch tensor with intensity values, either flat (n_points,) 
                      or already shaped (dim_h, dim_k, dim_l)
            map_shape: Tuple with desired 3D shape (dim_h, dim_k, dim_l)
            
        Returns:
            NumPy array with intensity map reshaped to 3D with shape map_shape
            
        Note:
            If the input tensor is already 3D and matches map_shape, it will be
            returned as-is (after detaching and converting to NumPy).
        """
        if intensity is None:
            return None
            
        # Detach from computation graph if it requires gradients
        if intensity.requires_grad:
            intensity = intensity.detach()
        
        # Move to CPU if on another device
        if intensity.device.type != 'cpu':
            intensity = intensity.cpu()
        
        # Convert to NumPy
        intensity_np = intensity.numpy()
        
        # Reshape if necessary
        if intensity_np.ndim == 1:
            # Flat array needs reshaping
            total_points = map_shape[0] * map_shape[1] * map_shape[2]
            if intensity_np.size != total_points:
                raise ValueError(f"Intensity tensor size {intensity_np.size} doesn't match "
                                f"map_shape total size {total_points}")
            intensity_np = intensity_np.reshape(map_shape)
        elif intensity_np.shape != map_shape:
            # Already 3D but wrong shape
            raise ValueError(f"Intensity tensor shape {intensity_np.shape} doesn't match "
                            f"requested map_shape {map_shape}")
        
        return intensity_np

class ModelAdapters:
    """
    Adapters for the various model classes in eryx.
    
    This class contains methods to convert between the NumPy and PyTorch
    versions of the various model classes used in diffuse scattering calculations.
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize the adapters for high-level model conversion.
        
        This class provides integrated conversion between NumPy and PyTorch
        implementations of the various model classes used in diffuse scattering
        calculations.
        
        Args:
            device: The PyTorch device to place tensors on. If None, uses CUDA if
                   available, otherwise CPU.
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pdb_to_tensor = PDBToTensor(device)
        self.grid_to_tensor = GridToTensor(device)
        self.tensor_to_numpy = TensorToNumpy()
    
    def adapt_one_phonon_inputs(self, np_model: Any) -> Dict[str, Any]:
        """
        Adapt inputs for the OnePhonon model from NumPy to PyTorch.
        
        This method extracts the necessary inputs from a NumPy-based OnePhonon model
        and converts them to PyTorch tensors for use with the PyTorch implementation.
        
        Args:
            np_model: OnePhonon instance from eryx.models
            
        Returns:
            Dictionary with PyTorch tensor versions of inputs structured for
            the PyTorch implementation
            
        Note:
            Expected model attributes include:
            - model: AtomicModel instance
            - q_grid: Grid of q-vectors
            - map_shape: Shape of the map
            - hsampling, ksampling, lsampling: Sampling parameters
            - gnm: GaussianNetworkModel instance (if available)
        """
        if np_model is None:
            raise ValueError("Cannot adapt None model")
            
        result = {}
        
        # Convert atomic model if available
        if hasattr(np_model, 'model') and np_model.model is not None:
            # Store the converted model in a separate key to avoid string indexing issues
            result['atomic_model'] = self.pdb_to_tensor.convert_atomic_model(np_model.model)
        
        # Convert grid data if available
        if hasattr(np_model, 'q_grid') and hasattr(np_model, 'map_shape'):
            result['q_grid'], result['map_shape'] = self.grid_to_tensor.convert_grid(
                np_model.q_grid, np_model.map_shape
            )
        
        # Convert sampling parameters
        for param in ['hsampling', 'ksampling', 'lsampling']:
            if hasattr(np_model, param):
                result[param] = getattr(np_model, param)
        
        # Convert GNM if available
        if hasattr(np_model, 'gnm') and np_model.gnm is not None:
            # Extract key parameters from GNM
            gnm_params = {}
            for param in ['enm_cutoff', 'gamma_intra', 'gamma_inter']:
                if hasattr(np_model.gnm, param):
                    gnm_params[param] = getattr(np_model.gnm, param)
            result['gnm_params'] = gnm_params
        
        # Convert other scalar parameters
        scalar_params = ['res_limit', 'expand_p1', 'group_by']
        for param in scalar_params:
            if hasattr(np_model, param):
                result[param] = getattr(np_model, param)
        
        # Handle the 'model' parameter separately to avoid confusion with the 'model' attribute
        if hasattr(np_model, 'model_type'):
            result['model_type'] = np_model.model_type
        elif hasattr(np_model, 'model') and isinstance(np_model.model, str):
            result['model_type'] = np_model.model
        
        return result
    
    def adapt_one_phonon_outputs(self, torch_outputs: Dict[str, torch.Tensor]) -> np.ndarray:
        """
        Adapt outputs from the PyTorch OnePhonon model back to NumPy.
        
        This method extracts the intensity map from the PyTorch model outputs
        and converts it back to a NumPy array for visualization or further processing.
        
        Args:
            torch_outputs: Dictionary with PyTorch tensor outputs, containing at minimum:
                          - 'intensity': The diffuse intensity tensor
                          - 'map_shape': The shape for the output map
            
        Returns:
            NumPy array with intensity map in the appropriate shape
            
        Note:
            The intensity tensor can be either flat (n_points,) or already shaped
            (dim_h, dim_k, dim_l). This method handles both cases.
        """
        if torch_outputs is None:
            raise ValueError("Cannot adapt None outputs")
            
        # Extract intensity map and map shape
        if 'intensity' not in torch_outputs:
            raise ValueError("Output dictionary must contain 'intensity' key")
            
        intensity = torch_outputs['intensity']
        
        # Get map shape
        if 'map_shape' in torch_outputs:
            map_shape = torch_outputs['map_shape']
        else:
            # If intensity is already shaped, use its shape
            if intensity.dim() == 3:
                map_shape = tuple(intensity.shape)
            else:
                raise ValueError("Output dictionary must contain 'map_shape' key for flat intensity")
        
        # Convert to NumPy with proper shape
        return self.tensor_to_numpy.convert_intensity_map(intensity, map_shape)
    
    def adapt_rigid_body_translations_inputs(self, np_model: Any) -> Dict[str, Any]:
        """
        Adapt inputs for the RigidBodyTranslations model from NumPy to PyTorch.
        
        Args:
            np_model: RigidBodyTranslations instance from eryx.models
            
        Returns:
            Dictionary with PyTorch tensor versions of inputs
        """
        # TODO: Extract necessary inputs
        # TODO: Convert to PyTorch tensors
        # TODO: Structure for easy passing to PyTorch implementation
        
        raise NotImplementedError("adapt_rigid_body_translations_inputs not implemented")
    
    # Add similar methods for other model classes
