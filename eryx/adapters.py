"""
Adapter components to bridge NumPy and PyTorch implementations.

This module contains adapter classes to convert between NumPy arrays and PyTorch tensors,
as well as domain-specific adapters for diffuse scattering calculations.

Key features:
- Array/tensor conversion preserving gradients
- State dictionary conversion for state-based testing
- Model initialization from state dictionaries
- Complex object conversion with nested structures
- Compatibility with Logger state format

Example usage:
    # Convert NumPy array to tensor
    adapter = PDBToTensor()
    tensor = adapter.array_to_tensor(array, requires_grad=True)
    
    # Convert complete state dictionary from Logger
    numpy_state = logger.loadStateLog("logs/eryx.models.OnePhonon._state_before__build_A.log")
    tensor_state = adapter.convert_state_dict(numpy_state)
    
    # Initialize model from state
    model_adapters = ModelAdapters()
    model = model_adapters.initialize_from_state(OnePhonon, state_dict)
    
    # Initialize OnePhonon model with special handling
    model = model_adapters.initialize_one_phonon_from_state(OnePhonon, state_dict)
    
    # Convert model state back to NumPy for comparison
    numpy_state = model_adapters.convert_state_for_comparison(model)
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
    
    def extract_element_weights(self, model: Any) -> torch.Tensor:
        """
        Extract atomic weights from model element information.
        
        Args:
            model: AtomicModel instance or any object with element information
            
        Returns:
            Tensor of element weights
            
        This method implements a robust multi-strategy approach to extract
        element weights from various sources, with proper fallbacks.
        """
        weights = []
        
        # Strategy 1: Try gemmi_structure if available
        if hasattr(model, '_gemmi_structure'):
            try:
                structure = model._gemmi_structure
                for model_obj in structure:
                    for chain in model_obj:
                        for residue in chain:
                            for atom in residue:
                                element = atom.element
                                if element and hasattr(element, 'weight'):
                                    weights.append(float(element.weight))
                                else:
                                    # Default to carbon weight
                                    weights.append(12.0)
                if weights:
                    return torch.tensor(weights, device=self.device, dtype=torch.float32)
            except Exception as e:
                print(f"Error extracting weights from gemmi structure: {e}")
        
        # Strategy 2: Try model.elements if available
        if hasattr(model, 'elements'):
            try:
                if isinstance(model.elements, list) and len(model.elements) > 0:
                    # Handle nested list structure
                    if isinstance(model.elements[0], list):
                        for structure in model.elements:
                            for element in structure:
                                if hasattr(element, 'weight'):
                                    weights.append(float(element.weight))
                                elif isinstance(element, dict) and 'weight' in element:
                                    weights.append(float(element['weight']))
                                elif isinstance(element, (int, float)):
                                    weights.append(float(element))
                                else:
                                    # Default weight
                                    weights.append(12.0)
                    # Direct list of elements
                    elif all(hasattr(e, 'weight') for e in model.elements if hasattr(e, '__dict__')):
                        weights = [float(e.weight) for e in model.elements]
                    # List of numeric values
                    elif all(isinstance(e, (int, float)) for e in model.elements):
                        weights = [float(e) for e in model.elements]
                
                if weights:
                    return torch.tensor(weights, device=self.device, dtype=torch.float32)
            except Exception as e:
                print(f"Error extracting weights from elements attribute: {e}")
        
        # Strategy 3: Try cached original weights
        if hasattr(model, '_original_weights'):
            try:
                weights = model._original_weights
                if weights:
                    return torch.tensor(weights, device=self.device, dtype=torch.float32)
            except Exception as e:
                print(f"Error extracting weights from _original_weights: {e}")
        
        # Fallback: return carbon weights
        if hasattr(model, 'xyz'):
            n_atoms = model.xyz.shape[0] * model.xyz.shape[1] if model.xyz.ndim > 2 else model.xyz.shape[0]
            return torch.ones(n_atoms, device=self.device, dtype=torch.float32) * 12.0
        
        # Ultimate fallback
        return torch.ones(1, device=self.device, dtype=torch.float32) * 12.0
        
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
        
        # Handle Gemmi structure specially
        if hasattr(model, 'structure'):
            result['structure_dict'] = self.convert_gemmi_to_tensor_dict(model.structure)
        
        # Extract and add element weights explicitly
        result['element_weights'] = self.extract_element_weights(model)
        
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
    
    def _safe_get_attr(self, obj: Any, attr_path: str, default: Any = None) -> Any:
        """
        Safely access an attribute through a path with dot notation.
        
        Args:
            obj: The object to access attributes from
            attr_path: Path to the attribute using dot notation (e.g., 'model.unit_cell_axes')
            default: Default value to return if attribute doesn't exist
            
        Returns:
            The attribute value or default if not found
            
        Example:
            >>> self._safe_get_attr(crystal, 'model.n_asu', 1)
            # Returns crystal.model.n_asu if it exists, otherwise 1
        """
        if obj is None:
            return default
            
        parts = attr_path.split('.')
        current = obj
        
        try:
            for part in parts:
                if not hasattr(current, part):
                    return default
                current = getattr(current, part)
            return current
        except Exception:
            return default

    class TorchCrystal:
        """
        A PyTorch-compatible wrapper for Crystal objects.
        
        This class wraps the original Crystal object and provides PyTorch tensor
        versions of its attributes and methods, ensuring proper gradient flow.
        """
        
        def __init__(self, original_crystal, adapter, device=None):
            """
            Initialize the TorchCrystal wrapper.
            
            Args:
                original_crystal: The original Crystal object to wrap
                adapter: The PDBToTensor adapter for tensor conversion
                device: PyTorch device to use (defaults to adapter's device)
            """
            self._original = original_crystal
            self._adapter = adapter
            self.device = device or adapter.device
            
            # Store key attributes as tensors
            if hasattr(original_crystal.model, 'unit_cell_axes'):
                self.unit_cell_axes = adapter.array_to_tensor(
                    original_crystal.model.unit_cell_axes, 
                    requires_grad=False
                )
            else:
                self.unit_cell_axes = None
                
            if hasattr(original_crystal.model, 'cell'):
                self.cell = adapter.array_to_tensor(
                    original_crystal.model.cell,
                    requires_grad=False
                )
            else:
                self.cell = None
                
            # Store scalar attributes directly
            self.n_cell = getattr(original_crystal, 'n_cell', 1)
            self.n_asu = getattr(original_crystal.model, 'n_asu', 1)
            
            # Calculate n_atoms_per_asu if possible
            if hasattr(original_crystal, 'get_asu_xyz'):
                try:
                    self.n_atoms_per_asu = original_crystal.get_asu_xyz().shape[0]
                except Exception:
                    self.n_atoms_per_asu = 0
            else:
                self.n_atoms_per_asu = 0
        
        def hkl_to_id(self, hkl=None):
            """
            Convert hkl indices to cell ID.
            
            Args:
                hkl: List of h, k, l indices
                
            Returns:
                Cell ID as an integer
            """
            if hasattr(self._original, 'hkl_to_id'):
                return self._original.hkl_to_id(hkl)
            return 0
            
        def id_to_hkl(self, cell_id=0):
            """
            Convert cell ID to hkl indices.
            
            Args:
                cell_id: Cell ID as an integer
                
            Returns:
                List of h, k, l indices
            """
            if hasattr(self._original, 'id_to_hkl'):
                return self._original.id_to_hkl(cell_id)
            return [0, 0, 0]
            
        def get_asu_xyz(self, asu_id=0, unit_cell=None):
            """
            Get atomic coordinates for a specific asymmetric unit in a specific unit cell.
            
            Args:
                asu_id: Asymmetric unit index
                unit_cell: Index of the unit cell along the 3 dimensions
                
            Returns:
                PyTorch tensor of atomic coordinates
            """
            if hasattr(self._original, 'get_asu_xyz'):
                xyz = self._original.get_asu_xyz(asu_id, unit_cell)
                return self._adapter.array_to_tensor(xyz, requires_grad=True)
            return torch.zeros((1, 3), device=self.device, requires_grad=True)
            
        def get_unitcell_origin(self, unit_cell=None):
            """
            Get the origin coordinates of a unit cell.
            
            Args:
                unit_cell: Index of the unit cell along the 3 dimensions
                
            Returns:
                PyTorch tensor of origin coordinates
            """
            if hasattr(self._original, 'get_unitcell_origin'):
                origin = self._original.get_unitcell_origin(unit_cell)
                return self._adapter.array_to_tensor(origin, requires_grad=True)
            return torch.zeros(3, device=self.device, requires_grad=True)
            
        def supercell_extent(self, nx=0, ny=0, nz=0):
            """
            Set the supercell extent.
            
            Args:
                nx, ny, nz: Extent in each dimension
                
            Returns:
                Result from the original crystal's method
            """
            if hasattr(self._original, 'supercell_extent'):
                return self._original.supercell_extent(nx, ny, nz)
            return None
    
    def convert_crystal(self, crystal: Any) -> 'TorchCrystal':
        """
        Convert a Crystal object to a TorchCrystal wrapper.
        
        Args:
            crystal: Crystal instance from eryx.pdb
            
        Returns:
            TorchCrystal wrapper with PyTorch tensor versions of the crystal attributes
        """
        if crystal is None:
            raise ValueError("Cannot convert None crystal")
            
        if not hasattr(crystal, 'model'):
            raise ValueError("Crystal object must have a 'model' attribute. Check object structure.")
        
        # Create and return a TorchCrystal wrapper
        return self.TorchCrystal(crystal, self, self.device)
    
    def convert_gnm(self, gnm: Any) -> Dict[str, Any]:
        """
        Convert a GaussianNetworkModel to PyTorch tensors.
        
        Args:
            gnm: GaussianNetworkModel instance from eryx.pdb
            
        Returns:
            Dictionary containing PyTorch tensor versions of the GNM attributes
        """
        if gnm is None:
            raise ValueError("Cannot convert None GNM")
            
        result = {
            # Basic parameters
            'enm_cutoff': gnm.enm_cutoff,
            'gamma_intra': gnm.gamma_intra,
            'gamma_inter': gnm.gamma_inter,
            
            # Convert gamma tensor
            'gamma': self.array_to_tensor(gnm.gamma, requires_grad=True),
            
            # Simple function to access neighbors
            'get_neighbors': lambda i_asu, i_cell, j_asu, i_at: self.array_to_tensor(
                gnm.asu_neighbors[i_asu][i_cell][j_asu][i_at], 
                requires_grad=False
            ),
            
            # Reference to original object for complex attributes
            '_original_gnm': gnm
        }
        
        # Handle crystal reference if present
        if hasattr(gnm, 'crystal'):
            # Validate the crystal has required attributes before converting
            if not hasattr(gnm.crystal, 'model'):
                # If crystal structure is invalid, log a warning and create a minimal valid structure
                import logging
                logging.warning("GNM crystal object missing 'model' attribute. Creating minimal conversion.")
                result['crystal'] = {
                    'n_cell': self._safe_get_attr(gnm.crystal, 'n_cell', 1),
                    'id_to_hkl': getattr(gnm.crystal, 'id_to_hkl', lambda x: [0, 0, 0]),
                    'hkl_to_id': getattr(gnm.crystal, 'hkl_to_id', lambda x: 0),
                    '_original_crystal': gnm.crystal
                }
            else:
                # Normal conversion with validation
                result['crystal'] = self.convert_crystal(gnm.crystal)
        
        return result
    
    def convert_gemmi_to_tensor_dict(self, gemmi_obj: Any) -> Dict[str, torch.Tensor]:
        """
        Convert Gemmi object to dictionary of tensors.
        
        Args:
            gemmi_obj: Gemmi object to convert
            
        Returns:
            Dictionary with tensor representations of numerical properties
        """
        # Import GemmiSerializer with fallback
        try:
            from eryx.autotest.serializer import GemmiSerializer
            gemmi_serializer = GemmiSerializer()
        except ImportError:
            # Create a minimal serializer if the full one is not available
            class MinimalGemmiSerializer:
                def serialize_gemmi_object(self, obj):
                    # Extract basic info
                    result = {"_gemmi_type": "Unknown"}
                    
                    # Try to get cell parameters if available
                    if hasattr(obj, "cell"):
                        try:
                            cell = obj.cell
                            result["cell"] = {
                                "a": getattr(cell, "a", 0.0),
                                "b": getattr(cell, "b", 0.0),
                                "c": getattr(cell, "c", 0.0),
                                "alpha": getattr(cell, "alpha", 0.0),
                                "beta": getattr(cell, "beta", 0.0),
                                "gamma": getattr(cell, "gamma", 0.0)
                            }
                        except Exception:
                            pass
                    
                    return result
            
            gemmi_serializer = MinimalGemmiSerializer()
        
        # First serialize to dictionary
        try:
            serialized = gemmi_serializer.serialize_gemmi_object(gemmi_obj)
            
            # Now convert numerical values to tensors
            result = {"_gemmi_type": serialized.get("_gemmi_type", "Unknown")}
            
            # Process based on type
            if serialized.get("_gemmi_type") == "Structure":
                # Handle Structure specially
                if "cell" in serialized and isinstance(serialized["cell"], dict):
                    cell_params = []
                    for param in ["a", "b", "c", "alpha", "beta", "gamma"]:
                        if param in serialized["cell"]:
                            cell_params.append(serialized["cell"][param])
                    
                    if len(cell_params) == 6:
                        result["cell_tensor"] = torch.tensor(
                            cell_params, device=self.device, dtype=torch.float32, 
                            requires_grad=True
                        )
            
            # Extract any other numerical properties that could be useful as tensors
            for key, value in serialized.items():
                if isinstance(value, (list, tuple)) and all(isinstance(x, (int, float)) for x in value):
                    # Convert numerical lists to tensors
                    result[f"{key}_tensor"] = torch.tensor(
                        value, device=self.device, dtype=torch.float32,
                        requires_grad=True
                    )
                elif isinstance(value, dict) and key not in ["cell"]:  # Skip cell as we handled it above
                    # Look for numerical values in dictionaries
                    tensor_values = []
                    tensor_keys = []
                    for k, v in value.items():
                        if isinstance(v, (int, float)):
                            tensor_keys.append(k)
                            tensor_values.append(v)
                    
                    if tensor_values:
                        result[f"{key}_tensor"] = torch.tensor(
                            tensor_values, device=self.device, dtype=torch.float32,
                            requires_grad=True
                        )
                        result[f"{key}_keys"] = tensor_keys
            
            return result
        except Exception as e:
            print(f"Warning: Failed to convert Gemmi object to tensors: {e}")
            return {
                "_gemmi_type": "ConversionFailed",
                "_error": str(e),
                # Create an empty tensor to avoid downstream errors
                "empty_tensor": torch.tensor([], device=self.device, dtype=torch.float32)
            }
    
    def array_to_tensor(self, array: np.ndarray, requires_grad: bool = True, dtype=None) -> torch.Tensor:
        """
        Convert a NumPy array to a PyTorch tensor with gradient support.
        
        Args:
            array: NumPy array to convert. Can be of any shape or dtype.
            requires_grad: Whether the tensor requires gradients for backpropagation.
                           Only applied to floating point tensors.
            dtype: Data type for the tensor. If None, uses torch.float32.
            
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
            
        if dtype is None:
            dtype = torch.float32  # Use consistent default dtype
            
        if array.size == 0:  # Handle empty arrays
            tensor = torch.from_numpy(array.copy()).to(dtype=dtype, device=self.device)
        else:
            # Use tensor constructor to specify dtype explicitly
            tensor = torch.tensor(array, dtype=dtype, device=self.device)
        
        # Only set requires_grad for floating point tensors
        if requires_grad and tensor.dtype.is_floating_point:
            tensor.requires_grad_(True)
            
        return tensor
        
    def convert_state_dict(self, state_dict, requires_grad=True):
        """
        Convert a state dictionary from NumPy arrays to PyTorch tensors.
        
        Args:
            state_dict: Dictionary with attribute name -> value mappings from Logger.loadStateLog()
            requires_grad: Whether tensors should require gradients
            
        Returns:
            Dictionary with same keys but values converted to tensors
            
        Raises:
            ValueError: If state_dict is None or not a dictionary
        """
        if state_dict is None:
            raise ValueError("Cannot convert None state dictionary")
        if not isinstance(state_dict, dict):
            raise ValueError(f"Expected dictionary, got {type(state_dict)}")
            
        result = {}
        for key, value in state_dict.items():
            if isinstance(value, np.ndarray):
                # Handle boolean arrays specially to preserve dtype
                if value.dtype == np.bool_:
                    result[key] = torch.tensor(value, dtype=torch.bool, device=self.device)
                else:
                    result[key] = self.array_to_tensor(value, requires_grad=requires_grad)
            elif isinstance(value, dict):
                result[key] = self.convert_state_dict(value, requires_grad=requires_grad)
            elif isinstance(value, list) and all(isinstance(x, np.ndarray) for x in value if isinstance(x, np.ndarray)):
                result[key] = [
                    self.array_to_tensor(x, requires_grad=requires_grad) if isinstance(x, np.ndarray) else x
                    for x in value
                ]
            elif isinstance(value, complex):
                # Handle complex scalar values
                real = torch.tensor(value.real, device=self.device, dtype=torch.float32)
                imag = torch.tensor(value.imag, device=self.device, dtype=torch.float32)
                result[key] = torch.complex(real, imag)
            else:
                # Pass through other values unchanged
                result[key] = value
        return result
        
    def convert_serialized_object(self, data, requires_grad=True):
        """
        Handle conversion of objects that might have been serialized by Logger.
        
        Args:
            data: Data that might be binary-serialized or a regular value
            requires_grad: Whether tensors should require gradients
            
        Returns:
            Converted data with PyTorch tensors
        """
        # If the data is a numpy array, convert it directly
        if isinstance(data, np.ndarray):
            return self.array_to_tensor(data, requires_grad=requires_grad)
        
        # If it's a dictionary, recursively convert its values
        if isinstance(data, dict):
            return self.convert_state_dict(data, requires_grad=requires_grad)
        
        # If it's a list, convert array elements
        if isinstance(data, list):
            return [
                self.array_to_tensor(x, requires_grad=requires_grad) 
                if isinstance(x, np.ndarray) else self.convert_serialized_object(x, requires_grad)
                for x in data
            ]
        
        # Handle other types
        return data
    
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
                    requires_grad: bool = True, dtype=torch.float32) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        """
        Convert a grid of q-vectors to PyTorch tensor with gradient support.
        
        Args:
            q_grid: NumPy array of shape (n_points, 3) with q-vectors
            map_shape: Tuple with 3D map shape (dim_h, dim_k, dim_l)
            requires_grad: Whether the tensor requires gradients for backpropagation
            dtype: Data type for the tensor (default: torch.float32)
            
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
            q_grid_tensor = torch.zeros((0, 3), dtype=dtype, device=self.device)
            q_grid_tensor.requires_grad_(requires_grad)
            return q_grid_tensor, map_shape
            
        # Convert to tensor with consistent dtype
        q_grid_tensor = torch.tensor(q_grid, dtype=dtype, device=self.device)
        
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
        if sym_ops is None:
            return {}
        
        return {
            key: torch.tensor(matrix, device=self.device, dtype=torch.float32)
            for key, matrix in sym_ops.items()
        }

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
        
    def convert_state_to_numpy(self, state_dict):
        """
        Convert a state dictionary from PyTorch tensors to NumPy arrays.
        
        Args:
            state_dict: Dictionary with attribute name -> tensor mappings
            
        Returns:
            Dictionary with same keys but values converted to NumPy arrays
            
        Raises:
            ValueError: If state_dict is None or not a dictionary
        """
        if state_dict is None:
            raise ValueError("Cannot convert None state dictionary")
        if not isinstance(state_dict, dict):
            raise ValueError(f"Expected dictionary, got {type(state_dict)}")
            
        result = {}
        for key, value in state_dict.items():
            if isinstance(value, torch.Tensor):
                # Handle complex tensors specially
                if torch.is_complex(value):
                    if value.numel() == 1:  # Single complex value
                        result[key] = np.complex128(complex(value.real.item(), value.imag.item()))
                    else:
                        result[key] = self.tensor_to_array(value)
                else:
                    result[key] = self.tensor_to_array(value)
            elif isinstance(value, dict):
                result[key] = self.convert_state_to_numpy(value)
            elif isinstance(value, list):
                # Convert lists with possible tensor elements
                result[key] = [
                    self.tensor_to_array(x) if isinstance(x, torch.Tensor) else x
                    for x in value
                ]
            else:
                # Pass through other values unchanged
                result[key] = value
        return result
        
    def extract_object_state(self, obj, include_private=False):
        """
        Extract an object's state as a dictionary and convert to NumPy arrays.
        
        Args:
            obj: PyTorch object to extract state from
            include_private: Whether to include private attributes (starting with '_')
            
        Returns:
            Dictionary with attribute names mapped to NumPy array values
            
        Raises:
            ValueError: If obj is None
        """
        if obj is None:
            raise ValueError("Cannot extract state from None object")
            
        state = {}
        for attr_name in dir(obj):
            # Skip private attributes unless explicitly requested
            if attr_name.startswith('_') and not include_private:
                continue
                
            # Skip callable attributes (methods)
            try:
                attr = getattr(obj, attr_name)
                if callable(attr):
                    continue
                    
                # Convert attribute based on type
                if isinstance(attr, torch.Tensor):
                    state[attr_name] = self.tensor_to_array(attr)
                elif isinstance(attr, dict):
                    state[attr_name] = self.convert_state_to_numpy(attr)
                elif isinstance(attr, list):
                    # Convert lists containing tensors
                    state[attr_name] = [
                        self.tensor_to_array(x) if isinstance(x, torch.Tensor) else x
                        for x in attr
                    ]
                else:
                    # Store other attributes directly
                    state[attr_name] = attr
            except (AttributeError, RuntimeError):
                # Skip attributes that can't be accessed
                continue
                
        return state
    
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
        
    def initialize_from_state(self, torch_class, state_data, device=None):
        """
        Initialize a PyTorch model from state data loaded from Logger.
        
        Args:
            torch_class: PyTorch model class to instantiate
            state_data: Dictionary with state data from Logger.loadStateLog()
            device: Device to place tensors on
            
        Returns:
            Initialized instance of torch_class
            
        Raises:
            ValueError: If torch_class or state_data is None
        """
        if torch_class is None:
            raise ValueError("torch_class cannot be None")
        if state_data is None:
            raise ValueError("state_data cannot be None")
            
        if device is None:
            device = self.device
            
        # Create empty instance
        model = torch_class.__new__(torch_class)
        
        # Convert state arrays to tensors
        tensor_state = self.pdb_to_tensor.convert_state_dict(state_data)
        
        # Set attributes
        for key, value in tensor_state.items():
            setattr(model, key, value)
            
        return model
        
    def convert_state_for_comparison(self, torch_model):
        """
        Extract and convert model state for comparison with ground truth.
        
        Args:
            torch_model: PyTorch model instance
            
        Returns:
            Dictionary with state converted to NumPy for comparison
            
        Raises:
            ValueError: If torch_model is None
        """
        if torch_model is None:
            raise ValueError("torch_model cannot be None")
            
        state = {}
        for key, value in torch_model.__dict__.items():
            if key.startswith('_'):
                continue
            if callable(value):
                continue
            
            if isinstance(value, torch.Tensor):
                state[key] = self.tensor_to_numpy.tensor_to_array(value)
            elif isinstance(value, dict):
                state[key] = self.tensor_to_numpy.convert_state_to_numpy(value)
            elif isinstance(value, list):
                state[key] = [
                    self.tensor_to_numpy.tensor_to_array(x) if isinstance(x, torch.Tensor) else x
                    for x in value
                ]
            else:
                state[key] = value
            
        return state
        
    def initialize_one_phonon_from_state(self, torch_class, state_data, device=None):
        """
        Initialize a PyTorch OnePhonon model from state data with special handling.
        
        Args:
            torch_class: PyTorch OnePhonon class
            state_data: Dictionary with state data from Logger.loadStateLog()
            device: Device to place tensors on
            
        Returns:
            Initialized OnePhonon instance
            
        Raises:
            ValueError: If torch_class or state_data is None
        """
        # Basic initialization
        model = self.initialize_from_state(torch_class, state_data, device)
        
        # Handle complex tensors in OnePhonon model
        # V and Winv need to be complex tensors for phonon calculations
        complex_attrs = ['V', 'Winv']
        for attr_name in complex_attrs:
            if hasattr(model, attr_name):
                attr = getattr(model, attr_name)
                if isinstance(attr, torch.Tensor) and not torch.is_complex(attr):
                    # Convert real tensor to complex by adding zero imaginary part
                    # Ensure we preserve requires_grad
                    requires_grad = attr.requires_grad
                    complex_attr = torch.complex(attr, torch.zeros_like(attr))
                    if requires_grad:
                        complex_attr.requires_grad_(True)
                    setattr(model, attr_name, complex_attr)
        
        return model
    
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
