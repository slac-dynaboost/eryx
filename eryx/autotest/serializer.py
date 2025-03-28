# spec
#module DebuggingSystem {
#
#    interface Serializer {
#        """
#        Serializes Python objects to a binary format using pickle.
#
#        Preconditions:
#        - `input_data` must be a picklable Python object.
#
#        Postconditions:
#        - Returns the serialized binary data of the input object.
#        - Raises ValueError if the input data is not picklable.
#        """
#        bytes serialize(Any input_data);
#
#        """
#        Deserializes Python objects from a binary format using pickle.
#
#        Preconditions:
#        - `serialized_data` must be a valid pickle-serialized binary string.
#
#        Postconditions:
#        - Returns the deserialized Python object.
#        - Raises ValueError if the binary data could not be deserialized.
#        """
#        Any deserialize(bytes serialized_data);
#    };

import doctest
import pickle
import numpy as np
import io
import re
from typing import Any, List, Dict, Optional, Union, Tuple, Type

# Try to import gemmi, but don't fail if it's not available
try:
    import gemmi
    GEMMI_AVAILABLE = True
except ImportError:
    GEMMI_AVAILABLE = False

class GemmiSerializer:
    """
    Specialized serializer for Gemmi objects.
    
    This class provides methods to convert Gemmi objects (Structure, Model, Chain, etc.)
    to and from serializable dictionaries that can be used with pickle.
    """
    
    def __init__(self):
        """Initialize the GemmiSerializer with type mappings."""
        # Common Gemmi types to detect
        self.gemmi_types = ["Structure", "Model", "Chain", "Residue", "Atom", "Cell", "UnitCell", "SpaceGroup"]
        
        # Map of Gemmi types to serialization functions
        self.serializers = {
            "Structure": self.serialize_structure,
            "Cell": self.serialize_cell,
            "UnitCell": self.serialize_cell,
            "Model": self.serialize_model,
            "Chain": self.serialize_chain,
            "Residue": self.serialize_residue,
            "Atom": self.serialize_atom,
            "SpaceGroup": self.serialize_spacegroup
        }
        
        # Map of Gemmi types to deserialization functions
        self.deserializers = {
            "Structure": self.deserialize_structure,
            "Cell": self.deserialize_cell,
            "UnitCell": self.deserialize_cell,
            "Model": self.deserialize_model,
            "Chain": self.deserialize_chain,
            "Residue": self.deserialize_residue,
            "Atom": self.deserialize_atom,
            "SpaceGroup": self.deserialize_spacegroup
        }
        
        # Flag to indicate if gemmi is available
        global GEMMI_AVAILABLE
    
    def is_gemmi_object(self, obj: Any) -> bool:
        """
        Check if an object is from the gemmi module.
        
        Args:
            obj: Object to check
            
        Returns:
            True if it's a Gemmi object, False otherwise
        """
        if obj is None:
            return False
            
        # Check module name
        module_name = getattr(obj.__class__, "__module__", "")
        if module_name.startswith("gemmi"):
            return True
            
        # Check class name
        class_name = obj.__class__.__name__
        if class_name in self.gemmi_types:
            # Additional validation based on attributes
            if class_name == "Structure" and self._is_valid_structure(obj):
                return True
            elif (class_name == "Cell" or class_name == "UnitCell") and self._is_valid_cell(obj):
                return True
            elif class_name == "Model" and self._is_valid_model(obj):
                return True
            # Add more validations as needed
        
        # Check for objects that contain Gemmi objects
        if hasattr(obj, 'structure') and self.is_gemmi_object(obj.structure):
            return True
            
        # Check for AtomicModel objects from eryx.pdb
        if module_name == "eryx.pdb" and class_name == "AtomicModel":
            return True
            
        # Check for GaussianNetworkModel objects from eryx.pdb
        if module_name == "eryx.pdb" and class_name == "GaussianNetworkModel":
            return True
            
        # Check for Crystal objects from eryx.pdb
        if module_name == "eryx.pdb" and class_name == "Crystal":
            return True
        
        return False
    
    def get_gemmi_type(self, obj: Any) -> Optional[str]:
        """
        Get the Gemmi type name for an object.
        
        Args:
            obj: Gemmi object
            
        Returns:
            Type name or None if not a recognized Gemmi type
        """
        if not self.is_gemmi_object(obj):
            return None
            
        module_name = getattr(obj.__class__, "__module__", "")
        class_name = obj.__class__.__name__
        
        # Handle eryx.pdb objects
        if module_name == "eryx.pdb":
            return class_name
        
        # Handle special cases
        if class_name == "UnitCell":
            return "Cell"
            
        return class_name if class_name in self.gemmi_types else None
    
    def serialize_gemmi_object(self, obj: Any) -> Dict[str, Any]:
        """
        Convert a Gemmi object to a serializable dictionary.
        
        Args:
            obj: Gemmi object to serialize
            
        Returns:
            Dictionary representation with _gemmi_type metadata
            
        Raises:
            ValueError: If object is not a recognized Gemmi type
        """
        try:
            # Handle eryx.pdb objects specially
            module_name = getattr(obj.__class__, "__module__", "")
            class_name = obj.__class__.__name__
            
            if module_name == "eryx.pdb":
                if class_name == "AtomicModel":
                    return self.serialize_atomic_model(obj)
                elif class_name == "GaussianNetworkModel":
                    return self.serialize_gnm(obj)
                elif class_name == "Crystal":
                    return self.serialize_crystal(obj)
            
            gemmi_type = self.get_gemmi_type(obj)
            if not gemmi_type:
                # Create a generic representation for unrecognized Gemmi objects
                result = {
                    "_gemmi_type": "Unknown",
                    "_module": module_name,
                    "_class": class_name,
                    "_repr": repr(obj)[:1000]  # Limit length of representation
                }
                
                # Try to extract common attributes
                for attr in ["name", "id", "serial", "number"]:
                    if hasattr(obj, attr):
                        try:
                            result[attr] = getattr(obj, attr)
                        except Exception:
                            pass
                
                return result
                
            # Get the appropriate serializer function
            serializer = self.serializers.get(gemmi_type)
            if not serializer:
                # Create a basic representation if no specific serializer is available
                result = {
                    "_gemmi_type": gemmi_type,
                    "_module": module_name,
                    "_class": class_name,
                    "_repr": repr(obj)[:1000]  # Limit length of representation
                }
                
                # Try to extract common attributes
                for attr in ["name", "id", "serial", "number"]:
                    if hasattr(obj, attr):
                        try:
                            result[attr] = getattr(obj, attr)
                        except Exception:
                            pass
                
                return result
                
            # Call the serializer and add type metadata
            result = serializer(obj)
            result["_gemmi_type"] = gemmi_type
            
            return result
        except Exception as e:
            # Return a minimal representation if serialization fails
            return {
                "_gemmi_type": "SerializationFailed",
                "_error": str(e),
                "_repr": repr(obj)[:1000] if obj is not None else "None"
            }
    
    def deserialize_gemmi_object(self, data: Dict[str, Any]) -> Any:
        """
        Convert a serialized dictionary back to a Gemmi object.
        
        Args:
            data: Dictionary with _gemmi_type and serialized data
            
        Returns:
            Reconstructed Gemmi object or placeholder dictionary
            
        Raises:
            ValueError: If data doesn't contain _gemmi_type or type is not recognized
        """
        if not isinstance(data, dict):
            raise ValueError(f"Expected dictionary, got {type(data)}")
            
        gemmi_type = data.get("_gemmi_type")
        if not gemmi_type:
            raise ValueError("Missing _gemmi_type in serialized data")
            
        # Handle eryx.pdb objects - always return the dictionary
        if gemmi_type in ["AtomicModel", "GaussianNetworkModel", "Crystal"]:
            return data
            
        # Check if gemmi is available
        if not GEMMI_AVAILABLE:
            # Return the dictionary as a fallback
            return data
            
        # Get the appropriate deserializer
        deserializer = self.deserializers.get(gemmi_type)
        if not deserializer:
            raise ValueError(f"No deserializer available for Gemmi type: {gemmi_type}")
            
        try:
            # Call the deserializer
            return deserializer(data)
        except Exception as e:
            # If deserialization fails, return the dictionary as fallback
            print(f"Warning: Failed to deserialize Gemmi {gemmi_type}: {str(e)}")
            return data
    
    def serialize_structure(self, structure: Any) -> Dict[str, Any]:
        """
        Serialize a gemmi.Structure to a dictionary.
        
        Args:
            structure: gemmi.Structure object
            
        Returns:
            Dictionary with serialized structure data
        """
        result = {
            "name": getattr(structure, "name", ""),
            "cell": self.serialize_cell(structure.cell),
            "spacegroup": getattr(structure, "spacegroup_hm", ""),
            "models": []
        }
        
        # Serialize each model
        for model in structure:
            result["models"].append(self.serialize_model(model))
            
        return result
    
    def deserialize_structure(self, data: Dict[str, Any]) -> Any:
        """
        Deserialize a dictionary to a gemmi.Structure.
        
        Args:
            data: Dictionary with serialized structure data
            
        Returns:
            gemmi.Structure object
        """
        structure = gemmi.Structure()
        
        # Set basic properties
        structure.name = data.get("name", "")
        
        # Set cell if available
        if "cell" in data:
            cell_data = data["cell"]
            if isinstance(cell_data, dict):
                cell = self.deserialize_cell(cell_data)
                structure.cell = cell
        
        # Set spacegroup if available
        if "spacegroup" in data:
            try:
                structure.spacegroup_hm = data["spacegroup"]
            except Exception:
                pass  # Ignore spacegroup errors
        
        # Add models
        for model_data in data.get("models", []):
            try:
                model = self.deserialize_model(model_data)
                structure.add_model(model)
            except Exception as e:
                print(f"Warning: Failed to deserialize model: {str(e)}")
        
        return structure
    
    def serialize_cell(self, cell: Any) -> Dict[str, Any]:
        """
        Serialize a gemmi.UnitCell to a dictionary.
        
        Args:
            cell: gemmi.UnitCell object
            
        Returns:
            Dictionary with cell parameters
        """
        return {
            "a": getattr(cell, "a", 0.0),
            "b": getattr(cell, "b", 0.0),
            "c": getattr(cell, "c", 0.0),
            "alpha": getattr(cell, "alpha", 0.0),
            "beta": getattr(cell, "beta", 0.0),
            "gamma": getattr(cell, "gamma", 0.0)
        }
    
    def deserialize_cell(self, data: Dict[str, Any]) -> Any:
        """
        Deserialize a dictionary to a gemmi.UnitCell.
        
        Args:
            data: Dictionary with cell parameters
            
        Returns:
            gemmi.UnitCell object
        """
        a = data.get("a", 0.0)
        b = data.get("b", 0.0)
        c = data.get("c", 0.0)
        alpha = data.get("alpha", 0.0)
        beta = data.get("beta", 0.0)
        gamma = data.get("gamma", 0.0)
        
        return gemmi.UnitCell(a, b, c, alpha, beta, gamma)
    
    def serialize_model(self, model: Any) -> Dict[str, Any]:
        """
        Serialize a gemmi.Model to a dictionary.
        
        Args:
            model: gemmi.Model object
            
        Returns:
            Dictionary with serialized model data
        """
        result = {
            "name": getattr(model, "name", ""),
            "chains": []
        }
        
        # Serialize each chain
        for chain in model:
            result["chains"].append(self.serialize_chain(chain))
            
        return result
    
    def deserialize_model(self, data: Dict[str, Any]) -> Any:
        """
        Deserialize a dictionary to a gemmi.Model.
        
        Args:
            data: Dictionary with serialized model data
            
        Returns:
            gemmi.Model object
        """
        model = gemmi.Model(data.get("name", ""))
        
        # Add chains
        for chain_data in data.get("chains", []):
            try:
                chain = self.deserialize_chain(chain_data)
                model.add_chain(chain)
            except Exception as e:
                print(f"Warning: Failed to deserialize chain: {str(e)}")
        
        return model
    
    def serialize_chain(self, chain: Any) -> Dict[str, Any]:
        """
        Serialize a gemmi.Chain to a dictionary.
        
        Args:
            chain: gemmi.Chain object
            
        Returns:
            Dictionary with serialized chain data
        """
        result = {
            "name": getattr(chain, "name", ""),
            "residues": []
        }
        
        # Serialize each residue
        for residue in chain:
            result["residues"].append(self.serialize_residue(residue))
            
        return result
    
    def deserialize_chain(self, data: Dict[str, Any]) -> Any:
        """
        Deserialize a dictionary to a gemmi.Chain.
        
        Args:
            data: Dictionary with serialized chain data
            
        Returns:
            gemmi.Chain object
        """
        chain = gemmi.Chain(data.get("name", ""))
        
        # Add residues
        for residue_data in data.get("residues", []):
            try:
                residue = self.deserialize_residue(residue_data)
                chain.add_residue(residue)
            except Exception as e:
                print(f"Warning: Failed to deserialize residue: {str(e)}")
        
        return chain
    
    def serialize_residue(self, residue: Any) -> Dict[str, Any]:
        """
        Serialize a gemmi.Residue to a dictionary.
        
        Args:
            residue: gemmi.Residue object
            
        Returns:
            Dictionary with serialized residue data
        """
        result = {
            "name": getattr(residue, "name", ""),
            "seqid": getattr(residue, "seqid", {}).num if hasattr(residue, "seqid") else 0,
            "atoms": []
        }
        
        # Serialize each atom
        for atom in residue:
            result["atoms"].append(self.serialize_atom(atom))
            
        return result
    
    def deserialize_residue(self, data: Dict[str, Any]) -> Any:
        """
        Deserialize a dictionary to a gemmi.Residue.
        
        Args:
            data: Dictionary with serialized residue data
            
        Returns:
            gemmi.Residue object
        """
        residue = gemmi.Residue()
        residue.name = data.get("name", "")
        
        # Set seqid if available
        seqid = gemmi.SeqId()
        seqid.num = data.get("seqid", 0)
        residue.seqid = seqid
        
        # Add atoms
        for atom_data in data.get("atoms", []):
            try:
                atom = self.deserialize_atom(atom_data)
                residue.add_atom(atom)
            except Exception as e:
                print(f"Warning: Failed to deserialize atom: {str(e)}")
        
        return residue
    
    def serialize_atom(self, atom: Any) -> Dict[str, Any]:
        """
        Serialize a gemmi.Atom to a dictionary.
        
        Args:
            atom: gemmi.Atom object
            
        Returns:
            Dictionary with serialized atom data
        """
        return {
            "name": getattr(atom, "name", ""),
            "element": getattr(atom, "element", {}).name if hasattr(atom, "element") else "",
            "pos": [
                getattr(atom, "pos", [0, 0, 0])[0],
                getattr(atom, "pos", [0, 0, 0])[1],
                getattr(atom, "pos", [0, 0, 0])[2]
            ],
            "b_iso": getattr(atom, "b_iso", 0.0),
            "occ": getattr(atom, "occ", 1.0)
        }
    
    def deserialize_atom(self, data: Dict[str, Any]) -> Any:
        """
        Deserialize a dictionary to a gemmi.Atom.
        
        Args:
            data: Dictionary with serialized atom data
            
        Returns:
            gemmi.Atom object
        """
        atom = gemmi.Atom()
        atom.name = data.get("name", "")
        
        # Set element if available
        element_name = data.get("element", "")
        if element_name:
            atom.element = gemmi.Element(element_name)
        
        # Set position
        pos = data.get("pos", [0, 0, 0])
        atom.pos = gemmi.Position(pos[0], pos[1], pos[2])
        
        # Set other properties
        atom.b_iso = data.get("b_iso", 0.0)
        atom.occ = data.get("occ", 1.0)
        
        return atom
    
    def serialize_spacegroup(self, spacegroup: Any) -> Dict[str, Any]:
        """
        Serialize a gemmi.SpaceGroup to a dictionary.
        
        Args:
            spacegroup: gemmi.SpaceGroup object
            
        Returns:
            Dictionary with serialized spacegroup data
        """
        return {
            "name": getattr(spacegroup, "name", ""),
            "hm": getattr(spacegroup, "hm", ""),
            "hall": getattr(spacegroup, "hall", ""),
            "number": getattr(spacegroup, "number", 0)
        }
    
    def deserialize_spacegroup(self, data: Dict[str, Any]) -> Any:
        """
        Deserialize a dictionary to a gemmi.SpaceGroup.
        
        Args:
            data: Dictionary with serialized spacegroup data
            
        Returns:
            gemmi.SpaceGroup object
        """
        # Try to create from name or number
        if "hm" in data and data["hm"]:
            return gemmi.find_spacegroup_by_name(data["hm"])
        elif "number" in data and data["number"] > 0:
            return gemmi.find_spacegroup_by_number(data["number"])
        else:
            return gemmi.find_spacegroup_by_name("P 1")  # Default to P1
    
    def _has_attributes(self, obj: Any, attrs: List[str]) -> bool:
        """Check if object has all required attributes."""
        return all(hasattr(obj, attr) for attr in attrs)
    
    def _is_valid_structure(self, obj: Any) -> bool:
        """Check if object appears to be a gemmi.Structure."""
        return (self._has_attributes(obj, ["cell", "spacegroup_hm"]) and
                hasattr(obj, "__iter__") and  # Has models
                self._has_attributes(obj, ["name", "add_model"]))
    
    def _is_valid_cell(self, obj: Any) -> bool:
        """Check if object appears to be a gemmi.UnitCell."""
        return self._has_attributes(obj, ["a", "b", "c", "alpha", "beta", "gamma"])
    
    def _is_valid_model(self, obj: Any) -> bool:
        """Check if object appears to be a gemmi.Model."""
        return (self._has_attributes(obj, ["name"]) and
                hasattr(obj, "__iter__") and  # Has chains
                self._has_attributes(obj, ["add_chain"]))
                
    def serialize_atomic_model(self, model: Any) -> Dict[str, Any]:
        """
        Serialize an AtomicModel to a dictionary.
        
        Args:
            model: eryx.pdb.AtomicModel object
            
        Returns:
            Dictionary with serialized model data
        """
        result = {
            "_gemmi_type": "AtomicModel",
            "_module": "eryx.pdb"
        }
        
        # Extract basic properties
        for attr in ["n_asu", "n_conf", "space_group"]:
            if hasattr(model, attr):
                result[attr] = getattr(model, attr)
        
        # Handle numpy arrays
        for array_attr in ["xyz", "ff_a", "ff_b", "ff_c", "adp", "cell", "A_inv", "unit_cell_axes"]:
            if hasattr(model, array_attr):
                attr_value = getattr(model, array_attr)
                if attr_value is not None:
                    # Store shape and dtype info for reconstruction
                    result[array_attr] = {
                        "shape": attr_value.shape,
                        "dtype": str(attr_value.dtype)
                    }
        
        # Handle structure if present
        if hasattr(model, "structure") and model.structure is not None:
            try:
                result["structure"] = self.serialize_structure(model.structure)
            except Exception as e:
                print(f"Warning: Failed to serialize structure: {e}")
        
        return result
    
    def serialize_gnm(self, gnm: Any) -> Dict[str, Any]:
        """
        Serialize a GaussianNetworkModel to a dictionary.
        
        Args:
            gnm: eryx.pdb.GaussianNetworkModel object
            
        Returns:
            Dictionary with serialized GNM data
        """
        result = {
            "_gemmi_type": "GaussianNetworkModel",
            "_module": "eryx.pdb"
        }
        
        # Extract basic properties
        for attr in ["enm_cutoff", "gamma_intra", "gamma_inter"]:
            if hasattr(gnm, attr):
                result[attr] = getattr(gnm, attr)
        
        # Handle numpy arrays
        for array_attr in ["gamma"]:
            if hasattr(gnm, array_attr):
                attr_value = getattr(gnm, array_attr)
                if attr_value is not None:
                    # Store shape and dtype info for reconstruction
                    result[array_attr] = {
                        "shape": attr_value.shape,
                        "dtype": str(attr_value.dtype)
                    }
        
        # Handle crystal if present
        if hasattr(gnm, "crystal") and gnm.crystal is not None:
            try:
                result["crystal"] = self.serialize_crystal(gnm.crystal)
            except Exception as e:
                print(f"Warning: Failed to serialize crystal: {e}")
        
        return result
    
    def serialize_crystal(self, crystal: Any) -> Dict[str, Any]:
        """
        Serialize a Crystal to a dictionary.
        
        Args:
            crystal: eryx.pdb.Crystal object
            
        Returns:
            Dictionary with serialized Crystal data
        """
        result = {
            "_gemmi_type": "Crystal",
            "_module": "eryx.pdb"
        }
        
        # Extract basic properties
        for attr in ["n_cell"]:
            if hasattr(crystal, attr):
                result[attr] = getattr(crystal, attr)
        
        # Handle model if present
        if hasattr(crystal, "model") and crystal.model is not None:
            try:
                result["model"] = self.serialize_atomic_model(crystal.model)
            except Exception as e:
                print(f"Warning: Failed to serialize model: {e}")
        
        return result


class Serializer:
    def __init__(self):
        """Initialize the serializer with specialized handlers."""
        # Initialize GemmiSerializer
        self.gemmi_serializer = GemmiSerializer()
    
    def serialize(self, input_data: Any) -> bytes:
        """
        Serializes Python objects to a binary format using pickle.

        Preconditions:
        - `input_data` must be a picklable Python object.

        Postconditions:
        - Returns the serialized binary data of the input object.
        - Raises ValueError if the input data is not picklable.

        >>> s = Serializer()
        >>> data = {'key': 'value'}
        >>> serialized_data = s.serialize(data)
        >>> type(serialized_data)
        <class 'bytes'>
        >>> deserialized_data = s.deserialize(serialized_data)
        >>> deserialized_data == data
        True
        >>> s.serialize(lambda x: x)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: Input data is not picklable
        >>> s.deserialize(b'not a pickle')  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: Could not deserialize the binary data
        """
        try:
            # Check if it's a Gemmi object
            if hasattr(self, 'gemmi_serializer') and self.gemmi_serializer.is_gemmi_object(input_data):
                # Convert Gemmi object to serializable dictionary
                serialized_dict = self.gemmi_serializer.serialize_gemmi_object(input_data)
                # Then pickle the dictionary
                return pickle.dumps({
                    "_serialized_gemmi": True,
                    "data": serialized_dict
                })
            
            # Handle special cases for better serialization
            if hasattr(input_data, '__torch_function__'):
                # This is likely a PyTorch tensor
                try:
                    import torch
                    if isinstance(input_data, torch.Tensor):
                        # Convert PyTorch tensor to numpy for serialization
                        return pickle.dumps({
                            '_tensor_data': input_data.detach().cpu().numpy(),
                            '_tensor_type': 'torch.Tensor',
                            '_tensor_dtype': str(input_data.dtype),
                            '_tensor_requires_grad': input_data.requires_grad,
                            '_tensor_device': str(input_data.device)
                        })
                except ImportError:
                    pass  # Fall back to standard pickle if torch not available
            
            # Handle numpy arrays with special metadata
            if isinstance(input_data, np.ndarray):
                buffer = io.BytesIO()
                np.save(buffer, input_data)
                # Store the actual array data, not just metadata
                return pickle.dumps({
                    '_array_data': buffer.getvalue(),
                    '_array_type': 'numpy.ndarray',
                    '_array_dtype': str(input_data.dtype),
                    '_array_shape': input_data.shape,
                    # Add the actual array data as a list for better serialization
                    '_array_values': input_data.tolist()
                })
            
            # Standard pickle for other types
            return pickle.dumps(input_data)
        except (pickle.PicklingError, AttributeError, TypeError) as e:
            # Handle unserializable objects gracefully
            return self._serialize_unserializable(input_data, str(e))

    def _serialize_unserializable(self, obj: Any, error_msg: str) -> bytes:
        """
        Create a serializable placeholder for unserializable objects.
        
        Args:
            obj: The object that couldn't be serialized
            error_msg: The error message from the serialization attempt
            
        Returns:
            Serialized bytes of a placeholder dictionary
        """
        # Get type information
        type_name = type(obj).__name__
        module_name = getattr(type(obj), "__module__", "unknown")
        full_type = f"{module_name}.{type_name}"
        
        # Create base placeholder
        placeholder = {
            "__unserializable__": True,
            "__type__": full_type,
            "__error__": error_msg,
            "__repr__": repr(obj)[:1000]  # Limit length of representation
        }
        
        # Check if it's a Gemmi object
        if self._is_gemmi_object(obj):
            placeholder["__gemmi_type__"] = True
            # Extract useful information if possible
            gemmi_info = self._extract_gemmi_info(obj)
            if gemmi_info:
                placeholder["__gemmi_info__"] = gemmi_info
        
        # Handle collections containing unserializable objects
        if isinstance(obj, dict):
            # Process dictionary items
            serialized_dict = {}
            for k, v in obj.items():
                try:
                    # Try to serialize the key
                    key = str(k)  # Convert key to string if not serializable
                    
                    # Try to serialize the value
                    try:
                        serialized_dict[key] = self.serialize(v)
                    except Exception as e:
                        # Create placeholder for unserializable value
                        serialized_dict[key] = self._serialize_unserializable(v, str(e))
                except Exception:
                    # Skip items that can't be processed at all
                    continue
            
            placeholder["__items__"] = serialized_dict
            
        elif isinstance(obj, (list, tuple)):
            # Process list/tuple items
            serialized_items = []
            for item in obj:
                try:
                    serialized_items.append(self.serialize(item))
                except Exception as e:
                    # Create placeholder for unserializable item
                    serialized_items.append(self._serialize_unserializable(item, str(e)))
            
            placeholder["__items__"] = serialized_items
            placeholder["__collection_type__"] = "list" if isinstance(obj, list) else "tuple"
        
        # Try to extract common attributes
        try:
            attrs = {}
            for attr_name in dir(obj):
                # Skip methods, private attributes, and special methods
                if attr_name.startswith('_') or callable(getattr(obj, attr_name, None)):
                    continue
                
                try:
                    attr_value = getattr(obj, attr_name)
                    # Only include simple types
                    if isinstance(attr_value, (str, int, float, bool, type(None))):
                        attrs[attr_name] = attr_value
                except Exception:
                    continue
            
            if attrs:
                placeholder["__attributes__"] = attrs
        except Exception:
            # Ignore errors in attribute extraction
            pass
        
        return pickle.dumps(placeholder)
    
    def _is_gemmi_object(self, obj: Any) -> bool:
        """
        Check if an object is from the gemmi module.
        
        Args:
            obj: Object to check
            
        Returns:
            True if it's a Gemmi object, False otherwise
        """
        if obj is None:
            return False
        
        # Check module name
        module_name = getattr(type(obj), "__module__", "")
        if module_name.startswith("gemmi"):
            return True
        
        # Check class name and module path
        type_str = str(type(obj))
        return "gemmi." in type_str
    
    def _extract_gemmi_info(self, obj: Any) -> Dict[str, Any]:
        """
        Extract useful information from a Gemmi object.
        
        Args:
            obj: Gemmi object
            
        Returns:
            Dictionary with extracted properties
        """
        info = {}
        
        # Try to extract common properties
        for attr in ["name", "id", "serial", "number"]:
            if hasattr(obj, attr):
                try:
                    info[attr] = getattr(obj, attr)
                except Exception:
                    pass
        
        # Extract cell parameters if available
        if hasattr(obj, "cell"):
            try:
                cell = obj.cell
                info["cell"] = {
                    "a": getattr(cell, "a", 0.0),
                    "b": getattr(cell, "b", 0.0),
                    "c": getattr(cell, "c", 0.0),
                    "alpha": getattr(cell, "alpha", 0.0),
                    "beta": getattr(cell, "beta", 0.0),
                    "gamma": getattr(cell, "gamma", 0.0)
                }
            except Exception:
                pass
        
        # Extract space group if available
        if hasattr(obj, "spacegroup_hm"):
            try:
                info["spacegroup"] = obj.spacegroup_hm
            except Exception:
                pass
        
        # Extract position if available (for atoms)
        if hasattr(obj, "pos"):
            try:
                pos = obj.pos
                info["position"] = [pos.x, pos.y, pos.z]
            except Exception:
                pass
        
        return info
    
    def deserialize(self, serialized_data: bytes) -> Any:
        """
        Deserializes Python objects from a binary format using pickle.

        Preconditions:
        - `serialized_data` must be a valid pickle-serialized binary string.

        Postconditions:
        - Returns the deserialized Python object.
        - Raises ValueError if the binary data could not be deserialized.

        >>> s = Serializer()
        >>> data = {'key': 'value'}
        >>> serialized_data = s.serialize(data)
        >>> deserialized_data = s.deserialize(serialized_data)
        >>> deserialized_data == data
        True
        >>> s.deserialize(b'not a pickle')  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: Could not deserialize the binary data
        """
        try:
            data = pickle.loads(serialized_data)
            
            # Check if this is a serialized Gemmi object
            if isinstance(data, dict) and data.get("_serialized_gemmi", False):
                # Deserialize Gemmi object if gemmi_serializer is available
                if hasattr(self, 'gemmi_serializer'):
                    return self.gemmi_serializer.deserialize_gemmi_object(data["data"])
                else:
                    # Return the data dictionary if gemmi_serializer is not available
                    return data["data"]
            
            # Handle unserializable object placeholders
            if isinstance(data, dict) and data.get("__unserializable__", False):
                # Return the placeholder dictionary
                return data
            
            # Handle special case for PyTorch tensors
            if isinstance(data, dict) and '_tensor_type' in data and data['_tensor_type'] == 'torch.Tensor':
                try:
                    import torch
                    tensor = torch.tensor(data['_tensor_data'])
                    if data['_tensor_requires_grad']:
                        tensor.requires_grad_(True)
                    # Move to specified device if possible, otherwise keep on CPU
                    try:
                        device_str = data['_tensor_device']
                        if 'cuda' in device_str and torch.cuda.is_available():
                            tensor = tensor.to(device=device_str)
                    except (RuntimeError, ValueError):
                        pass  # Keep on CPU if device transfer fails
                    return tensor
                except ImportError:
                    # Return numpy array if torch not available
                    return data['_tensor_data']
            
            # Handle special case for numpy arrays
            if isinstance(data, dict) and '_array_type' in data and data['_array_type'] == 'numpy.ndarray':
                buffer = io.BytesIO(data['_array_data'])
                return np.load(buffer)
            
            return data
        except (pickle.UnpicklingError, EOFError, AttributeError, ImportError, IndexError) as e:
            raise ValueError(f"Could not deserialize the binary data: {str(e)}")
    
    def serializeState(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Serialize a state dictionary with special handling for complex types.
        
        Args:
            state_dict: Dictionary containing object state
            
        Returns:
            Dictionary with serialized values
        """
        serialized_state = {}
        for key, value in state_dict.items():
            try:
                serialized_state[key] = self.serialize(value)
            except ValueError as e:
                print(f"Warning: Could not serialize {key}: {str(e)}")
                serialized_state[key] = self.serialize(f"<Unserializable: {type(value)}>")
        return serialized_state
    
    def deserializeState(self, serialized_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deserialize a state dictionary with special handling for complex types.
        
        Args:
            serialized_state: Dictionary with serialized values
            
        Returns:
            Dictionary with deserialized values
        """
        state_dict = {}
        for key, value in serialized_state.items():
            if isinstance(value, bytes):
                try:
                    state_dict[key] = self.deserialize(value)
                except ValueError as e:
                    print(f"Warning: Could not deserialize {key}: {str(e)}")
                    state_dict[key] = f"<Undeserializable data>"
            else:
                state_dict[key] = value
        return state_dict

doctest.testmod(verbose=True)

