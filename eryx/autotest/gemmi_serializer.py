# gemmi_serializer.py

import gemmi

class GemmiSerializer:
    """
    Utility for serializing and restoring Gemmi objects to/from Python dictionaries.
    This handles various Gemmi types including Structure, UnitCell, SpaceGroup, and more.
    
    Example usage:
        serializer = GemmiSerializer()
        
        # Serialize any Gemmi object
        my_struct = gemmi.read_structure("example.pdb")
        serialized_dict = serializer.serialize_gemmi(my_struct)
        
        # Deserialize back to Gemmi object
        restored_struct = serializer.deserialize_gemmi(serialized_dict)
        
        # Or use specialized methods for specific types
        cell = gemmi.UnitCell(10, 10, 10, 90, 90, 90)
        cell_dict = serializer.serialize_cell(cell)
        restored_cell = serializer.deserialize_cell(cell_dict)
    """
    
    def __init__(self):
        """Initialize the serializer with type mappings."""
        # Map of Gemmi class names to serialization functions
        self.serializers = {
            "Structure": self.serialize_structure,
            "UnitCell": self.serialize_cell,
            "Cell": self.serialize_cell,
            "SpaceGroup": self.serialize_spacegroup,
            "Model": self.serialize_model,
            "Chain": self.serialize_chain,
            "Residue": self.serialize_residue,
            "Atom": self.serialize_atom,
            "Element": self.serialize_element
        }
        
        # Map of Gemmi type names to deserialization functions
        self.deserializers = {
            "Structure": self.deserialize_structure,
            "UnitCell": self.deserialize_cell,
            "Cell": self.deserialize_cell,
            "SpaceGroup": self.deserialize_spacegroup,
            "Model": self.deserialize_model,
            "Chain": self.deserialize_chain,
            "Residue": self.deserialize_residue,
            "Atom": self.deserialize_atom,
            "Element": self.deserialize_element
        }
    def serialize_gemmi(self, obj) -> dict:
        """
        General method to serialize any Gemmi object.
        
        Args:
            obj: Any Gemmi object
            
        Returns:
            Dictionary representation with type information
            
        Raises:
            ValueError: If object is not a recognized Gemmi type
        """
        try:
            # Get the class name
            class_name = obj.__class__.__name__
            
            # Check if we have a specialized serializer
            if class_name in self.serializers:
                serializer = self.serializers[class_name]
                data = serializer(obj)
                # Add type information if not already present
                if "_gemmi_type" not in data:
                    data["_gemmi_type"] = class_name
                return data
            
            # Fall back to generic serialization
            return self.serialize_generic_gemmi(obj)
            
        except Exception as e:
            # Create a minimal representation if serialization fails
            return {
                "_gemmi_type": "SerializationFailed",
                "_error": str(e),
                "_class": obj.__class__.__name__,
                "_module": obj.__class__.__module__,
                "_repr": repr(obj)[:1000]  # Limit length of representation
            }
    
    def deserialize_gemmi(self, data: dict):
        """
        General method to deserialize any Gemmi object.
        
        Args:
            data: Dictionary with serialized Gemmi data
            
        Returns:
            Reconstructed Gemmi object or dictionary if reconstruction fails
            
        Raises:
            ValueError: If data doesn't contain type information
        """
        if not isinstance(data, dict):
            raise ValueError(f"Expected dictionary, got {type(data)}")
            
        # Get the Gemmi type
        gemmi_type = data.get("_gemmi_type")
        if not gemmi_type:
            # Try legacy format without _gemmi_type
            if "space_group" in data and "cell" in data and "models" in data:
                gemmi_type = "Structure"
            elif all(k in data for k in ["a", "b", "c", "alpha", "beta", "gamma"]):
                gemmi_type = "Cell"
            else:
                raise ValueError("Missing _gemmi_type in serialized data")
        
        # Check if we have a specialized deserializer
        if gemmi_type in self.deserializers:
            try:
                deserializer = self.deserializers[gemmi_type]
                return deserializer(data)
            except Exception as e:
                # Return the data dictionary if deserialization fails
                print(f"Warning: Failed to deserialize Gemmi {gemmi_type}: {str(e)}")
                return data
        
        # Return the data dictionary for unrecognized types
        return data

    def serialize_structure(self, structure: gemmi.Structure) -> dict:
        """
        Convert a Gemmi Structure into a Python dict of primitive types.
        
        The returned dictionary can be JSON/pickle‐serialized, 
        or used in your logging system for debugging/state capture.
        
        Args:
            structure: gemmi.Structure object
            
        Returns:
            Dictionary with serialized structure data
        """
        data = {
            "_gemmi_type": "Structure",
            "name": structure.name
        }
        
        # 1. Cell parameters
        cell = structure.cell
        data["cell"] = self.serialize_cell(cell)
        
        # 2. Space group
        data["space_group"] = structure.spacegroup_hm  # e.g. 'P 1 21 1'
        
        # 3. Models, Chains, Residues, Atoms
        models_list = []
        for model in structure:
            model_dict = self.serialize_model(model)
            models_list.append(model_dict)
        
        data["models"] = models_list
        
        return data

    def deserialize_structure(self, data: dict) -> gemmi.Structure:
        """
        Rebuild a gemmi.Structure from the dictionary produced by serialize_structure().
        
        Args:
            data: Dictionary with serialized structure data
            
        Returns:
            gemmi.Structure object
        """
        structure = gemmi.Structure()
        
        # Set name if available
        structure.name = data.get("name", "")
        
        # 1. Cell
        cell_data = data.get("cell", {})
        if cell_data:
            if isinstance(cell_data, dict):
                # If cell is a nested dictionary, deserialize it
                structure.cell = self.deserialize_cell(cell_data)
            else:
                # Legacy format or direct parameters
                structure.cell = gemmi.UnitCell(
                    cell_data.get("a", 0.0),
                    cell_data.get("b", 0.0),
                    cell_data.get("c", 0.0),
                    cell_data.get("alpha", 90.0),
                    cell_data.get("beta", 90.0),
                    cell_data.get("gamma", 90.0)
                )
        
        # 2. Space group
        space_group = data.get("space_group", None)
        if space_group:
            structure.spacegroup_hm = space_group
        
        # 3. Models, Chains, etc.
        models_list = data.get("models", [])
        for model_dict in models_list:
            if isinstance(model_dict, dict):
                model = self.deserialize_model(model_dict)
                structure.add_model(model)
        
        return structure
    
    def serialize_cell(self, cell) -> dict:
        """
        Serialize a gemmi.UnitCell or gemmi.Cell to a dictionary.
        
        Args:
            cell: gemmi.UnitCell or gemmi.Cell object
            
        Returns:
            Dictionary with cell parameters
        """
        return {
            "_gemmi_type": "Cell",
            "a": getattr(cell, "a", 0.0),
            "b": getattr(cell, "b", 0.0),
            "c": getattr(cell, "c", 0.0),
            "alpha": getattr(cell, "alpha", 90.0),
            "beta": getattr(cell, "beta", 90.0),
            "gamma": getattr(cell, "gamma", 90.0)
        }
    
    def deserialize_cell(self, data: dict):
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
        alpha = data.get("alpha", 90.0)
        beta = data.get("beta", 90.0)
        gamma = data.get("gamma", 90.0)
        
        return gemmi.UnitCell(a, b, c, alpha, beta, gamma)
    
    def serialize_spacegroup(self, spacegroup) -> dict:
        """
        Serialize a gemmi.SpaceGroup to a dictionary.
        
        Args:
            spacegroup: gemmi.SpaceGroup object
            
        Returns:
            Dictionary with serialized spacegroup data
        """
        return {
            "_gemmi_type": "SpaceGroup",
            "name": getattr(spacegroup, "name", ""),
            "hm": getattr(spacegroup, "hm", ""),
            "hall": getattr(spacegroup, "hall", ""),
            "number": getattr(spacegroup, "number", 0),
            "ccp4": getattr(spacegroup, "ccp4", 0)
        }
    
    def deserialize_spacegroup(self, data: dict):
        """
        Deserialize a dictionary to a gemmi.SpaceGroup.
        
        Args:
            data: Dictionary with serialized spacegroup data
            
        Returns:
            gemmi.SpaceGroup object or None if creation fails
        """
        try:
            # Try to create from name or number
            if "hm" in data and data["hm"]:
                return gemmi.find_spacegroup_by_name(data["hm"])
            elif "name" in data and data["name"]:
                return gemmi.find_spacegroup_by_name(data["name"])
            elif "number" in data and data["number"] > 0:
                return gemmi.find_spacegroup_by_number(data["number"])
            elif "ccp4" in data and data["ccp4"] > 0:
                return gemmi.find_spacegroup_by_number(data["ccp4"])
            else:
                return gemmi.find_spacegroup_by_name("P 1")  # Default to P1
        except Exception as e:
            print(f"Warning: Failed to deserialize SpaceGroup: {str(e)}")
            return None

    def serialize_model(self, model) -> dict:
        """
        Serialize a gemmi.Model to a dictionary.
        
        Args:
            model: gemmi.Model object
            
        Returns:
            Dictionary with serialized model data
        """
        result = {
            "_gemmi_type": "Model",
            "name": getattr(model, "name", ""),
            "chains": []
        }
        
        # Serialize each chain
        for chain in model:
            result["chains"].append(self.serialize_chain(chain))
            
        return result
    
    def deserialize_model(self, data: dict):
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
    
    def serialize_chain(self, chain) -> dict:
        """
        Serialize a gemmi.Chain to a dictionary.
        
        Args:
            chain: gemmi.Chain object
            
        Returns:
            Dictionary with serialized chain data
        """
        result = {
            "_gemmi_type": "Chain",
            "name": getattr(chain, "name", ""),
            "residues": []
        }
        
        # Serialize each residue
        for residue in chain:
            result["residues"].append(self.serialize_residue(residue))
            
        return result
    
    def deserialize_chain(self, data: dict):
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
    
    def serialize_residue(self, residue) -> dict:
        """
        Serialize a gemmi.Residue to a dictionary.
        
        Args:
            residue: gemmi.Residue object
            
        Returns:
            Dictionary with serialized residue data
        """
        result = {
            "_gemmi_type": "Residue",
            "name": getattr(residue, "name", ""),
            "seqid_num": getattr(residue, "seqid", {}).num if hasattr(residue, "seqid") else 0,
            "seqid_icode": getattr(residue, "seqid", {}).icode if hasattr(residue, "seqid") else "",
            "atoms": []
        }
        
        # Serialize each atom
        for atom in residue:
            result["atoms"].append(self.serialize_atom(atom))
            
        return result
    
    def deserialize_residue(self, data: dict):
        """
        Deserialize a dictionary to a gemmi.Residue.
        
        Args:
            data: Dictionary with serialized residue data
            
        Returns:
            gemmi.Residue object
        """
        residue = gemmi.Residue()
        residue.name = data.get("name", "")
        
        # Set seqid if available - use direct constructor
        seqid_num = data.get("seqid_num", 0)
        seqid_icode = data.get("seqid_icode", "")
        try:
            # Use the correct constructor signature
            residue.seqid = gemmi.SeqId(seqid_num, seqid_icode)
        except Exception as e:
            # Fallback if constructor fails
            print(f"Warning: Failed to create SeqId: {str(e)}")
        
        # Add atoms
        for atom_data in data.get("atoms", []):
            try:
                atom = self.deserialize_atom(atom_data)
                residue.add_atom(atom)
            except Exception as e:
                print(f"Warning: Failed to deserialize atom: {str(e)}")
        
        return residue
    
    def serialize_atom(self, atom) -> dict:
        """
        Serialize a gemmi.Atom to a dictionary.
        
        Args:
            atom: gemmi.Atom object
            
        Returns:
            Dictionary with serialized atom data
        """
        result = {
            "_gemmi_type": "Atom",
            "name": getattr(atom, "name", ""),
            "pos": [
                getattr(atom, "pos", [0, 0, 0])[0],
                getattr(atom, "pos", [0, 0, 0])[1],
                getattr(atom, "pos", [0, 0, 0])[2]
            ],
            "b_iso": getattr(atom, "b_iso", 0.0),
            "occ": getattr(atom, "occ", 1.0),
            "charge": getattr(atom, "charge", 0),
            "serial": getattr(atom, "serial", 0)
        }
        
        # Enhanced element handling
        if hasattr(atom, "element"):
            try:
                # Use the element serializer for consistency
                result["element"] = self.serialize_element(atom.element)
            except Exception as e:
                # Fallback to simple string representation
                result["element"] = str(atom.element)
                result["_element_error"] = str(e)
        
        return result
    
    def deserialize_atom(self, data: dict):
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
        if "element" in data:
            element_data = data["element"]
            if isinstance(element_data, dict) and "_gemmi_type" in element_data:
                # Use the element deserializer
                atom.element = self.deserialize_element(element_data)
            elif isinstance(element_data, str):
                # Handle string representation
                atom.element = gemmi.Element(element_data)
            else:
                # Fallback
                atom.element = gemmi.Element("")
        # Try to infer element from atom name if available
        elif "name" in data and data["name"]:
            # Extract first 1-2 characters that might be an element symbol
            # This handles both standard atom names (CA, N, O) and non-standard ones
            import re
            match = re.match(r'([A-Z][a-z]?)', data["name"])
            if match:
                potential_element = match.group(1)
                # Try to create element - gemmi will validate if it's a real element
                try:
                    atom.element = gemmi.Element(potential_element)
                except Exception:
                    # If that fails, just use the first character if it's a valid element
                    if len(data["name"]) > 0:
                        try:
                            atom.element = gemmi.Element(data["name"][0])
                        except Exception:
                            atom.element = gemmi.Element("")
        
        # Set position
        pos = data.get("pos", [0, 0, 0])
        atom.pos = gemmi.Position(pos[0], pos[1], pos[2])
        
        # Set other properties
        atom.b_iso = data.get("b_iso", 0.0)
        atom.occ = data.get("occ", 1.0)
        atom.charge = data.get("charge", 0)
        atom.serial = data.get("serial", 0)
        
        return atom
        
    def serialize_element(self, element) -> dict:
        """
        Serialize a gemmi.Element to a dictionary.
        
        Args:
            element: gemmi.Element object
            
        Returns:
            Dictionary with serialized element data
        """
        try:
            # Extract the actual symbol from the element
            # The str(element) returns something like '<gemmi.Element: C>'
            symbol = str(element)
            # Extract just the symbol part using regex
            import re
            match = re.search(r'<gemmi\.Element: ([A-Za-z0-9]+)>', symbol)
            clean_symbol = match.group(1) if match else ""
            
            return {
                "_gemmi_type": "Element",
                "name": getattr(element, "name", ""),
                "symbol": clean_symbol,
                "weight": getattr(element, "weight", 0.0),
                "atomic_number": getattr(element, "atomic_number", 0)
            }
        except Exception as e:
            # Comprehensive fallback with error information
            return {
                "_gemmi_type": "Element",
                "symbol": str(element),
                "_error": str(e)
            }
    
    def deserialize_element(self, data: dict):
        """
        Deserialize a dictionary to a gemmi.Element.
        
        Args:
            data: Dictionary with serialized element data
            
        Returns:
            gemmi.Element object or data dictionary if deserialization fails
        """
        try:
            import gemmi
            # Try different fields in priority order
            if "symbol" in data:
                # Clean up the symbol if it's still in the <gemmi.Element: X> format
                symbol = data["symbol"]
                import re
                match = re.search(r'<gemmi\.Element: ([A-Za-z0-9]+)>', symbol)
                if match:
                    symbol = match.group(1)
                return gemmi.Element(symbol)
            elif "name" in data and data["name"]:
                return gemmi.Element(data["name"])
            else:
                return gemmi.Element("")
        except Exception as e:
            print(f"Warning: Failed to deserialize Element: {e}")
            # Return the data as fallback
            return data
    
    def serialize_generic_gemmi(self, obj) -> dict:
        """
        Fallback serialization for Gemmi types without specialized handlers.
        
        Args:
            obj: Any Gemmi object
            
        Returns:
            Dictionary with basic representation
        """
        result = {
            "_gemmi_type": obj.__class__.__name__,
            "_module": obj.__class__.__module__,
            "_repr": repr(obj)
        }
        
        # Try to extract common attributes
        for attr in ["name", "id", "serial", "number", "hm", "hall"]:
            if hasattr(obj, attr):
                try:
                    result[attr] = getattr(obj, attr)
                except Exception:
                    pass
        
        # Try to extract position if available
        if hasattr(obj, "pos"):
            try:
                pos = obj.pos
                result["position"] = [pos.x, pos.y, pos.z]
            except Exception:
                pass
        
        return result
    
    def deserialize_generic_gemmi(self, data: dict):
        """
        Fallback deserialization for Gemmi types without specialized handlers.
        
        Args:
            data: Dictionary with serialized data
            
        Returns:
            Dictionary representation (original data)
        """
        # For generic types, just return the dictionary
        return data
