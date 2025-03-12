import unittest
import torch
import numpy as np
from eryx.pdb import AtomicModel
from eryx.adapters import PDBToTensor
import gemmi

class TestElementWeights(unittest.TestCase):
    """Test extraction and conversion of element weights from PDB files."""
    
    def test_element_weights_extraction(self):
        """Test that element weights are correctly extracted from PDB files and preserved during conversion."""
        pdb_path = "tests/pdbs/5zck_p1.pdb"
        
        # Load the PDB file directly with gemmi to check raw data
        try:
            structure = gemmi.read_structure(pdb_path)
            print("\nDirect Gemmi inspection:")
            element_counts = {}
            element_weights = {}
            
            for model in structure:
                for chain in model:
                    for residue in chain:
                        for atom in residue:
                            element = atom.element
                            if element:
                                element_name = element.name
                                weight = gemmi.Element(element_name).weight
                                element_counts[element_name] = element_counts.get(element_name, 0) + 1
                                element_weights[element_name] = weight
                                print(f"Atom: {atom.name}, Element: {element_name}, Weight: {weight}")
                            else:
                                # Try to determine element from atom name
                                atom_name = atom.name.strip()
                                if atom_name:
                                    # Extract first 1-2 characters as potential element symbol
                                    if atom_name[0].isalpha():
                                        if len(atom_name) > 1 and atom_name[1].isalpha():
                                            elem_symbol = atom_name[:2].capitalize()
                                        else:
                                            elem_symbol = atom_name[0].upper()
                                            
                                        try:
                                            # Try to get weight from element symbol
                                            weight = gemmi.Element(elem_symbol).weight
                                            element_counts[elem_symbol] = element_counts.get(elem_symbol, 0) + 1
                                            element_weights[elem_symbol] = weight
                                            print(f"Atom: {atom.name}, Inferred Element: {elem_symbol}, Weight: {weight}")
                                            continue
                                        except:
                                            pass
                                
                                print(f"Atom: {atom.name}, Element: Unknown, Weight: N/A")
            
            print("\nElement summary:")
            for elem, count in element_counts.items():
                print(f"Element {elem}: Count={count}, Weight={element_weights[elem]}")
                
        except Exception as e:
            print(f"Error inspecting PDB with gemmi: {e}")
        
        # Load the NP AtomicModel
        model = AtomicModel(pdb_path, expand_p1=True)
        
        # Directly extract weights from the NP model
        raw_weights = []
        if hasattr(model, 'elements') and model.elements:
            for structure in model.elements:
                for element in structure:
                    # Convert to float for consistency
                    raw_weights.append(float(element.weight))
        
        # Log raw weights for inspection
        print("\nRaw weights from NP model:", raw_weights)
        print(f"Number of weights: {len(raw_weights)}")
        print(f"Min weight: {min(raw_weights) if raw_weights else 'N/A'}")
        print(f"Max weight: {max(raw_weights) if raw_weights else 'N/A'}")
        
        # Check if all weights are zero
        all_zero_raw = all(w == 0 for w in raw_weights) if raw_weights else True
        
        # Now, use the adapter to convert the model
        adapter = PDBToTensor()
        state = adapter.convert_atomic_model(model)
        
        # Print the keys in the state dictionary
        print("\nKeys in converted state:", list(state.keys()))
        
        # The adapter should include the 'elements' key with converted info.
        # Extract weights from the converted state (if available)
        converted_weights = []
        if "elements" in state:
            print("\nExamining converted elements:")
            print(f"Type of elements: {type(state['elements'])}")
            print(f"Length of elements: {len(state['elements'])}")
            
            if isinstance(state["elements"], list) and state["elements"]:
                print(f"Type of first element: {type(state['elements'][0])}")
                
                for structure in state["elements"]:
                    if isinstance(structure, list):
                        for element in structure:
                            if hasattr(element, 'weight'):
                                converted_weights.append(float(element.weight))
                            elif isinstance(element, dict) and 'weight' in element:
                                converted_weights.append(float(element['weight']))
                            else:
                                print(f"Element has no weight attribute: {element}")
        
        print("\nConverted weights:", converted_weights)
        print(f"Number of converted weights: {len(converted_weights)}")
        if converted_weights:
            print(f"Min converted weight: {min(converted_weights)}")
            print(f"Max converted weight: {max(converted_weights)}")
        
        # Check if all converted weights are zero
        all_zero_converted = all(w == 0 for w in converted_weights) if converted_weights else True
        
        # Print model attributes for debugging
        print("\nModel attributes:")
        for attr in dir(model):
            if not attr.startswith('_') and not callable(getattr(model, attr)):
                try:
                    value = getattr(model, attr)
                    if isinstance(value, (int, float, str, bool, list, dict, tuple)):
                        print(f"{attr}: {value}")
                    else:
                        print(f"{attr}: {type(value)}")
                except:
                    print(f"{attr}: <error getting value>")
        
        # Print serialization info
        print("\nSerialization info:")
        if hasattr(adapter, 'serialize_gemmi_object'):
            print("Adapter has serialize_gemmi_object method")
        
        # Assertions
        if all_zero_raw:
            print("\nWARNING: All extracted weights from the PDB are zero; the PDB might not include weight info.")
        
        if all_zero_converted:
            print("\nWARNING: All weights remain zero after conversion; check serialization/deserialization of element info.")
        
        # Check if the weights are preserved during conversion
        if raw_weights and converted_weights:
            weights_preserved = len(raw_weights) == len(converted_weights) and all(
                abs(r - c) < 1e-6 for r, c in zip(raw_weights, converted_weights)
            )
            self.assertTrue(weights_preserved, "Weights were not preserved during conversion")

if __name__ == '__main__':
    unittest.main()
