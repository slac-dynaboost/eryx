# test_gemmi_serializer.py

import unittest
import gemmi
from eryx.autotest.gemmi_serializer import GemmiSerializer

class TestGemmiSerializer(unittest.TestCase):

    def setUp(self):
        # Build a small gemmi.Structure from scratch
        self.structure = gemmi.Structure()
        
        # Setup cell
        self.structure.cell = gemmi.UnitCell(55.0, 66.0, 77.0, 90.0, 100.0, 120.0)
        self.structure.spacegroup_hm = 'P 2 2 2'
        
        # Create one model
        model = gemmi.Model("test_model")
        
        # One chain
        chain = gemmi.Chain("A")
        
        # One residue
        residue = gemmi.Residue()
        residue.name = "ALA"
        residue.seqid = gemmi.SeqId(10, " ")


        # One atom
        atom = gemmi.Atom()
        atom.name = "CA"
        atom.pos = gemmi.Position(1.234, 2.345, 3.456)
        atom.occ = 0.9
        atom.b_iso = 20.0
        atom.element = gemmi.Element("C")
        
        residue.add_atom(atom)
        chain.add_residue(residue)
        model.add_chain(chain)
        self.structure.add_model(model)

    def test_serialize_deserialize(self):
        serializer = GemmiSerializer()
        
        # Serialize
        serialized_dict = serializer.serialize_structure(self.structure)
        
        # Basic checks
        self.assertIn("cell", serialized_dict)
        self.assertIn("models", serialized_dict)
        
        # Check cell contents
        self.assertAlmostEqual(serialized_dict["cell"]["a"], 55.0)
        self.assertAlmostEqual(serialized_dict["cell"]["beta"], 100.0)
        
        # Check model/chain residue
        self.assertEqual(len(serialized_dict["models"]), 1)
        model_dict = serialized_dict["models"][0]
        self.assertEqual(model_dict["name"], "test_model")
        
        self.assertEqual(len(model_dict["chains"]), 1)
        chain_dict = model_dict["chains"][0]
        self.assertEqual(chain_dict["name"], "A")
        
        self.assertEqual(len(chain_dict["residues"]), 1)
        residue_dict = chain_dict["residues"][0]
        self.assertEqual(residue_dict["name"], "ALA")
        self.assertEqual(residue_dict["seqid_num"], 10)
        
        self.assertEqual(len(residue_dict["atoms"]), 1)
        atom_dict = residue_dict["atoms"][0]
        self.assertEqual(atom_dict["name"], "CA")
        self.assertAlmostEqual(atom_dict["pos"][0], 1.234, places=3)
        
        # Now deserialize
        restored_structure = serializer.deserialize_structure(serialized_dict)
        
        # Compare key aspects of the round-trip
        self.assertAlmostEqual(restored_structure.cell.a, 55.0)
        self.assertAlmostEqual(restored_structure.cell.beta, 100.0)
        self.assertEqual(restored_structure.spacegroup_hm, 'P 2 2 2')
        
        # check model
        self.assertEqual(len(restored_structure), 1)  # 1 model
        model_rest = restored_structure[0]
        self.assertEqual(model_rest.name, "test_model")
        
        # chain
        self.assertEqual(len(model_rest), 1)
        chain_rest = model_rest[0]
        self.assertEqual(chain_rest.name, "A")
        
        # residue
        self.assertEqual(len(chain_rest), 1)
        res_rest = chain_rest[0]
        self.assertEqual(res_rest.name, "ALA")
        self.assertEqual(res_rest.seqid.num, 10)
        
        # atom
        print("\n=== GEMMI RESIDUE ATOM ACCESS DEMONSTRATION ===")
        
        # Try to access .atoms directly (will fail but we'll catch it)
        try:
            print(f"Trying to access res_rest.atoms directly...")
            atoms_attr = res_rest.atoms
            print(f"  Result: {atoms_attr}")  # This won't execute
        except AttributeError as e:
            print(f"  Error: {e}")
            
        # Show the correct way - iteration
        print("\nCorrect way - iterate over residue:")
        atom_count = 0
        for atom in res_rest:
            atom_count += 1
            print(f"  Found atom: {atom.name}, element: {atom.element.name}, position: ({atom.pos.x:.3f}, {atom.pos.y:.3f}, {atom.pos.z:.3f})")
        print(f"  Total atoms found by iteration: {atom_count}")
        
        # Convert to list for easier handling
        print("\nUsing list conversion:")
        atoms_list = list(res_rest)  # Convert atoms iterator to list
        print(f"  Number of atoms in list: {len(atoms_list)}")
        print(f"  First atom in list: {atoms_list[0].name}")
        
        # Now do the actual test assertions
        self.assertEqual(len(atoms_list), 1)
        atom_rest = atoms_list[0]
        self.assertEqual(atom_rest.name, "CA")
        self.assertAlmostEqual(atom_rest.pos.x, 1.234, places=3)
        self.assertAlmostEqual(atom_rest.b_iso, 20.0)
        self.assertEqual(atom_rest.element.name, "C")

    def test_serialize_element(self):
        # Create a Gemmi Element for Nitrogen.
        element = gemmi.Element("N")
        serializer = GemmiSerializer()
        serialized = serializer.serialize_gemmi(element)
        
        # Print serialized output for debugging.
        print("Serialized element:", serialized)
        
        # Check that the serialized dictionary contains a 'symbol' key.
        self.assertIn("symbol", serialized, "Serialized element should have a 'symbol' key.")
        symbol = serialized["symbol"]
        self.assertEqual(symbol, "N", f"Expected symbol 'N', got '{symbol}'")
        
        # Check for a weight-related key.
        weight = serialized.get("weight", None)
        atomic_weight = serialized.get("atomic_weight", None)
        
        # For a valid element, at least one of these should be nonzero/non-None.
        self.assertTrue(
            (weight is not None and weight != 0.0) or (atomic_weight is not None and atomic_weight != 0.0),
            "Expected nonzero weight or atomic_weight for element N."
        )

if __name__ == '__main__':
    unittest.main()

