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

if __name__ == '__main__':
    unittest.main()

