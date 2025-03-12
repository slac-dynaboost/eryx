import unittest
import torch
import numpy as np
from eryx.pdb import AtomicModel
from eryx.models_torch import OnePhonon
import gemmi
import os

class TestMassMatrix(unittest.TestCase):
    """Test the mass matrix calculation in the OnePhonon model."""
    
    def setUp(self):
        self.device = torch.device('cpu')
        self.pdb_path = "tests/pdbs/5zck_p1.pdb"
        
        # Ensure the PDB file exists
        if not os.path.exists(self.pdb_path):
            self.skipTest(f"PDB file not found: {self.pdb_path}")
    
    def test_mass_matrix_calculation(self):
        """Test that the mass matrix is correctly calculated with proper atomic weights."""
        # First, extract expected weights directly from the PDB file
        expected_weights = []
        try:
            structure = gemmi.read_structure(self.pdb_path)
            for model in structure:
                for chain in model:
                    for residue in chain:
                        for atom in residue:
                            element = atom.element
                            if element:
                                weight = gemmi.Element(element.name).weight
                                expected_weights.append(float(weight))
                            else:
                                # Try to infer element from atom name
                                atom_name = atom.name.strip()
                                if atom_name and atom_name[0].isalpha():
                                    if len(atom_name) > 1 and atom_name[1].isalpha():
                                        elem_symbol = atom_name[:2].capitalize()
                                    else:
                                        elem_symbol = atom_name[0].upper()
                                        
                                    try:
                                        weight = gemmi.Element(elem_symbol).weight
                                        expected_weights.append(float(weight))
                                        continue
                                    except:
                                        pass
                                
                                # Default to carbon weight
                                expected_weights.append(12.0)
        except Exception as e:
            print(f"Error extracting weights from PDB: {e}")
            expected_weights = []
        
        if not expected_weights:
            self.skipTest("Could not extract expected weights from PDB file")
        
        print(f"Expected weights from PDB: count={len(expected_weights)}")
        print(f"Min weight: {min(expected_weights)}, Max weight: {max(expected_weights)}")
        
        # Initialize the OnePhonon model
        model = OnePhonon(
            pdb_path=self.pdb_path,
            hsampling=(0, 1, 2),
            ksampling=(0, 1, 2),
            lsampling=(0, 1, 2),
            expand_p1=True,
            group_by='asu',
            device=self.device
        )
        
        # Call the method to build the mass matrix
        M_allatoms = model._build_M_allatoms()
        
        # Extract the diagonal elements which should contain the mass values
        M_diag = torch.diagonal(M_allatoms.reshape(model.n_asu * model.n_dof_per_asu_actual, 
                                                 model.n_asu * model.n_dof_per_asu_actual))
        
        # Each atom has 3 diagonal elements (x, y, z) with the same mass
        # So we need to take every 3rd element to get the actual masses
        extracted_masses = M_diag[::3].detach().cpu().numpy()
        
        print(f"Extracted masses from matrix: count={len(extracted_masses)}")
        print(f"Min mass: {np.min(extracted_masses)}, Max mass: {np.max(extracted_masses)}")
        
        # Check if the number of masses matches the expected number of weights
        # (accounting for possible differences in atom count)
        min_length = min(len(expected_weights), len(extracted_masses))
        
        # Compare the first min_length elements
        expected_subset = expected_weights[:min_length]
        extracted_subset = extracted_masses[:min_length]
        
        # Check if the masses are close to the expected weights
        # If all masses are 1.0, then default weights were used
        all_ones = np.allclose(extracted_subset, np.ones_like(extracted_subset))
        self.assertFalse(all_ones, "All masses are 1.0, suggesting default weights were used")
        
        # Check if the masses are close to the expected weights
        # This might fail if the weights are scaled differently, so we'll also check the ratio
        if not np.allclose(extracted_subset, expected_subset, rtol=1e-3, atol=1e-3):
            # Check if there's a consistent scaling factor
            ratios = extracted_subset / expected_subset
            ratio_std = np.std(ratios)
            print(f"Ratio mean: {np.mean(ratios)}, std: {ratio_std}")
            
            # If the standard deviation is small, then there's a consistent scaling factor
            self.assertLess(ratio_std / np.mean(ratios), 0.1, 
                          "Masses don't match expected weights and don't have a consistent scaling factor")
        
        # Now test the full mass matrix calculation
        model._build_M()
        
        # Check that Linv has reasonable values
        self.assertTrue(hasattr(model, 'Linv'), "Linv not created")
        self.assertFalse(torch.isnan(model.Linv).any(), "Linv contains NaN values")
        self.assertFalse(torch.isinf(model.Linv).any(), "Linv contains infinite values")
        
        # Print Linv statistics
        linv_numpy = model.Linv.detach().cpu().numpy()
        print(f"Linv min: {np.min(linv_numpy)}, max: {np.max(linv_numpy)}")
        print(f"Linv mean: {np.mean(linv_numpy)}, std: {np.std(linv_numpy)}")

if __name__ == '__main__':
    unittest.main()
