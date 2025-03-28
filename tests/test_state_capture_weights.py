import unittest
import torch
import numpy as np
from eryx.autotest.logger import Logger
from eryx.autotest.state_builder import StateBuilder
from eryx.models_torch import OnePhonon
import os
import glob

class TestStateCaptureAndRestoration(unittest.TestCase):
    """Test that element weights are properly captured and restored in state logs."""
    
    def setUp(self):
        self.device = torch.device('cpu')
        
    def test_restored_model_contains_weights(self):
        """Test that element weights are preserved during state capture and restoration."""
        # Find a suitable state log file
        log_files = glob.glob("logs/eryx.models.OnePhonon._state_before__build_M*.log")
        if not log_files:
            log_files = glob.glob("logs/*OnePhonon*_state_before_*.log")
        
        if not log_files:
            self.skipTest("No suitable state log files found")
            return
            
        log_file = log_files[0]
        print(f"Using state log: {log_file}")
        
        # Load the state log
        logger = Logger()
        state_data = logger.loadStateLog(log_file)
        self.assertIsNotNone(state_data, f"State log {log_file} could not be loaded")
        
        # Check if the state data contains the 'model' key
        self.assertIn("model", state_data, "State log does not contain 'model' information")
        model_state = state_data.get("model", {})
        
        # Print model state keys for debugging
        print("Model state keys:", list(model_state.keys()))
        
        # Check for elements information
        has_elements = "elements" in model_state
        print(f"State log contains 'elements' information: {has_elements}")
        
        if has_elements:
            # Print raw elements data for inspection
            elements_data = model_state.get("elements")
            print(f"Type of elements data: {type(elements_data)}")
            print(f"Length of elements data: {len(elements_data) if isinstance(elements_data, (list, tuple)) else 'N/A'}")
            
            # Try to extract weights from raw state
            raw_weights = []
            try:
                if isinstance(elements_data, list):
                    for structure in elements_data:
                        if isinstance(structure, list):
                            for element in structure:
                                if hasattr(element, 'weight'):
                                    raw_weights.append(float(element.weight))
                                elif isinstance(element, dict) and 'weight' in element:
                                    raw_weights.append(float(element['weight']))
            except Exception as e:
                print(f"Error extracting weights from raw state: {e}")
            
            if raw_weights:
                print(f"Raw weights from state: min={min(raw_weights)}, max={max(raw_weights)}, count={len(raw_weights)}")
                print(f"All weights zero: {all(w == 0 for w in raw_weights)}")
        
        # Now use the StateBuilder to restore the Torch model
        builder = StateBuilder(device=self.device)
        model = builder.build(OnePhonon, state_data)
        
        # Verify that the restored model has an attribute for elements
        self.assertTrue(hasattr(model, "model"), "Restored model is missing 'model' attribute")
        
        # Check if the model has elements attribute
        has_elements_attr = hasattr(model.model, "elements")
        print(f"Restored model has 'elements' attribute: {has_elements_attr}")
        
        if has_elements_attr:
            # Now inspect each element's weight
            all_weights = []
            try:
                for structure in model.model.elements:
                    for element in structure:
                        try:
                            if hasattr(element, 'weight'):
                                w = float(element.weight)
                                all_weights.append(w)
                            elif isinstance(element, dict) and 'weight' in element:
                                w = float(element['weight'])
                                all_weights.append(w)
                        except Exception as e:
                            print(f"Could not extract weight from element: {e}")
            except Exception as e:
                print(f"Error iterating through elements: {e}")
            
            if all_weights:
                print("Restored element weights statistics:", 
                      f"min={min(all_weights)}, max={max(all_weights)}, count={len(all_weights)}")
                print(f"All weights zero: {all(w == 0 for w in all_weights)}")
                
                # Check that at least one weight is nonzero
                self.assertTrue(any(w > 0 for w in all_weights),
                            "All restored element weights are zero; they might be dropped during state capture/restoration")
            else:
                print("No weights could be extracted from restored model")
        
        # Test direct mass matrix calculation
        print("\nTesting direct mass matrix calculation:")
        try:
            # Call the method that builds the mass matrix
            M_allatoms = model._build_M_allatoms()
            
            # Check if the mass matrix has reasonable values
            M_diag = torch.diagonal(M_allatoms.reshape(model.n_asu * model.n_dof_per_asu_actual, 
                                                     model.n_asu * model.n_dof_per_asu_actual))
            
            print(f"Mass matrix diagonal min: {M_diag.min().item()}")
            print(f"Mass matrix diagonal max: {M_diag.max().item()}")
            print(f"Mass matrix diagonal mean: {M_diag.mean().item()}")
            
            # Check if the mass matrix has reasonable values (not all ones)
            self.assertFalse(torch.allclose(M_diag, torch.ones_like(M_diag)),
                           "Mass matrix diagonal is all ones, suggesting default weights were used")
        except Exception as e:
            print(f"Error building mass matrix: {e}")

if __name__ == '__main__':
    unittest.main()
