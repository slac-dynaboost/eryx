# Spec Prompt: Parallel Test Suite for PyTorch Implementation
## High-Level Objective
 - Create a parallel set of tests that will validate the functional correctness of the soon-to-be-developed PyTorch-based OnePhonon model and its supporting GaussianNetworkModel. The new tests should     
 mirror our existing NumPy-based tests (found in files such as `tests/test_gaussian_network_model.py` and `tests/test_onephonon.py`) but will import and verify outputs from the Torch implementations.     
                                                                                                                                                                                                            
 ## Mid-Level Objectives                                                                                                                                                                                    
 - Duplicate the existing unit and integration tests for the GaussianNetworkModel and OnePhonon models into separate files dedicated to the Torch version.                                                  
 - Update the test imports so that the Torch implementations are used (e.g. import `GaussianNetworkModel` from `eryx.gaussian_network_torch` and `OnePhonon` from `eryx.one_phonon_torch`).                 
 - Ensure any outputs that are now Torch tensors are converted to NumPy arrays using:                                                                                                                       
   ```python                                                                                                                                                                                                
   if isinstance(variable, torch.Tensor):                                                                                                                                                                   
       variable = variable.detach().cpu().numpy()                                                                                                                                                           
                                                                                                                                                                                                            

before comparing with reference data.                                                                                                                                                                       

 • Preserve the existing test logic—including numerical tolerances, shape assertions, and log analysis—and ensure it applies equally to the new implementation.                                             


                                                                                            Implementation Notes                                                                                            

 • File Duplication:                                                                                                                                                                                        
   Create new files:                                                                                                                                                                                        
    • tests/test_gaussian_network_model_torch.py                                                                                                                                                            
    • tests/test_onephonon_torch.py                                                                                                                                                                         
 • Import Updates:                                                                                                                                                                                          
   In the torch test files, update the imports:                                                                                                                                                             
                                                                                                                                                                                                            
    - from eryx.pdb import GaussianNetworkModel                                                                                                                                                             
    + from eryx.gaussian_network_torch import GaussianNetworkModel                                                                                                                                          
                                                                                                                                                                                                            
   and:                                                                                                                                                                                                     
                                                                                                                                                                                                            
    - from eryx.models import OnePhonon                                                                                                                                                                     
    + from eryx.one_phonon_torch import OnePhonon                                                                                                                                                           
                                                                                                                                                                                                            
 • Tensor Conversion:                                                                                                                                                                                       
   For every test that receives output from a model method (e.g., compute_hessian(), compute_K(), or apply_disorder()), insert conversion code such as:                                                     
                                                                                                                                                                                                            
    - hessian = gnm_model.compute_hessian()                                                                                                                                                                 
    + hessian = gnm_model.compute_hessian()                                                                                                                                                                 
    + if isinstance(hessian, torch.Tensor):                                                                                                                                                                 
    +     hessian = hessian.detach().cpu().numpy()                                                                                                                                                          
                                                                                                                                                                                                            
   Do this similarly for any tensor outputs within the OnePhonon tests (for example, after calling apply_disorder()).                                                                                       
 • Log-Based Checks:                                                                                                                                                                                        
   If any tests rely on logs (using the LogAnalyzer), confirm that the expected log entries are still valid. Modify expected strings only if the Torch version’s logging output differs.                    


                                                                                                  Context                                                                                                   

                                                                                             Beginning Context                                                                                              

 • Existing tests in tests/test_gaussian_network_model.py                                                                                                                                                   
 • Existing tests in tests/test_onephonon.py                                                                                                                                                                
 • The Torch implementations, which will eventually reside in eryx.gaussian_network_torch and eryx.one_phonon_torch                                                                                         

                                                                                               Ending Context                                                                                               

 • Parallel test files:                                                                                                                                                                                     
    • tests/test_gaussian_network_model_torch.py                                                                                                                                                            
    • tests/test_onephonon_torch.py                                                                                                                                                                         
 • Both test suites will use the same reference data and configuration files found in tests/test_data/reference and tests/test_utils                                                                        


                                                                                              Low-Level Tasks                                                                                               

 1 Duplicate and Update Gaussian Network Model Tests                                                                                                                                                        
                                                                                                                                                                                                            
    CREATE tests/test_gaussian_network_model_torch.py:                                                                                                                                                      
        - Duplicate the content from tests/test_gaussian_network_model.py.                                                                                                                                  
        - UPDATE the import:                                                                                                                                                                                
              from eryx.pdb import GaussianNetworkModel                                                                                                                                                     
          to:                                                                                                                                                                                               
              from eryx.gaussian_network_torch import GaussianNetworkModel                                                                                                                                  
        - For each test that retrieves a tensor output (e.g. from compute_hessian() or compute_K()),                                                                                                        
          insert conversion logic:                                                                                                                                                                          
              if isinstance(result, torch.Tensor):                                                                                                                                                          
                  result = result.detach().cpu().numpy()                                                                                                                                                    
                                                                                                                                                                                                            
 2 Duplicate and Update OnePhonon Model Tests                                                                                                                                                               
                                                                                                                                                                                                            
    CREATE tests/test_onephonon_torch.py:                                                                                                                                                                   
        - Duplicate the content from tests/test_onephonon.py.                                                                                                                                               
        - UPDATE the import:                                                                                                                                                                                
              from eryx.models import OnePhonon                                                                                                                                                             
          to:                                                                                                                                                                                               
              from eryx.one_phonon_torch import OnePhonon                                                                                                                                                   
        - Locate tests that involve outputs like diffraction patterns or structure factors.                                                                                                                 
          Immediately after calling the method (e.g. apply_disorder()), insert:                                                                                                                             
              if isinstance(output, torch.Tensor):                                                                                                                                                          
                  output = output.detach().cpu().numpy()                                                                                                                                                    
          before performing any flattening or numerical comparisons.                                                                                                                                        
                                                                                                                                                                                                            
 3 Verify Consistency                                                                                                                                                                                       
    • Ensure that all numerical comparisons (using np.testing.assert_allclose, pytest.approx, etc.) operate on NumPy arrays.                                                                                
    • Confirm that log parsing via LogAnalyzer (in tests under tests/test_utils/log_analysis.py) continues to function as expected.                                                                         
    • Validate that configuration and reference data files remain unchanged and apply to both test suites.                                                                                                  

