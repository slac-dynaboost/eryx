 # PyTorch Implementation Specification for GNM/OnePhonon Models                                                                                                                  
                                                                                                                                                                                  
 ## High-Level Objective                                                                                                                                                          
 - Create PyTorch implementations of the Gaussian Network Model and One-Phonon model while maintaining numerical accuracy with the original numpy implementation                  
                                                                                                                                                                                  
 ## Mid-Level Objectives                                                                                                                                                          
 1. Create PyTorch GNM implementation with validated numerical accuracy                                                                                                           
 2. Create PyTorch OnePhonon implementation using the torch GNM                                                                                                                   
 3. Maintain test coverage ensuring numerical equivalence                                                                                                                         
 4. Support both CPU and GPU computation                                                                                                                                          
 5. Keep data loading/preprocessing in numpy for now                                                                                                                              
                                                                                                                                                                                  
 ## Implementation Notes                                                                                                                                                          
                                                                                                                                                                                  
 ### Dependencies                                                                                                                                                                 
 - PyTorch >= 2.0                                                                                                                                                                 
 - Existing numpy implementation                                                                                                                                                  
 - pytest for testing                                                                                                                                                             
 - Sample PDB files in tests/pdbs/                                                                                                                                                
                                                                                                                                                                                  
 ### Technical Guidelines                                                                                                                                                         
 - Focus on numerical accuracy first, optimization later                                                                                                                          
 - Use torch.tensor for all core computations                                                                                                                                     
 - Keep numpy for initial data loading/preprocessing                                                                                                                              
 - Add detailed logging for debugging                                                                                                                                             
 - Follow existing code structure where possible                                                                                                                                  
                                                                                                                                                                                  
 ## Context                                                                                                                                                                       
                                                                                                                                                                                  
 ### Beginning Context                                                                                                                                                            
                                                                                                                                                                                  

/eryx /tests /pdbs/ 5zck.pdb /eryx /pdb.py  # Contains numpy GNM /models.py  # Contains numpy OnePhonon                                                                           

                                                                                                                                                                                  
                                                                                                                                                                                  
 ### Ending Context                                                                                                                                                               
                                                                                                                                                                                  

/eryx /tests /test_gaussian_network_torch.py /test_one_phonon_torch.py /eryx /gaussian_network_torch.py /one_phonon_torch.py                                                      

                                                                                                                                                                                  
                                                                                                                                                                                  
 ## Low-Level Tasks                                                                                                                                                               
                                                                                                                                                                                  
 1. Create GaussianNetworkModelTorch Class                                                                                                                                        
 ```aider                                                                                                                                                                         
 CREATE eryx/gaussian_network_torch.py:                                                                                                                                           
   ADD GaussianNetworkModelTorch class:                                                                                                                                           
     - Takes same inputs as numpy version                                                                                                                                         
     - Converts key computations to PyTorch                                                                                                                                       
     - Maintains numpy preprocessing                                                                                                                                              
     - Implements compute_hessian() and compute_Kinv()                                                                                                                            
     - Matches numpy output exactly                                                                                                                                               
                                                                                                                                                                                  

 2 Create OnePhononTorch Class                                                                                                                                                    

                                                                                                                                                                                  
 CREATE eryx/one_phonon_torch.py:                                                                                                                                                 
   ADD OnePhononTorch class:                                                                                                                                                      
     - Uses GaussianNetworkModelTorch                                                                                                                                             
     - Implements apply_disorder()                                                                                                                                                
     - Matches numpy output exactly                                                                                                                                               
                                                                                                                                                                                  

 3 Create GNM Tests                                                                                                                                                               

                                                                                                                                                                                  
 TRANSLATE tests/test_gaussian_network_model.py to torch:                                                                                                                         
  TODO details
                                                                                                                                                                                  

 TRANSLATE tests/test_onephonon.py to torch:                                                                                                                         
  TODO details

                                                                                                                                                                                  
                                                                                                                                                                                  

Key Implementation Details:                                                                                                                                                       

 1 GaussianNetworkModelTorch:                                                                                                                                                     

 • Keep AtomicModel loading in numpy                                                                                                                                              
 • Convert coordinates to torch.tensor after loading                                                                                                                              
 • Implement compute_hessian() using torch operations                                                                                                                             
 • Implement compute_Kinv() using torch.linalg                                                                                                                                    
 • Add device support via .to(device)                                                                                                                                             

 2 OnePhononTorch:                                                                                                                                                                

 • Use GaussianNetworkModelTorch for phonon computation                                                                                                                           
 • Keep molecular transform computation in numpy initially                                                                                                                        
 • Convert key matrices to torch tensors                                                                                                                                          
 • Implement apply_disorder() using torch operations                                                                                                                              

 3 Testing Strategy:                                                                                                                                                              
 TODO basically have to mirror the existing tests

 4 Validation Requirements:                                                                                                                                                       

 • Exact numerical match with numpy (within float precision)                                                                                                                      
 • All tests must pass on both CPU and GPU                                                                                                                                        
 • Memory usage should be reasonable                                                                                                                                              
 • Gradients should flow properly                                                                                                                                                 

The implementation should proceed in order:                                                                                                                                       

 1 GNM implementation and tests                                                                                                                                                   
 2 OnePhonon implementation and tests                                                                                                                                             
 3 GPU support                                                                                                                                                                    
