# Revised Phased Implementation Plan for PyTorch Port

## Overview and Resources

This document outlines the phased implementation plan for the PyTorch port, focusing specifically on the components listed in `to_convert.json`. For comprehensive information about the project, refer to these related documents:

- **[Implementation Plan](./plan.md)**: High-level overview of the port structure and boundaries
- **[Architecture Document](./architecture.md)**: Component descriptions and interactions
- **[Implementation Specifications](./specs/spec1.md)**: Detailed specifications for implementation
- **[Project Rules](./project_rules.md)**: Guidelines and requirements for implementation
- **[Component Inventory](./to_convert.json)**: Definitive list of components to be ported

## Function-to-Task Mapping

This table maps each function from `to_convert.json` to specific implementation tasks:

| File | Function | Implementation Phase | Task |
|------|----------|----------------------|------|
| **map_utils.py** | generate_grid | Phase 3 | Task 3.1 |
| **map_utils.py** | compute_resolution | Phase 3 | Task 3.1 |
| **map_utils.py** | get_resolution_mask | Phase 3 | Task 3.1 |
| **scatter.py** | compute_form_factors | Phase 3 | Task 3.2 |
| **scatter.py** | structure_factors_batch | Phase 3 | Task 3.2 |
| **scatter.py** | structure_factors | Phase 3 | Task 3.2 |
| **models.py** | OnePhonon.__init__ | Phase 4 | Task 4.1.1 |
| **models.py** | OnePhonon._setup | Phase 4 | Task 4.1.1 |
| **models.py** | OnePhonon._setup_phonons | Phase 4 | Task 4.1.1 |
| **models.py** | OnePhonon._build_A | Phase 4 | Task 4.1.2 |
| **models.py** | OnePhonon._build_M | Phase 4 | Task 4.1.2 |
| **models.py** | OnePhonon._build_M_allatoms | Phase 4 | Task 4.1.2 |
| **models.py** | OnePhonon._project_M | Phase 4 | Task 4.1.2 |
| **models.py** | OnePhonon._build_kvec_Brillouin | Phase 4 | Task 4.1.3 |
| **models.py** | OnePhonon._center_kvec | Phase 4 | Task 4.1.3 |
| **models.py** | OnePhonon._at_kvec_from_miller_points | Phase 4 | Task 4.1.3 |
| **models.py** | OnePhonon.compute_gnm_phonons | Phase 4 | Task 4.1.4 |
| **models.py** | OnePhonon.compute_hessian | Phase 4 | Task 4.1.4 |
| **models.py** | OnePhonon.compute_covariance_matrix | Phase 4 | Task 4.1.5 |
| **models.py** | OnePhonon.apply_disorder | Phase 4 | Task 4.1.6 |
| **pdb.py** | GaussianNetworkModel.compute_hessian | Phase 4 | Task 4.1.4 |
| **pdb.py** | GaussianNetworkModel.compute_K | Phase 4 | Task 4.1.4 |
| **pdb.py** | GaussianNetworkModel.compute_Kinv | Phase 4 | Task 4.1.4 |
| **pdb.py** | AtomicModel._get_xyz_asus | Phase 2 | Task 2.1 |
| **pdb.py** | AtomicModel.flatten_model | Phase 2 | Task 2.1 |
| **pdb.py** | Crystal.get_asu_xyz | Phase 2 | Task 2.1 |

## Implementation Checkpoints

| Checkpoint | Description | Functions Covered | Verification Method |
|------------|-------------|-------------------|---------------------|
| CP1 | Core Utilities Complete | N/A (utility classes) | Unit tests pass |
| CP2 | Adapters Complete | AtomicModel and Crystal methods | Unit tests pass |
| CP3 | Map Utils Complete | generate_grid, compute_resolution, get_resolution_mask | Ground truth tests pass |
| CP4 | Scatter Complete | compute_form_factors, structure_factors_batch, structure_factors | Ground truth tests pass |
| CP5 | Matrix Construction Complete | _build_A, _build_M, _build_M_allatoms, _project_M | State-based tests pass |
| CP6 | K-vector Methods Complete | _build_kvec_Brillouin, _center_kvec, _at_kvec_from_miller_points | State-based tests pass |
| CP7 | Phonon Calculation Complete | compute_gnm_phonons, compute_hessian, GNM methods | State-based tests pass |
| CP8 | Covariance Matrix Complete | compute_covariance_matrix | State-based tests pass |
| CP9 | Apply Disorder Complete | apply_disorder | State-based tests pass |
| CP10 | End-to-End Integration Complete | All OnePhonon methods | Integration tests pass |

## Ground Truth Testing Strategy

### Data Storage and Access
Ground truth data is stored in the `logs/` directory using two complementary approaches:

1. **Function-Based Ground Truth**:
   - Inputs and outputs for pure functions captured via the `@debug` decorator
   - Each log file follows naming convention `logs/eryx.module.function.log`
   - Contains serialized inputs and expected outputs for testing

2. **State-Based Ground Truth**:
   - Complete object state before and after method execution for stateful objects
   - Log files follow naming convention:
     - Before: `logs/eryx.module.class._state_before_methodname.log`
     - After: `logs/eryx.module.class._state_after_methodname.log`
   - Contains serialized object state including all attributes needed for testing

### Accessing Ground Truth Data
The testing framework accesses ground truth data through the following process:

For function-based testing:
1. Use `Logger.searchLogDirectory()` to find relevant log files for a component
2. Use `Logger.loadLog()` to deserialize input/output pairs for testing
3. Feed inputs through the PyTorch implementation and compare outputs

For state-based testing:
1. Use `Logger.searchStateLogDirectory()` to find before/after state log files
2. Use `Logger.loadStateLog()` to deserialize object state
3. Initialize PyTorch object with "before" state using adapters
4. Execute method under test
5. Compare resulting state with expected "after" state

### Testing Framework
The `TorchTesting` class in `eryx/autotest/torch_testing.py` now provides specialized methods for both function-based and state-based testing:

- **Function-Based Testing**:
  - `testTorchCallable(log_path_prefix, torch_func)`: Tests a PyTorch function against NumPy ground truth
  - `create_tensor_test_case(log_path_prefix, torch_func, numpy_func)`: Creates complete test cases

- **State-Based Testing**:
  - `testTorchCallableWithState(log_path_prefix, torch_class, method_name)`: Tests a PyTorch method using state
  - `initializeFromState(torch_class, state_data)`: Creates and initializes an object from state data
  - `compareStates(expected_state, actual_state, tolerances)`: Compares object states with appropriate tolerances

- **Gradient Testing**:
  - `check_gradients(torch_func, inputs)`: Validates gradient computation for functions
  - `check_state_gradients(torch_obj, method_name, args)`: Validates gradient flow through object state

### Component-to-Test Mapping with Adapter Usage

| Component | Source Class | Adapter | Input Conversion | Output Conversion | Testing Approach |
|-----------|--------------|---------|------------------|-------------------|------------------|
| `ComplexTensorOps` | N/A (utility) | N/A | N/A | N/A | Function-based |
| `EigenOps` | N/A (utility) | N/A | N/A | N/A | Function-based |
| `map_utils_torch.generate_grid` | `map_utils.generate_grid` | `GridToTensor` | `convert_grid()` | `tensor_to_array()` | Function-based |
| `map_utils_torch.compute_resolution` | `map_utils.compute_resolution` | `GridToTensor` | `array_to_tensor()` | `tensor_to_array()` | Function-based |
| `map_utils_torch.get_resolution_mask` | `map_utils.get_resolution_mask` | `GridToTensor` | `array_to_tensor()` | `tensor_to_array()` | Function-based |
| `scatter_torch.compute_form_factors` | `scatter.compute_form_factors` | `PDBToTensor` | `array_to_tensor()` | `tensor_to_array()` | Function-based |
| `scatter_torch.structure_factors_batch` | `scatter.structure_factors_batch` | `PDBToTensor` | `array_to_tensor()` | `tensor_to_array()` | Function-based |
| `scatter_torch.structure_factors` | `scatter.structure_factors` | `PDBToTensor` | `array_to_tensor()` | `tensor_to_array()` | Function-based |
| `models_torch.OnePhonon.__init__` | `models.OnePhonon` | `PDBToTensor`, `GridToTensor` | `convert_atomic_model()`, `convert_grid()` | N/A | State-based |
| `models_torch.OnePhonon._build_A` | `models.OnePhonon._build_A` | `ModelAdapters` | State from log | State comparison | State-based |
| `models_torch.OnePhonon._build_M` | `models.OnePhonon._build_M` | `ModelAdapters` | State from log | State comparison | State-based |
| `models_torch.OnePhonon.compute_gnm_phonons` | `models.OnePhonon.compute_gnm_phonons` | `ModelAdapters` | State from log | State comparison | State-based |
| `models_torch.OnePhonon.compute_covariance_matrix` | `models.OnePhonon.compute_covariance_matrix` | `ModelAdapters` | State from log | State comparison | State-based |
| `models_torch.OnePhonon.apply_disorder` | `models.OnePhonon.apply_disorder` | `ModelAdapters` | State from log | `tensor_to_array()` | State-based |
| `models_torch.GaussianNetworkModel methods` | `pdb.GaussianNetworkModel` | `PDBToTensor` | `convert_gnm()` | `tensor_to_array()` | State-based |

### Component-to-Test Mapping with Ground Truth Data

| PyTorch Component | Ground Truth Data | Testing Approach | Tolerances |
|-------------------|-------------------|------------------|------------|
| `ComplexTensorOps` | N/A (utility class) | Function-based | rtol=1e-5, atol=1e-8 |
| `EigenOps` | N/A (utility class) | Function-based | rtol=1e-5, atol=1e-8 |
| `map_utils_torch.generate_grid` | `logs/eryx.map_utils.generate_grid.log` | Function-based | rtol=1e-5, atol=1e-8 |
| `map_utils_torch.compute_resolution` | `logs/eryx.map_utils.compute_resolution.log` | Function-based | rtol=1e-5, atol=1e-8 |
| `map_utils_torch.get_resolution_mask` | `logs/eryx.map_utils.get_resolution_mask.log` | Function-based | exact match |
| `scatter_torch.compute_form_factors` | `logs/eryx.scatter.compute_form_factors.log` | Function-based | rtol=1e-4, atol=1e-7 |
| `scatter_torch.structure_factors_batch` | `logs/eryx.scatter.structure_factors_batch.log` | Function-based | rtol=1e-4, atol=1e-7 |
| `scatter_torch.structure_factors` | `logs/eryx.scatter.structure_factors.log` | Function-based | rtol=1e-4, atol=1e-7 |
| `models_torch.OnePhonon._build_A` | `logs/eryx.models.OnePhonon._state_before__build_A.log`<br>`logs/eryx.models.OnePhonon._state_after__build_A.log` | State-based | rtol=1e-5, atol=1e-8 |
| `models_torch.OnePhonon._build_M` | `logs/eryx.models.OnePhonon._state_before__build_M.log`<br>`logs/eryx.models.OnePhonon._state_after__build_M.log` | State-based | rtol=1e-5, atol=1e-8 |
| `models_torch.OnePhonon.compute_gnm_phonons` | `logs/eryx.models.OnePhonon._state_before_compute_gnm_phonons.log`<br>`logs/eryx.models.OnePhonon._state_after_compute_gnm_phonons.log` | State-based | rtol=1e-4, atol=1e-6 |
| `models_torch.OnePhonon.compute_covariance_matrix` | `logs/eryx.models.OnePhonon._state_before_compute_covariance_matrix.log`<br>`logs/eryx.models.OnePhonon._state_after_compute_covariance_matrix.log` | State-based | rtol=1e-4, atol=1e-6 |
| `models_torch.OnePhonon.apply_disorder` | `logs/eryx.models.OnePhonon._state_before_apply_disorder.log`<br>`logs/eryx.models.OnePhonon._state_after_apply_disorder.log` | State-based | rtol=1e-3, atol=1e-5 |
| `models_torch.OnePhonon._build_kvec_Brillouin` | `logs/eryx.models.OnePhonon._state_before__build_kvec_Brillouin.log`<br>`logs/eryx.models.OnePhonon._state_after__build_kvec_Brillouin.log` | State-based | rtol=1e-5, atol=1e-8 |
| `models_torch.OnePhonon._center_kvec` | `logs/eryx.models._center_kvec.log` | Function-based | rtol=1e-5, atol=1e-8 |
| `models_torch.OnePhonon._at_kvec_from_miller_points` | `logs/eryx.models._at_kvec_from_miller_points.log` | Function-based | exact match |
| `models_torch.OnePhonon._build_M_allatoms` | `logs/eryx.models.OnePhonon._state_before__build_M_allatoms.log`<br>`logs/eryx.models.OnePhonon._state_after__build_M_allatoms.log` | State-based | rtol=1e-5, atol=1e-8 |
| `models_torch.OnePhonon._project_M` | `logs/eryx.models.OnePhonon._state_before__project_M.log`<br>`logs/eryx.models.OnePhonon._state_after__project_M.log` | State-based | rtol=1e-5, atol=1e-8 |
| `models_torch.GaussianNetworkModel.compute_hessian` | `logs/eryx.pdb.GaussianNetworkModel._state_before_compute_hessian.log`<br>`logs/eryx.pdb.GaussianNetworkModel._state_after_compute_hessian.log` | State-based | rtol=1e-5, atol=1e-8 |
| `models_torch.GaussianNetworkModel.compute_K` | `logs/eryx.pdb.GaussianNetworkModel._state_before_compute_K.log`<br>`logs/eryx.pdb.GaussianNetworkModel._state_after_compute_K.log` | State-based | rtol=1e-5, atol=1e-8 |
| `models_torch.GaussianNetworkModel.compute_Kinv` | `logs/eryx.pdb.GaussianNetworkModel._state_before_compute_Kinv.log`<br>`logs/eryx.pdb.GaussianNetworkModel._state_after_compute_Kinv.log` | State-based | rtol=1e-5, atol=1e-8 |
| `models_torch.AtomicModel._get_xyz_asus` | `logs/eryx.pdb._get_xyz_asus.log` | Function-based | rtol=1e-5, atol=1e-8 |
| `models_torch.AtomicModel.flatten_model` | `logs/eryx.pdb.flatten_model.log` | Function-based | exact match |
| `models_torch.Crystal.get_asu_xyz` | `logs/eryx.pdb.get_asu_xyz.log` | Function-based | rtol=1e-5, atol=1e-8 |

## System Boundaries and Differentiability Constraints

Based on the components specified in `to_convert.json`, we distinguish between components that should remain in NumPy and those that should be converted to PyTorch with differentiability:

### Non-Differentiable Components (Keep in NumPy)
- **Data Loading**: PDB loading, cell parameter extraction, symmetry operations
- **Preprocessing**: Frame extraction, form factor extraction, neighbor list construction
- **Postprocessing**: Result saving, visualization, statistical analysis

### Differentiable Components (Convert to PyTorch)
- **Core Physics Simulation**: Matrix construction, Hessian computation, phonon calculations
- **Structure Factor Calculation**: Form factors, structure factors, complex operations
- **Model Application**: `OnePhonon.apply_disorder()` for diffuse scattering calculation
- **Grid Operations**: Grid generation, resolution operations

## Adapter Usage Examples

### Example 1: Converting AtomicModel to Tensor Dictionary

```python
def convert_atomic_model_example(pdb_path, device=None):
    # Create NumPy AtomicModel
    np_model = AtomicModel(pdb_path, expand_p1=True)
    
    # Initialize adapter
    pdb_adapter = PDBToTensor(device=device)
    
    # Convert to tensor dictionary
    model_data = pdb_adapter.convert_atomic_model(np_model)
    
    # Access tensor attributes with gradient support
    xyz_tensor = model_data['xyz']  # PyTorch tensor
    cell_tensor = model_data['cell']  # PyTorch tensor
    
    return model_data
```

### Example 2: OnePhonon Initialization with Adapters

```python
def init_one_phonon_with_adapters(pdb_path, device=None):
    # Import NumPy class
    from eryx.pdb import AtomicModel
    from eryx.adapters import PDBToTensor, GridToTensor
    
    # Initialize adapters
    pdb_adapter = PDBToTensor(device=device)
    grid_adapter = GridToTensor(device=device)
    
    # Create NumPy model
    np_model = AtomicModel(pdb_path, expand_p1=True)
    
    # Convert for PyTorch usage
    model_data = pdb_adapter.convert_atomic_model(np_model)
    
    # This would then be used in the OnePhonon.__init__ implementation
    return model_data
```

### Example 3: State-Based Testing with Adapters

```python
def test_one_phonon_method(method_name, device=None):
    # Load before state
    before_state = Logger.loadStateLog(f"logs/eryx.models.OnePhonon._state_before_{method_name}.log")
    
    # Initialize PyTorch object with state
    torch_obj = TorchTesting.initializeFromState(OnePhonon, before_state, device=device)
    
    # Call method
    getattr(torch_obj, method_name)()
    
    # Load expected after state
    expected_state = Logger.loadStateLog(f"logs/eryx.models.OnePhonon._state_after_{method_name}.log")
    
    # Compare states
    result = TorchTesting.compareStates(expected_state, torch_obj)
    
    return result
```

## Revised Implementation Plan

### Phase 1: Core Utilities Implementation (2 weeks)

#### Task 1.1: Implement ComplexTensorOps in torch_utils.py
- Implement differentiable complex exponential for phase calculations
- Implement complex multiplication with gradient preservation
- Implement complex absolute square value calculation
- Implement Debye-Waller factor calculation
- Add comprehensive tests verifying gradient flow

**Prerequisites**: None  
**Deliverables**: 
- ComplexTensorOps class in torch_utils.py
- Unit tests in test_torch_utils.py

**Component Interactions:**
- Output used by: scatter_torch.structure_factors_batch
- Critical for: Phase calculations, structure factors
- Gradient flow: Must support backpropagation through complex operations

**Testing Approach:**
- Unit tests with known input/output values (e.g., e^(iπ/2) = i)
- Gradient validation using finite differences
- No ground truth data needed (pure utility class)

#### Task 1.2: Implement EigenOps in torch_utils.py
- Implement SVD-based approach for eigenvalue decomposition
- Implement proper gradient handling for degenerate eigenvalues
- Implement linear system solver with gradient support
- Document gradient flow limitations and stability considerations

**Prerequisites**: None  
**Deliverables**: 
- EigenOps class in torch_utils.py
- Unit tests in test_torch_utils.py

**Component Interactions:**
- Output used by: OnePhonon.compute_gnm_phonons()
- Critical for: Phonon mode calculation, covariance matrix
- Gradient flow: Must support backpropagation through eigendecomposition

**Testing Approach:**
- Unit tests with known matrices and their eigendecomposition
- Gradient validation using finite differences
- No ground truth data needed (pure utility class)

#### Task 1.3: Implement GradientUtils in torch_utils.py
- Create finite difference validation tools
- Implement gradient norm calculations
- Add gradient visualization helpers
- Document validation methodology and appropriate tolerances

**Prerequisites**: None  
**Deliverables**: 
- GradientUtils class in torch_utils.py
- Utility tests in test_torch_utils.py

**Component Interactions:**
- Used for: Validating gradients in all model components
- Critical for: Verifying analytical gradient implementations
- Flow: Compare analytical gradients with numerical approximations

**Testing Approach:**
- Unit tests with simple functions with known gradients
- Test gradient computation accuracy with different step sizes
- No ground truth data needed (pure utility class)

### Phase 2: Adapter Implementation (1 week)
Status Update: PDBToTensor (methods convert_atomic_model, array_to_tensor, and convert_dict_of_arrays) is fully implemented, while convert_crystal() and convert_gnm() remain unimplemented. GridToTensor's convert_grid() and convert_mask() as well as TensorToNumpy are implemented; however, convert_symmetry_ops() is not implemented. In ModelAdapters, adapt_one_phonon_inputs and adapt_one_phonon_outputs are complete while adapt_rigid_body_translations_inputs is not implemented.

#### Task 2.1: Implement PDBToTensor in adapters.py
- Implement convert_atomic_model() method with specific tensor conversions
- Implement methods to handle AtomicModel._get_xyz_asus
- Implement methods to handle AtomicModel.flatten_model
- Implement methods to handle Crystal.get_asu_xyz
- Implement convert_crystal() and convert_gnm() methods
- Support explicit device placement with proper defaults
- Add comprehensive docstrings describing tensor shapes and types

**Prerequisites**: Phase 1 complete  
**Deliverables**: 
- PDBToTensor class in adapters.py
- Unit tests in test_adapters.py

**Component Interactions:**
- Input from: AtomicModel, Crystal, GaussianNetworkModel
- Output to: PyTorch model implementations
- Critical conversions: Coordinates, form factors, cell parameters
- Gradient flow: Preserve structure for backpropagation

**Testing Approach:**
- Unit tests with sample PDB data
- Verify correct conversion of arrays to tensors
- Test device placement and gradient enablement
- No ground truth data needed (pure utility class)

#### Task 2.2: Implement GridToTensor in adapters.py
- Implement convert_grid() method with gradient preservation
- Implement convert_mask() method
- Document which components need to be differentiable
- Add tests verifying conversion correctness and gradient preservation

**Prerequisites**: Phase 1 complete  
**Deliverables**: 
- GridToTensor class in adapters.py
- Unit tests in test_adapters.py

**Component Interactions:**
- Input from: Grid parameters, resolution masks
- Output to: map_utils_torch functions
- Critical conversions: q-grid, resolution mask
- Gradient flow: Grid points need gradients, masks typically don't

**Testing Approach:**
- Unit tests with sample grid data
- Verify correct conversion of arrays to tensors
- Test propagation of requires_grad property
- No ground truth data needed (pure utility class)

#### Task 2.3: Implement TensorToNumpy in adapters.py
- Implement tensor_to_array() with proper detachment
- Implement convert_intensity_map()
- Document shape handling and any detach/clone operations
- Add tests for various tensor types and shapes

**Prerequisites**: Phase 1 complete  
**Deliverables**: 
- TensorToNumpy class in adapters.py
- Unit tests in test_adapters.py

**Component Interactions:**
- Input from: PyTorch model outputs
- Output to: NumPy arrays for visualization
- Critical conversions: Intensity maps
- Gradient flow: N/A (one-way conversion)

**Testing Approach:**
- Unit tests with various tensor shapes and types
- Verify correct detachment and CPU conversion
- Test handling of complex tensors and nested structures
- No ground truth data needed (pure utility class)

### Phase 3: Core Function Implementation (2 weeks)

#### Task 3.1: Implement map_utils_torch.py
- Implement generate_grid() using PyTorch tensor operations
- Implement compute_resolution() for resolution calculations
- Implement get_resolution_mask() for masking the grid
- Ensure all functions preserve gradient information

**Prerequisites**: Phase 2 complete  
**Deliverables**: 
- PyTorch implementations in map_utils_torch.py
- Ground truth tests in test_map_utils_torch.py

**Component Interactions:**
- Input from: GridToTensor adapter
- Output to: scatter_torch, OnePhonon model
- Critical operations: Grid generation, resolution handling
- Gradient flow: Must preserve q-vector derivatives

**Testing Approach:**
- Test against ground truth data in `logs/eryx.map_utils.*.log`
- Use `TorchTesting.testTorchCallable()` with appropriate tolerances
- Add gradient validation for differentiable functions
- Test performance with various input sizes

#### Task 3.2: Implement scatter_torch.py
- Implement compute_form_factors() using ComplexTensorOps
- Implement structure_factors_batch() with gradient preservation
- Implement structure_factors() with efficient batching
- Document tensor shapes and gradient requirements

**Prerequisites**: Tasks 1.1 and 2.1-2.2 complete  
**Deliverables**: 
- PyTorch implementations in scatter_torch.py
- Ground truth tests in test_scatter_torch.py

**Component Interactions:**
- Uses: ComplexTensorOps for complex operations
- Input from: PDBToTensor (atomic data), map_utils_torch (q-grid)
- Output to: OnePhonon model
- Gradient flow: Through complex exponentials and phase factors

**Testing Approach:**
- Test against ground truth data in `logs/eryx.scatter.*.log`
- Use `TorchTesting.testTorchCallable()` with specific tolerances (rtol=1e-4, atol=1e-7)
- Test with various batch sizes and input configurations
- Verify gradient flow through complex number operations

### Phase 4: OnePhonon Model Implementation (3 weeks)

#### Task 4.1.1: Implement OnePhonon Initialization
- Implement __init__ method with device handling
- Implement _setup method with adapter calls
- Implement _setup_phonons for initialization
- Import original NumPy classes and use adapters for conversion

**Prerequisites**: Phase 3 complete  
**Deliverables**: 
- Initialization methods in models_torch.py
- State-based tests in test_models_torch.py

**Testing Approach:**
- Test against state-based ground truth data
- Verify proper device placement
- Test parameter initialization

#### Task 4.1.2: Implement Matrix Construction Methods
- Implement _build_A method for displacement projection
- Implement _build_M method for mass matrix
- Implement _build_M_allatoms for atomic masses
- Implement _project_M for projecting mass matrix

**Prerequisites**: Task 4.1.1 complete  
**Deliverables**: 
- Matrix construction methods in models_torch.py
- State-based tests in test_models_torch.py

**Testing Approach:**
- Test against state-based ground truth data
- Verify matrix shapes and values match expected state
- Test gradient flow through matrices

#### Task 4.1.3: Implement K-vector Methods
- Implement _build_kvec_Brillouin for k-vector generation
- Implement _center_kvec for centering operations
- Implement _at_kvec_from_miller_points for index mapping

**Prerequisites**: Task 4.1.2 complete  
**Deliverables**: 
- K-vector methods in models_torch.py
- Mixed function-based and state-based tests

**Testing Approach:**
- Test _center_kvec and _at_kvec_from_miller_points using function-based tests
- Test _build_kvec_Brillouin using state-based tests
- Verify vector shapes and values

#### Task 4.1.4: Implement Phonon Calculation Methods
- Implement compute_hessian method
- Implement compute_gnm_phonons using EigenOps
- Implement GaussianNetworkModel.compute_K
- Implement GaussianNetworkModel.compute_Kinv
- Use adapter pattern for accessing original GNM methods where needed

**Prerequisites**: Tasks 1.2 and 4.1.3 complete  
**Deliverables**: 
- Phonon calculation methods in models_torch.py
- State-based tests in test_models_torch.py

**Testing Approach:**
- Test using state-based testing
- Verify eigenvalues and eigenvectors match expected state
- Test with different parameters
- Validate eigendecomposition gradients

#### Task 4.1.5: Implement Covariance Matrix Calculation
- Implement compute_covariance_matrix method
- Ensure proper scaling to match experimental ADPs
- Add gradient validation

**Prerequisites**: Task 4.1.4 complete  
**Deliverables**: 
- Covariance calculation method in models_torch.py
- State-based tests in test_models_torch.py

**Testing Approach:**
- Test using state-based testing
- Verify matrix values match expected state
- Test gradient flow

#### Task 4.1.6: Implement Apply Disorder Method
- Implement apply_disorder with end-to-end gradient flow
- Integrate with structure factor calculations
- Optimize for memory efficiency

**Prerequisites**: Tasks 3.2 and 4.1.5 complete  
**Deliverables**: 
- apply_disorder method in models_torch.py
- State-based tests in test_models_torch.py

**Testing Approach:**
- Test using state-based testing
- Verify final intensity values match expected state
- Test end-to-end gradient flow
- Benchmark performance

### Phase 5: Integration and Testing (2 weeks)

#### Task 5.1: Extend Testing Framework for State-Based Testing
- Implement state capturing in the Logger class
- Add state serialization/deserialization methods
- Create state comparison utilities with tolerances
- Implement method to initialize objects from state data

**Prerequisites**: Phase 1 complete  
**Deliverables**: 
- Extended Logger class with state capabilities
- Extended TorchTesting class with state-based methods
- Unit tests for state capture and comparison

**Component Interactions:**
- Inputs: Object states, class definitions
- Uses: Serialization/deserialization system
- Output: Test results comparing states
- Demonstrates: Comprehensive state validation

**Testing Approach:**
- Create test fixtures with known state changes
- Verify state capture and restoration accuracy
- Test with different levels of state complexity
- Validate numerical comparison methods

#### Task 5.2: Generate State-Based Ground Truth

- Extend run_debug.py to capture complete object states
- Generate state logs for OnePhonon methods
- Ensure proper state attribute filtering for large objects
- Verify state log completeness

**Prerequisites**: Task 5.1 complete  
**Deliverables**: 
- Complete set of state logs for stateful methods
- Verification scripts for state logs
- Documentation of state log format

**Component Interactions:**
- Uses: Extended Logger class for state capture
- Creates: State logs for stateful objects
- Demonstrates: Full object state tracking

**Testing Approach:**
- Verify state logs capture all relevant attributes
- Check log file naming and organization
- Validate state completeness for restoration

#### Task 5.3: Implement run_torch.py
- Create end-to-end simulation script mirroring run_debug.py
- Add device management and error handling
- Implement example of OnePhonon model execution
- Document workflow with tensor operations
- Include adapter usage for NumPy<->PyTorch conversion

**Prerequisites**: Phase 4 complete and state-based testing implemented  
**Deliverables**: 
- run_torch.py script
- Integration tests in test_integration.py

**Component Interactions:**
- Inputs: Simulation parameters, PDB files
- Uses: OnePhonon model implementation with adapters
- Output: Diffuse intensity maps
- Demonstrates: End-to-end gradient flow

**Testing Approach:**
- Compare end-to-end output with NumPy implementation
- Verify matching output with ground truth data
- Test with different parameter configurations
- Benchmark performance against NumPy implementation

#### Task 5.4: Implement Comprehensive Testing
- Create component-level tests for all implementations
- Add gradient validation tests using GradientUtils
- Implement integration tests for OnePhonon model
- Add performance benchmarks comparing NumPy and PyTorch versions

**Prerequisites**: Task 5.3 complete  
**Deliverables**: 
- Comprehensive test suite
- Performance benchmarks

**Component Interactions:**
- Tests: All component interactions in the architecture
- Uses: Ground truth data from NumPy implementation
- Validates: Correctness and gradient computation
- Ensures: Compatibility across component boundaries

**Testing Approach:**
- Create test modules for each component
- Use test template for ground truth validation
- Add specific gradient tests for differentiable components
- Test end-to-end workflow with OnePhonon model

## Implementation Priorities

Based on the component interactions and critical differentiability points:

1. **ComplexTensorOps**: Foundation for structure factor calculations
2. **EigenOps**: Critical for phonon calculations in OnePhonon
3. **PDBToTensor and GridToTensor**: Needed for proper input representation
4. **scatter_torch**: Core for structure factor calculations
5. **OnePhonon model**: Main model with eigendecomposition challenges
6. **Extended Testing Framework**: Essential for state-based testing of stateful components

## Timeline and Dependencies

- **Phase 1 (2 weeks)**: Core Utilities - Can start immediately with ground truth data
- **Phase 2 (1 week)**: Adapters - Depends on Phase 1 for tensor operations
- **Phase 3 (2 weeks)**: Core Functions - Depends on Phases 1-2
- **Phase 4 (3 weeks)**: OnePhonon Model - Depends on Phases 1-3
- **Phase 5 (2 weeks)**: Integration and Testing - Depends on all previous phases, with Tasks 5.1-5.2 ideally starting earlier

**Total Timeline: 10 weeks**

## Final Verification Checklist

To ensure completeness, each item in this checklist must be verified:

- [ ] All functions from `to_convert.json` have PyTorch implementations
- [ ] All implementations pass ground truth tests with specified tolerances:
  - [ ] Function-based tests for pure functions
  - [ ] State-based tests for stateful methods
- [ ] All differentiable operations support gradient calculation
- [ ] End-to-end gradient flow is verified through the entire OnePhonon model
- [ ] All implementations handle both CPU and GPU execution
- [ ] Performance is benchmarked against NumPy implementation
- [ ] Documentation is complete with proper type hints and docstrings
- [ ] State-based testing infrastructure is working correctly
- [ ] All state logs are complete and valid for testing
- [ ] All adapter conversions maintain gradient flow and device placement
- [ ] Import patterns follow the adapter-based approach convention
