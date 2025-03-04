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
| CP5 | Matrix Construction Complete | _build_A, _build_M, _build_M_allatoms, _project_M | Ground truth tests pass |
| CP6 | K-vector Methods Complete | _build_kvec_Brillouin, _center_kvec, _at_kvec_from_miller_points | Ground truth tests pass |
| CP7 | Phonon Calculation Complete | compute_gnm_phonons, compute_hessian, GNM methods | Ground truth tests pass |
| CP8 | Covariance Matrix Complete | compute_covariance_matrix | Ground truth tests pass |
| CP9 | Apply Disorder Complete | apply_disorder | Ground truth tests pass |
| CP10 | End-to-End Integration Complete | All OnePhonon methods | Integration tests pass |

## Ground Truth Testing Strategy

### Data Storage and Access
Ground truth data is stored in the `logs/` directory, with each function's inputs and outputs captured via the `@debug` decorator. Each log file follows the naming convention `logs/eryx.module.function.log` and contains serialized inputs and expected outputs.

### Accessing Ground Truth Data
The testing framework accesses ground truth data through the following process:
1. Use `Logger.searchLogDirectory()` to find relevant log files for a component
2. Use `Logger.loadLog()` to deserialize input/output pairs for testing
3. Feed inputs through the PyTorch implementation and compare outputs

### Testing Framework
The `TorchTesting` class in `eryx/autotest/torch_testing.py` provides specialized methods for testing PyTorch implementations:
- `testTorchCallable(log_path_prefix, torch_func)`: Tests a PyTorch function against NumPy ground truth
- `create_tensor_test_case(log_path_prefix, torch_func, numpy_func)`: Creates complete test cases
- `check_gradients(torch_func, inputs)`: Validates gradient computation

### Component-to-Test Mapping

The following table maps PyTorch components to their corresponding ground truth data and testing approaches:

| PyTorch Component | Ground Truth Data | Testing Approach | Tolerances |
|-------------------|-------------------|------------------|------------|
| `ComplexTensorOps` | N/A (utility class) | Unit tests with known values | rtol=1e-5, atol=1e-8 |
| `EigenOps` | N/A (utility class) | Unit tests with known values | rtol=1e-5, atol=1e-8 |
| `map_utils_torch.generate_grid` | `logs/eryx.map_utils.generate_grid.log` | Compare grid outputs | rtol=1e-5, atol=1e-8 |
| `map_utils_torch.compute_resolution` | `logs/eryx.map_utils.compute_resolution.log` | Compare resolution values | rtol=1e-5, atol=1e-8 |
| `map_utils_torch.get_resolution_mask` | `logs/eryx.map_utils.get_resolution_mask.log` | Compare masks | exact match |
| `scatter_torch.compute_form_factors` | `logs/eryx.scatter.compute_form_factors.log` | Compare form factors | rtol=1e-4, atol=1e-7 |
| `scatter_torch.structure_factors_batch` | `logs/eryx.scatter.structure_factors_batch.log` | Compare structure factors | rtol=1e-4, atol=1e-7 |
| `scatter_torch.structure_factors` | `logs/eryx.scatter.structure_factors.log` | Compare structure factors | rtol=1e-4, atol=1e-7 |
| `models_torch.OnePhonon._build_A` | `logs/eryx.models._build_A.log` | Compare matrices | rtol=1e-5, atol=1e-8 |
| `models_torch.OnePhonon._build_M` | `logs/eryx.models._build_M.log` | Compare matrices | rtol=1e-5, atol=1e-8 |
| `models_torch.OnePhonon.compute_gnm_phonons` | `logs/eryx.models.compute_gnm_phonons.log` | Compare eigenvalues and vectors | rtol=1e-4, atol=1e-6 |
| `models_torch.OnePhonon.compute_covariance_matrix` | `logs/eryx.models.compute_covariance_matrix.log` | Compare matrices | rtol=1e-4, atol=1e-6 |
| `models_torch.OnePhonon.apply_disorder` | `logs/eryx.models.apply_disorder.log` | Compare diffuse intensity | rtol=1e-3, atol=1e-5 |
| `models_torch.OnePhonon._build_kvec_Brillouin` | `logs/eryx.models._build_kvec_Brillouin.log` | Compare vectors | rtol=1e-5, atol=1e-8 |
| `models_torch.OnePhonon._center_kvec` | `logs/eryx.models._center_kvec.log` | Compare values | rtol=1e-5, atol=1e-8 |
| `models_torch.OnePhonon._at_kvec_from_miller_points` | `logs/eryx.models._at_kvec_from_miller_points.log` | Compare indices | exact match |
| `models_torch.OnePhonon._build_M_allatoms` | `logs/eryx.models._build_M_allatoms.log` | Compare matrices | rtol=1e-5, atol=1e-8 |
| `models_torch.OnePhonon._project_M` | `logs/eryx.models._project_M.log` | Compare matrices | rtol=1e-5, atol=1e-8 |
| `models_torch.GaussianNetworkModel.compute_hessian` | `logs/eryx.pdb.compute_hessian.log` | Compare matrices | rtol=1e-5, atol=1e-8 |
| `models_torch.GaussianNetworkModel.compute_K` | `logs/eryx.pdb.compute_K.log` | Compare matrices | rtol=1e-5, atol=1e-8 |
| `models_torch.GaussianNetworkModel.compute_Kinv` | `logs/eryx.pdb.compute_Kinv.log` | Compare matrices | rtol=1e-5, atol=1e-8 |
| `models_torch.AtomicModel._get_xyz_asus` | `logs/eryx.pdb._get_xyz_asus.log` | Compare coordinates | rtol=1e-5, atol=1e-8 |
| `models_torch.AtomicModel.flatten_model` | `logs/eryx.pdb.flatten_model.log` | Compare arrays | exact match |
| `models_torch.Crystal.get_asu_xyz` | `logs/eryx.pdb.get_asu_xyz.log` | Compare coordinates | rtol=1e-5, atol=1e-8 |


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
Status Update: PDBToTensor (methods convert_atomic_model, array_to_tensor, and convert_dict_of_arrays) is fully implemented, while convert_crystal() and convert_gnm() remain unimplemented. GridToTensor’s convert_grid() and convert_mask() as well as TensorToNumpy are implemented; however, convert_symmetry_ops() is not implemented. In ModelAdapters, adapt_one_phonon_inputs and adapt_one_phonon_outputs are complete while adapt_rigid_body_translations_inputs is not implemented.

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

**Prerequisites**: Phase 3 complete  
**Deliverables**: 
- Initialization methods in models_torch.py
- Unit tests in test_models_torch.py

**Testing Approach:**
- Test against ground truth data logs
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
- Ground truth tests in test_models_torch.py

**Testing Approach:**
- Test against ground truth data logs
- Verify matrix shapes and values
- Test gradient flow through matrices

#### Task 4.1.3: Implement K-vector Methods
- Implement _build_kvec_Brillouin for k-vector generation
- Implement _center_kvec for centering operations
- Implement _at_kvec_from_miller_points for index mapping

**Prerequisites**: Task 4.1.2 complete  
**Deliverables**: 
- K-vector methods in models_torch.py
- Ground truth tests in test_models_torch.py

**Testing Approach:**
- Test against ground truth data
- Verify correct vector generation
- Test index mapping accuracy

#### Task 4.1.4: Implement Phonon Calculation Methods
- Implement compute_hessian method
- Implement compute_gnm_phonons using EigenOps
- Implement GaussianNetworkModel.compute_K
- Implement GaussianNetworkModel.compute_Kinv

**Prerequisites**: Tasks 1.2 and 4.1.3 complete  
**Deliverables**: 
- Phonon calculation methods in models_torch.py
- Ground truth tests in test_models_torch.py

**Testing Approach:**
- Test against ground truth data logs
- Verify eigenvalues and eigenvectors
- Test with different parameters
- Validate eigendecomposition gradients

#### Task 4.1.5: Implement Covariance Matrix Calculation
- Implement compute_covariance_matrix method
- Ensure proper scaling to match experimental ADPs
- Add gradient validation

**Prerequisites**: Task 4.1.4 complete  
**Deliverables**: 
- Covariance calculation method in models_torch.py
- Ground truth tests in test_models_torch.py

**Testing Approach:**
- Test against ground truth data
- Verify matrix values and properties
- Test gradient flow

#### Task 4.1.6: Implement Apply Disorder Method
- Implement apply_disorder with end-to-end gradient flow
- Integrate with structure factor calculations
- Optimize for memory efficiency

**Prerequisites**: Tasks 3.2 and 4.1.5 complete  
**Deliverables**: 
- apply_disorder method in models_torch.py
- Ground truth tests in test_models_torch.py

**Testing Approach:**
- Test against ground truth data
- Verify final intensity values
- Test end-to-end gradient flow
- Benchmark performance

### Phase 5: Integration and Testing (2 weeks)

#### Task 5.1: Implement run_torch.py
- Create end-to-end simulation script mirroring run_debug.py
- Add device management and error handling
- Implement example of OnePhonon model execution
- Document workflow with tensor operations

**Prerequisites**: Phase 4 complete  
**Deliverables**: 
- run_torch.py script
- Integration tests in test_integration.py

**Component Interactions:**
- Inputs: Simulation parameters, PDB files
- Uses: OnePhonon model implementation
- Output: Diffuse intensity maps
- Demonstrates: End-to-end gradient flow

**Testing Approach:**
- Compare end-to-end output with NumPy implementation
- Verify matching output with ground truth data
- Test with different parameter configurations
- Benchmark performance against NumPy implementation

#### Task 5.2: Implement Comprehensive Testing
- Create component-level tests for all implementations
- Add gradient validation tests using GradientUtils
- Implement integration tests for OnePhonon model
- Add performance benchmarks comparing NumPy and PyTorch versions

**Prerequisites**: Task 5.1 complete  
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

## Timeline and Dependencies

- **Phase 1 (2 weeks)**: Core Utilities - Can start immediately with ground truth data
- **Phase 2 (1 week)**: Adapters - Depends on Phase 1 for tensor operations
- **Phase 3 (2 weeks)**: Core Functions - Depends on Phases 1-2
- **Phase 4 (3 weeks)**: OnePhonon Model - Depends on Phases 1-3
- **Phase 5 (2 weeks)**: Integration - Depends on all previous phases

**Total Timeline: 10 weeks**

## Final Verification Checklist

To ensure completeness, each item in this checklist must be verified:

- [ ] All functions from `to_convert.json` have PyTorch implementations
- [ ] All implementations pass ground truth tests with specified tolerances
- [ ] All differentiable operations support gradient calculation
- [ ] End-to-end gradient flow is verified through the entire OnePhonon model
- [ ] All implementations handle both CPU and GPU execution
- [ ] Performance is benchmarked against NumPy implementation
- [ ] Documentation is complete with proper type hints and docstrings
