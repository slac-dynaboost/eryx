# State-Based Testing in Eryx

This document describes how to use the state-based testing framework for the PyTorch port of Eryx.

## Overview

State-based testing captures object state before and after method execution, allowing for:
- Testing stateful components like the OnePhonon model
- Verifying PyTorch implementations against NumPy ground truth
- Validating internal state changes during complex operations

## Implementation Status

The state-based testing framework is now **fully implemented** and operational. All components of the framework are complete:

- ✅ StateCapture class for capturing object state
- ✅ Debug decorator with state capture support
- ✅ Log generation and verification scripts
- ✅ State-based test helpers and utilities
- ✅ Serialization framework for complex objects
- ✅ Test implementations for all OnePhonon methods

All matrix construction methods (_build_A, _build_M, _build_M_allatoms, _project_M) have been implemented and tested using the state-based approach, as confirmed by the passing tests in `tests/test_models_torch.py`.

## Using State Capture

### Basic Usage

Methods are decorated with `@debug` to capture state:

```python
from eryx.autotest.debug import debug

class MyClass:
    @debug
    def my_method(self, param):
        # Method implementation that changes object state
        self.value = param
```

### Configuration Options

The decorator accepts configuration parameters:

```python
@debug(capture_state=True, max_depth=5, exclude_attrs=["temp_data", "_private_attr"])
def complex_method(self, param):
    # Method implementation
    pass
```

Options:
- `capture_state`: Enable/disable state capture (default: True)
- `max_depth`: Maximum recursion depth for nested objects (default: 10)
- `exclude_attrs`: List of attribute patterns to exclude

## Generated Log Files

Running code with decorated methods produces:

1. Regular function logs: `logs/eryx.module.function.log`
2. Before state logs: `logs/eryx.module.Class._state_before_method.log`
3. After state logs: `logs/eryx.module.Class._state_after_method.log`

## Serialization Framework

The state-based testing framework uses `ObjectSerializer` for robust serialization of complex objects:

- **Consistent Format**: All state logs use a consistent JSON-based format
- **Complex Object Support**: Handles NumPy arrays, PyTorch tensors, Gemmi objects, and custom classes
- **Type Preservation**: Maintains type information for proper reconstruction
- **Gradient Support**: Preserves gradient requirements for tensors
- **Dill Integration**: Uses dill for serializing complex objects with functions and lambdas

### Inspecting State Logs

Use the `inspect_state_log.py` script to examine log contents:

```bash
# View log in tree format
python scripts/inspect_state_log.py logs/eryx.module.Class._state_method.log

# View in JSON format
python scripts/inspect_state_log.py logs/eryx.module.Class._state_method.log --format json
```

## Verifying Logs

Use the `verify_logs.py` script to check log completeness:

```bash
# Basic verification
python scripts/verify_logs.py

# Verify with required attributes
python scripts/verify_logs.py --required-attrs "q_grid,map_shape,hkl_grid"

# Save detailed results to file
python scripts/verify_logs.py --output verification_results.json
```

## State-Based Testing Pattern

In test files, use this pattern:

```python
def test_method_state_based(self):
    # 1. Load before state
    before_state = load_test_state(self.logger, module_name, class_name, method_name)
    
    # 2. Build test object with StateBuilder
    model = build_test_object(TorchClass, before_state, device=self.device)
    
    # 3. Call method under test
    model.method()
    
    # 4. Verify results (attributes, tensor properties, etc.)
    self.assertTrue(hasattr(model, 'expected_attribute'))
    
    # 5. Load after state and compare if needed
    after_state = load_test_state(self.logger, module_name, class_name, method_name, before=False)
    expected_tensor = ensure_tensor(after_state.get('tensor_attr'), device='cpu')
    self.assertTrue(np.allclose(model.tensor_attr.detach().cpu().numpy(), expected_tensor))
```

> **Example Implementation**: For a complete working example of state-based testing, see the `test_build_A_state_based()` method in `tests/test_models_torch.py` and `test_build_kvec_Brillouin_state_based()` method in `tests/test_models_torch_kvector.py`. These tests demonstrate proper state loading, tensor comparison, and error handling with flexible path resolution.

## Regenerating State Logs

When you make changes to the code that affect state capture, regenerate the logs:

```bash
# Generate logs for all components
python scripts/generate_state_logs.py --component all

# Generate logs for specific components
python scripts/generate_state_logs.py --component onePhonon
```

## Best Practices

1. **Be Selective**: Only capture necessary attributes with `exclude_attrs`
2. **Test Important State**: Check method-specific attributes in verification
3. **Match Ground Truth**: Ensure PyTorch implementation matches the NumPy version's state changes
4. **Document State Dependencies**: Note which attributes methods read from and modify
5. **Handle Complex Objects**: Use `ensure_tensor()` to convert state values to tensors
6. **Verify Logs**: Always run `verify_logs.py` after regenerating logs
7. **Inspect Problematic Logs**: Use `inspect_state_log.py` to debug issues
8. **Proper Element Serialization**: Ensure element symbols are properly extracted from Gemmi objects
9. **Use Dill for Complex Objects**: For objects with functions or lambdas like GaussianNetworkModel, use dill serialization
# State-Based Testing with StateBuilder

This document describes the approach to state-based testing using the StateBuilder pattern and ObjectSerializer.

## Implementation Status

The StateBuilder pattern and related components are now **fully implemented** and operational:

- ✅ StateBuilder class for constructing test objects from state data
- ✅ Test helper functions for loading state and building objects
- ✅ Gradient flow verification utilities
- ✅ Tensor conversion utilities for state data
- ✅ Comprehensive test implementations using this pattern

All matrix construction methods (_build_A, _build_M, etc.) have been successfully tested using this approach, as demonstrated by the passing tests in `tests/test_models_torch.py`.

## Overview

State-based testing involves:
1. Capturing object state before and after method execution
2. Using that state to reconstruct test objects
3. Verifying method behavior by comparing with expected results

The approach ensures test objects have the correct structure, with attributes in their expected locations, while leveraging the robust ObjectSerializer for consistent serialization.

## Key Components

### ObjectSerializer

`ObjectSerializer` is the foundation for state serialization:

- Handles complex Python objects including NumPy arrays, PyTorch tensors, and custom classes
- Preserves type information for accurate reconstruction
- Supports special handling for domain-specific objects like Gemmi structures
- Provides consistent serialization format across all components

### StateBuilder

`StateBuilder` is the core class that builds properly structured test objects:

```python
from eryx.autotest.state_builder import StateBuilder
from eryx.models_torch import OnePhonon

# Load state data
state_data = logger.loadStateLog("logs/eryx.models.OnePhonon._state_before__build_kvec_Brillouin.log")

# Build test object
builder = StateBuilder(device=torch.device('cpu'))
model = builder.build(OnePhonon, state_data)
```

### Test Helpers

Helper functions simplify common testing operations:

```python
from eryx.autotest.test_helpers import (
    load_test_state,
    build_test_object,
    verify_gradient_flow,
    ensure_tensor
)

# Load state and build object in one step
before_state = load_test_state(logger, 'eryx.models', 'OnePhonon', '_build_kvec_Brillouin')
model = build_test_object(OnePhonon, before_state, device=device)

# Call method under test
model._build_kvec_Brillouin()

# Verify gradient flow
loss = torch.sum(model.kvec)
loss.backward()
assert verify_gradient_flow(model.kvec, model.model.A_inv)

# Convert state values to tensors
expected_tensor = ensure_tensor(after_state.get('tensor_attr'), device='cpu')
```

## Standard Test Pattern

Follow this pattern for state-based tests:

```python
def test_method_name(self):
    # 1. Load state data
    before_state = load_test_state(self.logger, module_name, class_name, method_name)
    
    # 2. Build test object 
    model = build_test_object(TorchClass, before_state, device=self.device)
    
    # 3. Call method under test
    method = getattr(model, method_name)
    method()
    
    # 4. Verify method effects
    # (Check attributes, tensor properties, etc.)
    self.assertTrue(hasattr(model, 'expected_attribute'))
    self.assertEqual(model.tensor.shape, expected_shape)
    
    # 5. Verify gradient flow
    loss = torch.sum(model.output_tensor)
    loss.backward()
    self.assertIsNotNone(model.input_tensor.grad)
    
    # 6. Compare with expected state (if needed)
    after_state = load_test_state(self.logger, module_name, class_name, method_name, before=False)
    expected_tensor = ensure_tensor(after_state.get('tensor_attr'), device='cpu')
    self.assertTrue(np.allclose(
        model.tensor_attr.detach().cpu().numpy(), 
        expected_tensor,
        rtol=1e-5, atol=1e-8
    ))
```

## Handling Common Patterns

### Testing Matrix Construction

For methods that build matrices like `_build_A`, `_build_M`:

```python
# 1. Load state and build model
before_state = load_test_state(self.logger, 'eryx.models', 'OnePhonon', '_build_A')
model = build_test_object(OnePhonon, before_state)

# 2. Call method
model._build_A()

# 3. Verify Amat tensor properties
self.assertTrue(hasattr(model, 'Amat'))
self.assertEqual(model.Amat.shape, expected_shape)
self.assertTrue(model.Amat.requires_grad)

# 4. Verify gradient flow
loss = torch.sum(model.Amat)
loss.backward()
# Check relevant input tensor gradients
```

### Testing GaussianNetworkModel

For testing GaussianNetworkModel with dill serialization:

```python
# 1. Load state with dill support
before_state = load_test_state(self.logger, 'eryx.pdb', 'GaussianNetworkModel', 'compute_hessian')
model = build_test_object(GaussianNetworkModel, before_state)

# 2. Call method
hessian = model.compute_hessian()

# 3. Verify result properties
self.assertIsInstance(hessian, torch.Tensor)
self.assertEqual(hessian.dtype, torch.complex64)
self.assertEqual(hessian.shape, expected_shape)

# 4. Compare with expected output
after_state = load_test_state(self.logger, 'eryx.pdb', 'GaussianNetworkModel', 'compute_hessian', before=False)
expected_hessian = ensure_tensor(after_state.get('hessian'), device='cpu')
self.assertTrue(np.allclose(
    hessian.detach().cpu().numpy(),
    expected_hessian.detach().cpu().numpy(),
    rtol=1e-5, atol=1e-8
))
```

### Testing K-vector Methods

For k-vector calculation methods:

```python
# 1. Load state and build model
before_state = load_test_state(self.logger, 'eryx.models', 'OnePhonon', '_build_kvec_Brillouin')
model = build_test_object(OnePhonon, before_state)

# 2. Verify A_inv is in the right place
self.assertTrue(hasattr(model, 'model') and hasattr(model.model, 'A_inv'))

# 3. Call method
model._build_kvec_Brillouin()

# 4. Verify kvec tensors and gradient flow
self.assertTrue(hasattr(model, 'kvec'))
# ... more checks
```

## Inspecting and Debugging State Logs

When working with state logs:

1. **Inspect Log Contents**: Use the inspection tool to examine log structure
   ```bash
   python scripts/inspect_state_log.py logs/eryx.models.OnePhonon._state_before__build_kvec_Brillouin.log
   ```

2. **Verify Log Completeness**: Check that logs contain required attributes
   ```bash
   python scripts/verify_logs.py --required-attrs "model,A_inv"
   ```

3. **Debug Tensor Conversion**: Use `ensure_tensor()` to handle different tensor formats
   ```python
   # Convert any tensor-like object to a proper tensor
   tensor = ensure_tensor(state_value, device=self.device)
   ```

4. **Print Diagnostic Information**: Add debug prints to understand state structure
   ```python
   print(f"State keys: {before_state.keys()}")
   if 'model' in before_state:
       print(f"Model keys: {before_state['model'].keys()}")
   ```

## Generating and Managing State Logs

### Generating State Logs

```bash
# Generate logs for all components
python scripts/generate_state_logs.py --component all

# Generate logs for specific components
python scripts/generate_state_logs.py --component onePhonon
python scripts/generate_state_logs.py --component mapUtils
python scripts/generate_state_logs.py --component scatter
```

### Verifying State Logs

```bash
# Check all logs
python scripts/verify_logs.py

# Check with specific requirements
python scripts/verify_logs.py --required-attrs "model,A_inv,kvec,kvec_norm"

# Save detailed results to file
python scripts/verify_logs.py --output verification_results.json
```

## Best Practices

1. **Always Regenerate Logs** after significant code changes
2. **Verify Logs** before running tests to ensure they contain required data
3. **Use Helper Functions** like `ensure_tensor()` for consistent handling
4. **Set Appropriate Tolerances** for numerical comparisons (typically 1e-5 for rtol)
5. **Add Debug Prints** in tests to diagnose failures
6. **Check Attribute Locations** to ensure proper object structure
7. **Inspect Problematic Logs** using the inspection tool
