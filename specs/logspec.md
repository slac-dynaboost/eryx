# Logging Implementation Specification 

## High-Level Objective
- Implement non-invasive, comprehensive logging throughout the OnePhonon model implementation to enable testing and validation

## Mid-Level Objectives
- Create logging utilities and decorators
- Add logging to all key methods
- Capture computational values and array shapes
- Enable test validation of computation chain

## Implementation Notes

### Dependencies and Requirements
- Python logging module
- functools for decorators
- numpy for array handling
- contextlib for context managers
- time module for duration tracking

### Technical Guidance
- Use decorators exclusively for method logging
- Log method entry/exit pairs
- Include array shapes and key values
- Keep original code structure intact
- Ensure thread safety
- Capture duration of operations

## Logging Format Specification

1. Method Entry Logs
```python
"[{class_name}.{method_name}] Enter with args={args}, kwargs={kwargs}"
```

2. Method Exit Logs
```python
"[{class_name}.{method_name}] Exit. Duration={duration:.2f}s"
```

3. Return Value Logs
```python
"[{class_name}.{method_name}] Return value: {repr(result)}"
# For arrays, include shape and sample:
"[{class_name}.{method_name}] Return array shape: {shape}, first few values: {values[:5]}"
```

4. Array Shape Logs
```python
"[Array Shape] {array_name}.shape={shape}"
# For None arrays:
"[Array Shape] {array_name} is None"
```

5. Computation Value Logs
```python
"[{class_name}.{method_name}] {calculation_name}={value:.6f}"
```

6. Error Logs
```python
"[{class_name}.{method_name}] Error: {error_message}"
```

## Context

### Beginning Context
```
/eryx
  /eryx
    models.py
    pdb.py
  /tests
    /test_utils
      log_capture.py
      log_analysis.py
```

### Ending Context
```
/eryx
  /eryx
    logging_utils.py (new)
    models.py (with logging)
    pdb.py (with logging)
```

## Low-Level Tasks

1. Create Logging Utilities
```aider
CREATE eryx/logging_utils.py:
  ADD logging configuration:
    format: '[%(levelname)s] [%(asctime)s] [%(name)s] %(message)s'
    level: DEBUG
    
  ADD log_method_call decorator:
    - Logs method entry with args/kwargs
    - Times execution
    - Logs method exit with duration
    - Logs return value appropriately
    - Handles array returns specially
    
  ADD TimedOperation context manager:
    - Times block execution
    - Logs operation duration
    
  ADD log_property_access decorator:
    - Logs property access
    - Logs value
    
  ADD log_array_shape function:
    - Handles None arrays
    - Logs array shapes
```

2. Add Core Logging 
```aider
UPDATE eryx/models.py:
  IMPORT logging_utils
  ADD decorators to OnePhonon methods:
    - __init__
    - _setup
    - _setup_phonons
    - apply_disorder
    - _build_A
    - _build_M
    - _build_kvec_Brillouin
    - compute_gnm_phonons
  
  ADD array shape logging:
    - xyz arrays
    - grid arrays
    - matrices
```

3. Add Interaction Logging
```aider
UPDATE eryx/pdb.py:
  IMPORT logging_utils
  ADD decorators to methods:
    AtomicModel:
      - __init__
      - _get_gemmi_structure
      - _extract_cell
      - _get_sym_ops
      - extract_frame
    
    GaussianNetworkModel:
      - __init__
      - _setup_atomic_model
      - _setup_gaussian_network_model
      - build_gamma
      - build_neighbor_list
      - compute_hessian
```

Core Method Logging Requirements:

1. OnePhonon.__init__:
- Log all input parameters
- Log array initialization

2. _setup:
- Log grid generation
- Log array shapes
- Log atomic model setup

3. apply_disorder:
- Log computation stages
- Log array shapes
- Log key calculation values

4. _build_A and _build_M:
- Log matrix shapes
- Log completion

5. compute_hessian:
- Log matrix shapes
- Log eigenvalues

The focus is on capturing the full computation chain while maintaining clean code structure through decorators and utilities.
