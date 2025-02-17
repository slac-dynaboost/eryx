# Test Infrastructure Specification

## High-Level Objective
- Create comprehensive test infrastructure for the OnePhonon model including config generation, log capture/analysis, and reference data generation.

## Mid-Level Objectives
- Build config generation system for test cases
- Create log capture and analysis framework
- Generate reference data for tests
- Establish test data directory structure

## Implementation Notes

### Dependencies and Requirements
- pytest 
- numpy for reference data
- logging module
- Core eryx codebase
- Sample PDB: tests/pdbs/5zck.pdb
- PyYAML for config handling

### Technical Guidance
- Create utilities first, then generate test data
- Reference data should be repeatable/deterministic
- Log capture should be non-invasive
- Config generator should support both valid and edge cases
- All files under tests/test_utils/

## Context

### Beginning Context
```
/eryx
  /tests
    /pdbs
      5zck.pdb
```

### Ending Context
```
/eryx
  /tests
    /test_utils
      config_generator.py
      generate_reference_data.py  
      generate_test_logs.py
      log_analysis.py
      log_capture.py
      model_runner.py
    /test_data
      /configs
        base_config.yaml
        /edge_case_configs
          placeholder_config.yaml
      /logs
        /base_run
          base_run.log
        /edge_cases
          edge_run_0.log
          edge_run_1.log
          edge_run_2.log
      /reference
        form_factors.npy
        structure_factors.npy
        diffraction_pattern.npy
        test_params.npz
```

## Low-Level Tasks

1. Create Config Generator
```aider
CREATE tests/test_utils/config_generator.py:
  ADD generate_base_config() function:
    - Returns dict with valid default config
    - Includes all required OnePhonon parameters
    - Uses test PDB path
  
  ADD generate_edge_case_configs() function:
    - Returns list of variant configs
    - Includes boundary conditions
    - Includes invalid parameters
```

2. Create Log Management
```aider
CREATE tests/test_utils/log_capture.py:
  ADD LogCapture class (extends logging.Handler):
    - Captures logs in memory
    - Preserves log formatting
    - Provides access to captured logs
  
  ADD LogStore class:
    - Handles log persistence
    - Saves logs to files
    - Loads logs from files

CREATE tests/test_utils/log_analysis.py:
  ADD LogAnalyzer class:
    - Parses log entries
    - Extracts method arguments
    - Validates log sequences
    - Gets error messages
```

3. Create Model Runner
```aider
CREATE tests/test_utils/model_runner.py:
  ADD ModelRunner class:
    - Takes config dict
    - Sets up logging
    - Runs OnePhonon model
    - Captures run metadata
    - Handles errors
```

4. Create Reference Data Generator 
```aider
CREATE tests/test_utils/generate_reference_data.py:
  ADD generate_reference_data() function:
    - Creates test case with small grid
    - Generates form factors
    - Generates structure factors
    - Generates diffraction pattern
    - Saves test parameters
```

5. Create Log Generator
```aider
CREATE tests/test_utils/generate_test_logs.py:
  ADD main() function:
    - Generates logs for base config
    - Generates logs for edge cases
    - Saves logs to test_data directory
```

Each task must include:
- Complete error handling
- Input validation 
- Docstrings
- Type hints
- Logging of key operations

Config Parameters for Base Case:
```python
base_config = {
    "setup": {
        "pdb_path": "tests/pdbs/5zck.pdb",
        "root_dir": "test_output",
        "hsampling": [-4, 4, 1],
        "ksampling": [-17, 17, 1], 
        "lsampling": [-29, 29, 1],
        "res_limit": 0,
        "batch_size": 10000,
        "n_processes": 8,
    },
    "OnePhonon": {
        "gnm_cutoff": 4.0,
        "gamma_intra": 1.0,
        "gamma_inter": 1.0,
        "expand_p1": True
    }
}
```

Edge Cases to Include:
```python
edge_cases = [
    # Low resolution limit
    {"setup": {"res_limit": 5.0}},
    
    # High batch size/processes
    {"setup": {"batch_size": 20000, "n_processes": 16}},
    
    # Invalid gamma
    {"OnePhonon": {"gamma_inter": -1.0}}
]
```

Expected Log Format for Reference Data Generation:
```
[DEBUG] Generating form factors for q points: 0 to 9
[DEBUG] First 5 form factor values: [x1, x2, x3, x4, x5]
[DEBUG] Generating structure factors
[DEBUG] First element real: x, imag: y
[DEBUG] Generating diffraction pattern
[DEBUG] Non-nan values in pattern: n
```
