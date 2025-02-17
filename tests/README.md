# eryx Test Suite

This directory contains the test suite for the eryx package, with a focus on validating the OnePhonon model implementation.

## Directory Structure

```
tests/
├── pdbs/                       # Test PDB files
├── test_data/                  # Generated test data
│   ├── configs/                # Test configurations
│   ├── logs/                   # Generated test logs
│   └── reference/              # Reference data for validation
├── test_utils/                 # Test utilities
│   ├── config_generator.py     # Test config generation
│   ├── generate_reference_data.py  # Reference data generation
│   ├── generate_test_logs.py   # Test log generation
│   ├── log_analysis.py         # Log parsing utilities
│   ├── log_capture.py          # Log capture utilities
│   └── model_runner.py         # Model execution wrapper
└── test_*.py                   # Test files
```

## Test Setup

Before running tests, you need to generate test data and logs:

1. Generate reference data:
```bash
python -m tests.test_utils.generate_reference_data
```

2. Generate test logs:
```bash
python -m tests.test_utils.generate_test_logs
```

## Running Tests

Run the complete test suite:
```bash
pytest tests/ -v
```

Run specific test files:
```bash
pytest tests/test_onephonon.py -v          # Core model tests
pytest tests/test_diffraction_chain.py -v   # Diffraction computation tests
```

## Adding New Tests

1. Core model tests: Add to `test_onephonon.py`
   - Use `TestOnePhonon` class
   - Use fixtures for log files
   - Follow existing test patterns

2. Computation chain tests: Add to `test_diffraction_chain.py`
   - Use `onephonon` fixture
   - Compare against reference data
   - Use standard tolerances (rtol=1e-7)

3. Add edge cases:
   - Add config to `test_utils/config_generator.py`
   - Generate new logs
   - Add corresponding tests

## Test Data

Reference data is generated with controlled parameters:
- Small grid: `-2 to 2` with step `1`
- Standard model parameters:
  - `gnm_cutoff=4.0`
  - `gamma_intra=1.0`
  - `gamma_inter=1.0`
  - `expand_p1=True`

## Log Analysis

Logs follow a standard format:
```
[class_name.method_name] Enter with args={args}, kwargs={kwargs}
[class_name.method_name] Exit. Duration={duration:.2f}s
[Array Shape] array_name.shape={shape}
```

Use `LogAnalyzer` class from `test_utils/log_analysis.py` to parse logs.

## Common Issues

1. Missing test data:
   - Run reference data generation script
   - Check PDB file exists in tests/pdbs/

2. Test failures:
   - Check reference data matches current implementation
   - Verify log format hasn't changed
   - Check numerical tolerances

## Notes

- Tests use pytest fixtures extensively
- Numerical comparisons use numpy.testing
- Logs capture computation chain details
- Edge cases test error handling


## Resources
https://claude.ai/chat/4fecadb0-a584-4f19-8ab2-81387adc9980
