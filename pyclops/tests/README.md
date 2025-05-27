# PyClops Test Suite

This directory contains the test suite for PyClops. The tests are organized as follows:

```
tests/
├── README.md                # This file
├── conftest.py              # Shared test fixtures and configurations
├── peptides/                # Test peptide structures
│   ├── chignolin.pdb
│   └── pd1_binder.pdb
└── test_chemical_loss_handler.py  # Tests for ChemicalLossHandler
```

## Running Tests

### Basic Test Execution

To run all tests:

```bash
pytest
```

To run tests with verbose output:

```bash
pytest -v
```

To run tests and show print statements:

```bash
pytest -s
```

### Running Specific Tests

Run a specific test file:

```bash
pytest tests/test_chemical_loss_handler.py
```

Run a specific test function:

```bash
pytest tests/test_chemical_loss_handler.py::test_initialization
```

Run tests matching a pattern:

```bash
pytest -k "initialization or summary"
```

### Test Coverage

To run tests with coverage reporting:

```bash
pytest --cov=pyclops
```

To generate an HTML coverage report:

```bash
pytest --cov=pyclops --cov-report=html
```

### Test Organization

1. **Fixtures** (`conftest.py`):

   - Shared test fixtures
   - Common setup and teardown
   - Environment configuration

2. **Test Data** (`peptides/`):

   - PDB files for testing
   - Each file should be documented with its purpose

3. **Test Files**:
   - One test file per module
   - Tests organized by functionality
   - Clear test names and docstrings

## Best Practices

1. **Test Isolation**:

   - Each test should be independent
   - Use fixtures for setup and teardown
   - Avoid test interdependencies

2. **Test Naming**:

   - Use descriptive test names
   - Follow the pattern: `test_<functionality>_<scenario>`
   - Include docstrings explaining the test purpose

3. **Test Organization**:

   - Group related tests together
   - Use appropriate test markers
   - Keep tests focused and atomic

4. **Test Data**:
   - Keep test data in the `peptides/` directory
   - Document the purpose of each test file
   - Use realistic test cases

## Adding New Tests

1. Create a new test file following the naming convention: `test_<module_name>.py`
2. Add necessary fixtures to `conftest.py`
3. Add test data to appropriate directories
4. Write tests following the established patterns
5. Update this README if necessary

## Continuous Integration

The test suite is designed to be run in CI environments. Key considerations:

- Tests should be deterministic
- Use appropriate markers for slow tests
- Handle different environments (CPU/GPU)
- Provide clear error messages
