# Unit Tests

This directory contains comprehensive unit tests for the UMAP-DEA project.

## Test Structure

Tests are organized by module, mirroring the structure of the `src/` directory:

- **test_config.py** — Tests for configuration validation and parameter handling
- **test_dim_red.py** — Tests for dimensionality reduction (UMAP, PCA, embedding creation)
- **test_dgp.py** — Tests for data generation process
- **test_dea.py** — Tests for DEA calculations
- **test_eval.py** — Tests for evaluation metrics
- **conftest.py** — Shared fixtures used across tests

## Running Tests

### Run all tests:
```bash
pytest
```

### Run tests with coverage report:
```bash
pytest --cov=src --cov-report=html
```

### Run specific test file:
```bash
pytest tests/test_config.py
```

### Run specific test class:
```bash
pytest tests/test_config.py::TestSimulationConfigCreation
```

### Run specific test:
```bash
pytest tests/test_config.py::TestSimulationConfigCreation::test_valid_config_creation
```

### Run with verbose output:
```bash
pytest -v
```

### Run with detailed failure info:
```bash
pytest -vv --tb=long
```

## Test Coverage

The test suite includes:

- **Configuration tests** — Valid/invalid parameters, type checking, validation logic
- **Dimensionality reduction tests** — UMAP and PCA functionality, reproducibility, hyperparameter handling
- **Data generation tests** — Data structure, ranges, reproducibility with seeds
- **DEA calculation tests** — Efficiency score computation, valid ranges
- **Evaluation metric tests** — MAE, correlation coefficients, handling of NaNs

## Writing New Tests

When adding new functionality, follow these guidelines:

1. Create a test class for the function/module
2. Name tests descriptively starting with `test_`
3. Use fixtures from `conftest.py` for common data
4. Include docstrings explaining what is tested
5. Test both happy paths and error conditions

Example:
```python
class TestMyFunction:
    """Test my_function."""
    
    def test_basic_functionality(self, sample_data):
        """Test that my_function works correctly."""
        result = my_function(sample_data)
        assert result is not None
    
    def test_error_handling(self):
        """Test that my_function raises on invalid input."""
        with pytest.raises(ValueError):
            my_function(invalid_input)
```

## Continuous Integration

Tests are automatically run on commits. See `.github/workflows/tests.yml` for CI configuration.
