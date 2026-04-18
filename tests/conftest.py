"""
conftest.py
-----------
Pytest fixtures configuration file. 
Sets up Ray cluster for testing and provides dummy datasets.
"""

import pytest
import ray
from sklearn.datasets import make_classification, make_regression

@pytest.fixture(scope="session", autouse=True)
def init_ray_for_tests():
    """
    Initialize a lightweight local Ray cluster for the entire test session.
    'autouse=True' ensures this runs automatically before any tests.
    'scope="session"' ensures it only starts once for all tests to save time.
    """
    # Start Ray with minimal resources to keep tests fast and prevent memory hogs
    ray.init(
        num_cpus=2, 
        ignore_reinit_error=True, 
        include_dashboard=False,
        logging_level="ERROR" 
    )
    
    yield 
    
    if ray.is_initialized():
        ray.shutdown()

@pytest.fixture(scope="session")
def classification_data():
    """
    Generate dummy dataset for classification tests.
    Returns: (X, y)
    """
    X, y = make_classification(
        n_samples=100, 
        n_features=5, 
        n_informative=3, 
        random_state=42
    )
    return X, y

@pytest.fixture(scope="session")
def regression_data():
    """
    Generate dummy dataset for regression tests.
    Returns: (X, y)
    """
    X, y = make_regression(
        n_samples=100, 
        n_features=5, 
        n_informative=3, 
        random_state=42
    )
    return X, y