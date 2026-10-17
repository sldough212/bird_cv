import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "slow: runs real models; deselect with -m 'not slow'"
    )
