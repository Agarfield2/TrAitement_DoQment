"""
Pytest configuration : sys.path setup + slow marker.

The project layout has the canonical Phase 1 modules at the repo root
(ingestion, pipeline1, ...) and our wrappers under doqment/. Both must
be importable from any test file.
"""

import sys
from pathlib import Path


_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def pytest_configure(config):
    """
    Registers custom markers so unknown-marker warnings don't pollute output.

    Args:
        config: The pytest config object.
    """

    config.addinivalue_line(
        "markers",
        "slow: tests that load real models (skip by default).",
    )


def pytest_collection_modifyitems(config, items):
    """
    Skips @pytest.mark.slow tests unless `pytest --run-slow` is passed.

    Args:
        config: The pytest config object.
        items: Collected test items, modified in place.
    """

    if config.getoption("--run-slow"):
        return
    import pytest
    skip_slow = pytest.mark.skip(reason="needs --run-slow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


def pytest_addoption(parser):
    """
    Adds the --run-slow option to pytest CLI.

    Args:
        parser: The pytest argument parser.
    """

    parser.addoption(
        "--run-slow", action="store_true", default=False,
        help="run tests marked @pytest.mark.slow",
    )
