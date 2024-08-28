"""
Unit and regression test for the niupy package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import niupy


def test_niupy_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "niupy" in sys.modules
