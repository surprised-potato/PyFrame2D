# tests/test_basics.py
import pytest # type: ignore # We'll install pytest later, but good to import

def test_trivial_assertion():
    """
    A simple test to ensure the test setup works.
    """
    assert 1 + 1 == 2, "Basic math check failed!"

# You can add more simple tests here if you like
# def test_another_basic():
#    assert True