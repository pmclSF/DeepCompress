import sys
import os

# Add the 'src' directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest
import numpy as np
from colorbar import get_colorbar

def test_horizontal_colorbar():
    fig, cmap = get_colorbar(0, 128, orientation='horizontal')
    assert fig is not None, "Figure should be created"
    assert callable(cmap), "Color mapping function should be callable"
    
    test_values = [0, 64, 128]
    colors = [cmap(val) for val in test_values]
    assert len(colors) == len(test_values), "Number of colors should match the number of test values"

def test_vertical_colorbar():
    fig, cmap = get_colorbar(0, 128, orientation='vertical')
    assert fig is not None, "Figure should be created"
    assert callable(cmap), "Color mapping function should be callable"
    
    test_values = [0, 64, 128]
    colors = [cmap(val) for val in test_values]
    assert len(colors) == len(test_values), "Number of colors should match the number of test values"

def test_invalid_orientation():
    with pytest.raises(AssertionError):
        get_colorbar(0, 128, orientation='diagonal')

def test_default_figsize():
    fig, _ = get_colorbar(0, 128, orientation='horizontal')
    assert tuple(fig.get_size_inches()) == (6, 0.5), "Default figsize for horizontal orientation is incorrect"

    fig, _ = get_colorbar(0, 128, orientation='vertical')
    assert tuple(fig.get_size_inches()) == (1.0, 6), "Default figsize for vertical orientation is incorrect"
