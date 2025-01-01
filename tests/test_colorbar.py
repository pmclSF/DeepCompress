import pytest
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from colorbar import get_colorbar, ColorbarGenerator, ColorbarConfig

class TestColorbar:
    """Test suite for colorbar generation and color mapping functionality."""
    
    def setup_method(self):
        self.vmin = 0
        self.vmax = 100
        
    def teardown_method(self):
        plt.close('all')

    def test_horizontal_colorbar(self):
        fig, cmap = get_colorbar(self.vmin, self.vmax, orientation='horizontal')
        assert fig is not None
        assert callable(cmap)
        
        test_values = [0, 50, 100]
        colors = [cmap(val) for val in test_values]
        assert len(colors) == len(test_values)
        assert all(len(color) == 4 for color in colors)

    def test_vertical_colorbar(self):
        fig, cmap = get_colorbar(self.vmin, self.vmax, orientation='vertical')
        assert fig is not None
        assert callable(cmap)
        assert tuple(fig.get_size_inches()) == (1.0, 6)

    def test_custom_labels(self):
        labels = ['Low', 'Medium', 'High']
        positions = [0, 50, 100]
        
        fig, cmap = get_colorbar(
            self.vmin,
            self.vmax,
            tick_labels=labels,
            tick_positions=positions
        )
        
        cbar_ax = next(obj for obj in fig.get_children() if isinstance(obj, plt.Axes))
        tick_labels = [t.get_text() for t in cbar_ax.get_xticklabels()]
        assert tick_labels == labels

    def test_title_and_formatting(self):
        title = "Test Colorbar"
        fig, cmap = get_colorbar(
            self.vmin,
            self.vmax,
            title=title,
            label_format='{:.2f}',
            tick_rotation=45
        )
        
        cbar_ax = next(obj for obj in fig.get_children() if isinstance(obj, plt.Axes))
        assert cbar_ax.get_xlabel() == title
        assert all(t.get_rotation() == 45 for t in cbar_ax.get_xticklabels())

    def test_invalid_orientation(self):
        with pytest.raises(AssertionError):
            get_colorbar(self.vmin, self.vmax, orientation='diagonal')

    def test_color_mapping(self):
        fig, cmap = get_colorbar(self.vmin, self.vmax)
        
        color = cmap(50)
        assert len(color) == 4
        assert all(0 <= c <= 1 for c in color)
        
        values = np.array([0, 50, 100])
        colors = cmap(values)
        assert colors.shape == (3, 4)