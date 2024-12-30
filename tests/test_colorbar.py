import sys
import os
import pytest
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

# Add src directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from colorbar import get_colorbar, ColorbarGenerator, ColorbarConfig

class TestColorbar:
    def setup_method(self):
        """Setup for each test method."""
        self.vmin = 0
        self.vmax = 100
        
    def teardown_method(self):
        """Cleanup after each test."""
        plt.close('all')

    def test_horizontal_colorbar(self):
        """Test horizontal colorbar generation."""
        fig, cmap = get_colorbar(self.vmin, self.vmax, orientation='horizontal')
        assert fig is not None, "Figure should be created"
        assert callable(cmap), "Color mapping function should be callable"
        
        test_values = [0, 50, 100]
        colors = [cmap(val) for val in test_values]
        assert len(colors) == len(test_values)
        assert all(len(color) == 4 for color in colors)  # RGBA values

    def test_vertical_colorbar(self):
        """Test vertical colorbar generation."""
        fig, cmap = get_colorbar(self.vmin, self.vmax, orientation='vertical')
        assert fig is not None, "Figure should be created"
        assert callable(cmap), "Color mapping function should be callable"
        
        # Test figure dimensions
        assert tuple(fig.get_size_inches()) == (1.0, 6)

    def test_custom_labels(self):
        """Test colorbar with custom tick labels."""
        labels = ['Low', 'Medium', 'High']
        positions = [0, 50, 100]
        
        fig, cmap = get_colorbar(
            self.vmin,
            self.vmax,
            tick_labels=labels,
            tick_positions=positions
        )
        
        # Get colorbar axes
        cbar_ax = next(obj for obj in fig.get_children() if isinstance(obj, plt.Axes))
        tick_labels = [t.get_text() for t in cbar_ax.get_xticklabels()]
        
        assert tick_labels == labels, "Custom tick labels not set correctly"

    def test_title_and_formatting(self):
        """Test colorbar title and label formatting."""
        title = "Test Colorbar"
        fig, cmap = get_colorbar(
            self.vmin,
            self.vmax,
            title=title,
            label_format='{:.2f}',
            tick_rotation=45
        )
        
        # Get colorbar axes
        cbar_ax = next(obj for obj in fig.get_children() if isinstance(obj, plt.Axes))
        
        assert cbar_ax.get_xlabel() == title, "Title not set correctly"
        assert all(t.get_rotation() == 45 for t in cbar_ax.get_xticklabels())

    def test_invalid_orientation(self):
        """Test that invalid orientation raises error."""
        with pytest.raises(AssertionError):
            get_colorbar(self.vmin, self.vmax, orientation='diagonal')

    def test_color_mapping(self):
        """Test color mapping function output."""
        fig, cmap = get_colorbar(self.vmin, self.vmax)
        
        # Test single value
        color = cmap(50)
        assert len(color) == 4, "Color should be RGBA"
        assert all(0 <= c <= 1 for c in color), "Color values should be between 0 and 1"
        
        # Test array of values
        values = np.array([0, 50, 100])
        colors = cmap(values)
        assert colors.shape == (3, 4), "Should return array of RGBA colors"

    def test_save_mapping(self, tmp_path):
        """Test saving color mapping to file."""
        output_file = tmp_path / "colormap.json"
        
        generator = ColorbarGenerator(
            self.vmin,
            self.vmax,
            cmap='viridis',
            config=ColorbarConfig(title="Test Map")
        )
        
        # Save mapping
        generator.save_mapping(str(output_file))
        
        # Verify saved file
        assert output_file.exists()
        
        with open(output_file) as f:
            data = json.load(f)
        
        # Check content
        assert 'values' in data
        assert 'colors' in data
        assert 'vmin' in data
        assert 'vmax' in data
        assert 'cmap' in data
        assert data['vmin'] == self.vmin
        assert data['vmax'] == self.vmax
        
        # Check mapping values
        assert len(data['values']) == 256  # Default number of samples
        assert all(isinstance(v, float) for v in data['values'])

    def test_extended_colorbar(self):
        """Test colorbar with extended ends."""
        fig, cmap = get_colorbar(
            self.vmin,
            self.vmax,
            extend='both'
        )
        
        # Get colorbar
        cbar_ax = next(obj for obj in fig.get_children() if isinstance(obj, plt.Axes))
        cbar = cbar_ax.collections[0].colorbar
        
        assert cbar._extend == 'both'

if __name__ == '__main__':
    pytest.main([__file__])