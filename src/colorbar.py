import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import Tuple, Callable, Optional, List
from dataclasses import dataclass
import json
from pathlib import Path

@dataclass
class ColorbarConfig:
    """Configuration for colorbar appearance."""
    title: Optional[str] = None
    tick_labels: Optional[List[str]] = None
    tick_positions: Optional[List[float]] = None
    label_format: str = '{:.1f}'
    font_size: int = 10
    title_size: int = 12
    tick_rotation: int = 0
    extend: str = 'neither'

def get_colorbar(
    vmin: float,
    vmax: float,
    cmap: str = 'inferno',
    orientation: str = 'horizontal',
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    tick_labels: Optional[List[str]] = None,
    tick_positions: Optional[List[float]] = None,
    label_format: str = '{:.1f}',
    font_size: int = 10,
    title_size: int = 12,
    tick_rotation: int = 0,
    extend: str = 'neither'
) -> Tuple[plt.Figure, Callable]:
    """
    Generate a matplotlib colorbar as a standalone figure.
    """
    assert orientation in ('horizontal', 'vertical'), "Invalid orientation"

    # Set up figure
    if figsize is None:
        figsize = (6, 0.5) if orientation == 'horizontal' else (1.0, 6)
    fig = plt.figure(figsize=figsize)

    # Create dummy data for colorbar
    a = np.array([[vmin, vmax]])
    cmap = plt.get_cmap(cmap)
    img = plt.imshow(a, cmap=cmap, aspect='auto')
    plt.gca().set_visible(False)

    # Position the colorbar
    if orientation == 'horizontal':
        cax = plt.axes([0.1, 0.5, 0.8, 0.2])
    else:
        cax = plt.axes([0.3, 0.1, 0.2, 0.8])

    # Create colorbar
    cbar = plt.colorbar(img, cax=cax, orientation=orientation, extend=extend)

    # Set up ticks and labels
    if tick_labels is not None:
        if tick_positions is None:
            tick_positions = np.linspace(vmin, vmax, len(tick_labels))
        cbar.set_ticks(tick_positions)
        cbar.ax.set_xticklabels(tick_labels) if orientation == 'horizontal' else cbar.ax.set_yticklabels(tick_labels)
    else:
        # Format numeric labels
        formatter = mpl.ticker.FormatStrFormatter(label_format)
        cbar.ax.xaxis.set_major_formatter(formatter) if orientation == 'horizontal' else cbar.ax.yaxis.set_major_formatter(formatter)

    # Set font sizes
    cbar.ax.tick_params(labelsize=font_size)
    plt.setp(cbar.ax.get_xticklabels(), rotation=tick_rotation)

    # Set title
    if title:
        if orientation == 'horizontal':
            cbar.ax.set_xlabel(title, size=title_size)
        else:
            cbar.ax.set_ylabel(title, size=title_size)

    # Create color mapping function
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    def value_to_color(x: float) -> np.ndarray:
        return cmap(norm(x))

    return fig, value_to_color

def save_color_mapping(filename: str,
                      vmin: float,
                      vmax: float,
                      cmap: str,
                      num_samples: int = 256):
    """Save color mapping data to file."""
    # Create mapping
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(cmap)
    
    values = np.linspace(vmin, vmax, num_samples)
    colors = [cmap(norm(v)) for v in values]
    
    # Save to file
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    with open(filename, 'w') as f:
        json.dump({
            'values': values.tolist(),
            'colors': [list(c) for c in colors],
            'vmin': vmin,
            'vmax': vmax,
            'cmap': cmap.name
        }, f, indent=2)

if __name__ == '__main__':
    # Example usage
    fig, cmap = get_colorbar(
        0, 100,
        orientation='horizontal',
        title='Example Colorbar',
        tick_labels=['Low', 'Medium', 'High'],
        tick_positions=[0, 50, 100],
        tick_rotation=45,
        extend='both'
    )
    plt.show()