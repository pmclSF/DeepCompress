import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def get_colorbar(a_min, a_max, cmap='inferno', orientation='horizontal', figsize=None):
    """
    Generate a matplotlib colorbar as a standalone figure.

    Parameters:
    - a_min: Minimum value of the colorbar.
    - a_max: Maximum value of the colorbar.
    - cmap: Colormap to use (can be a string or a colormap instance).
    - orientation: Orientation of the colorbar ('horizontal' or 'vertical').
    - figsize: Figure size as a tuple (width, height). Defaults based on orientation.

    Returns:
    - fig: Matplotlib figure containing the colorbar.
    - value_to_color: Function mapping values to corresponding colors.
    """
    assert orientation in ('horizontal', 'vertical'), "Orientation must be 'horizontal' or 'vertical'."

    a = np.array([[a_min, a_max]])

    if figsize is None:
        figsize = (6, 0.5) if orientation == 'horizontal' else (1.0, 6)

    fig = plt.figure(figsize=figsize)

    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    img = plt.imshow(a, cmap=cmap, aspect='auto')
    plt.gca().set_visible(False)

    if orientation == 'horizontal':
        cax = plt.axes([0.05, 0.5, 0.9, 0.2])
    else:
        cax = plt.axes([0.2, 0.05, 0.15, 0.9])

    plt.colorbar(img, orientation=orientation, cax=cax)

    norm = mpl.colors.Normalize(vmin=a_min, vmax=a_max)

    return fig, lambda x: cmap(norm(x))

if __name__ == '__main__':
    fig, cmap = get_colorbar(0, 128, orientation='horizontal')
    plt.show()

    fig, cmap = get_colorbar(0, 128, orientation='vertical')
    plt.show()

    x = [0, 64, 128, 192, 256]
    print(x, [cmap(val) for val in x])