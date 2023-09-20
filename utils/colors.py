import matplotlib
matplotlib.use('agg');
from matplotlib import cm

import numpy as np

def get_random_colors(
    n_colors: int,
    dtype: str = 'float32',
    colormap: str = 'jet'
) -> np.ndarray :

    cmap = cm.get_cmap(colormap);
    cvals = np.linspace(0.0, 1.0, n_colors);
    np.random.shuffle(cvals);
    colors = np.array([cmap(i) for i in cvals]);

    if dtype.startswith('i') :
        colors = (colors * 255).astype(dtype);
    else :
        colors = colors.astype(dtype);
    
    return colors;  