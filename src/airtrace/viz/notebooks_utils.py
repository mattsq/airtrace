"""Utilities for Jupyter notebooks."""

import matplotlib.pyplot as plt
import seaborn as sns


def setup_notebook():
    """Set up notebook environment with good defaults."""
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 10

    # Enable inline plotting
    try:
        from IPython import get_ipython
        ipython = get_ipython()
        if ipython is not None:
            ipython.magic('matplotlib inline')
    except ImportError:
        pass

    print("Notebook environment set up successfully!")
