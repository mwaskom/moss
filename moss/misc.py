"""Miscellaneous utility functions."""
from IPython.core.display import Javascript, display


def save_notebook():
    """Save an IPython notebook from running code."""
    display(Javascript('IPython.notebook.save_notebook()'))
