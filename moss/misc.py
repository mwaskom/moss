"""Miscellaneous utility functions."""
from IPython.core.display import Javascript, display


def save_notebook():
    """Save an IPython notebook from running code.

    Note this must be returned from a notebook cell.

    """
    display(Javascript('IPython.notebook.save_notebook()'))


def sig_stars(p):
    """Return a R-style significance string corresponding to p values."""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    elif p < 0.1:
        return "."
    return ""
