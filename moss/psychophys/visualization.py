from __future__ import print_function, division
import numpy as np


def log0_safe_xticks(x, logsteps=True):
    """Get values for stimulus axis while handling of log(0)."""
    orig_values = np.sort(np.unique(x))

    if any(orig_values < 0):
        raise ValueError("`x` values must be non-negative")

    if logsteps:
        log_values = np.log(orig_values)
        dx = np.diff(log_values)
        step_size = np.median(dx[1:])
        log_values[0] = log_values[1] - step_size 
        values = np.exp(log_values)
    else:
        values = orig_values

    return values, orig_values


def plot_limits(x, logsteps=True, pad=.3):
    """Get limits for a psychophysics plot while handling log(0)."""
    values, _ = log0_safe_xticks(x, logsteps)

    if logsteps:
        log_values = np.log(values)
        dx = np.diff(log_values)
    else:
        dx = np.diff(values)
    step_size = np.median(dx)

    pad_size = step_size * pad
    if logsteps:
        lims = log_values[0] - pad_size, log_values[-1] + pad_size
        lims = tuple(np.exp(lims))
    else:
        lims = values[0] - pad_size, values[-1] + pad_size

    return lims
