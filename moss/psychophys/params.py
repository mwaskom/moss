"""Object for managing computational model parameters."""
from __future__ import print_function, division
import numpy as np
import pandas as pd


class ParamSet(object):
    """Object for managing model parameters during model fitting.

    The main contribution of this class is separation of free and fixed
    parameters in a way that works with scipy optimize functionality.
    Parameters can be accessed through the `.params` attribute, in the separate
    `.free` and `.fixed` attributes, or directly by names. Free parameters can
    be updated with an array that lacks semantic information (what scipy uses
    internally) and are mapped properly to the named parameters.

    """
    def __init__(self, initial, fix=None):
        """Set initial values and determine fixed parameters.

        Parameters
        ----------
        initial : Series or dictionary
            Initial values for parameters.
        fix : list of strings, optional
            Names of parameters to fix at initial values.

        """
        if isinstance(initial, dict):
            initial = pd.Series(initial)

        self.names = list(initial.index)
        self.params = initial.copy()

        if fix is None:
            fix = []

        if set(fix) - set(self.names):
            raise ValueError("Fixed parameters must appear in `initial`")

        self.fixed_names = [n for n in self.names if n in fix]
        self.free_names = [n for n in self.names if n not in fix]

    def __repr__(self):
        """Show the values and fixed status of each parameter."""
        repr = ""
        repr += "Free Parameters:\n"
        for name, val in self.free.iteritems():
            repr += "  {}: {:.3g}\n".format(name, val)
        if self.fixed_names:
            repr += "Fixed Parameters:\n"
            for name, val in self.fixed.iteritems():
                repr += "  {}: {:.3g}\n".format(name, val)
        return repr

    def __getattr__(self, name):
        """Allow dot access to params."""
        if name in self.params:
            return self.params[name]
        else:
            return object.__getattribute__(self, name)

    def __setattr__(self, name, val):
        """Allow dot access to params."""
        # TODO commenting out as it causes an infinite recursion error on Py3
        #if hasattr(self, "params") and name in self.params:
        #    self.params[name] = val
        #else:
        #    object.__setattr__(self, name, val)
        object.__setattr__(self, name, val)

    @property
    def free(self):
        """Return a vector of current free parameter values."""
        return self.params[self.free_names]

    @property
    def fixed(self):
        """Return a vector of current fixed parameter values."""
        return self.params[self.fixed_names]

    def update(self, params):
        """Set new values for the free parameters and return self.

        Parameters
        ----------
        params : ParamSet, Series, dictionary, or vector
            Either an equivalent ParamSet, Series or dictionary mapping
            parameter names to values, or a vector of parameters in the order
            of `self.free`.

        Returns
        -------
        self : ParamSet
            Returns self with new parameter values.

        """
        if isinstance(params, ParamSet):
            if params.free_names != self.free_names:
                err = "Input object must have same free parameters."
                raise ValueError(err)
            self.params.update(params.params)

        elif isinstance(params, pd.Series):
            if list(params.index) != self.free_names:
                err = "Input object must have same free parameters."
                raise ValueError(err)
            self.params.update(params)

        elif isinstance(params, dict):
            if set(params) - set(self.free_names):
                err = "Input object has unknown parameters"
                raise ValueError(err)
            elif set(self.free_names) - set(params):
                err = "Input object is missing parameters"
                raise ValueError(err)
            self.params.update(pd.Series(params))

        elif isinstance(params, (np.ndarray, list, tuple)):
            if len(params) != len(self.free_names):
                err = "Input object has wrong number of parameters."
                raise ValueError(err)
            new_params = pd.Series(params, self.free_names)
            self.params.update(new_params)

        else:
            err = "Type of `values` is not understood"
            raise ValueError(err)

        return self


