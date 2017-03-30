"""Psychophysical models with a common interface.

TODO get results from MATLAB functions and write test script to compare.

"""
from __future__ import print_function, division
import numpy as np
import pandas as pd

from scipy import stats
from scipy.optimize import minimize
#from statsmodels.tools import numdiff

import seaborn as sns
import matplotlib.pyplot as plt

from .params import ParamSet
from .visualization import log0_safe_xticks, plot_limits


class PsychophysicsModel(object):
    """Basic super-class for objects that fit psychophysical data."""
    default_params = pd.Series()
    default_fixed = []

    def err_func(self, params):
        """Objective to minimize when fitting the model."""
        raise NotImplementedError

    def fit(self, initial_params=None, fix=None, method="nelder-mead"):
        """External fit interface, allows specification of variables."""
        if initial_params is None:
            params = self.default_params
        else:
            params = self.default_params.copy()
            if isinstance(initial_params, dict):
                initial_params = pd.Series(initial_params)
            params.update(initial_params)
            # TODO fill in missing initial params from default
        if fix is None:
            fix = self.default_fixed
        self.params = ParamSet(params, fix)

        # TODO implement inferring defaults from the data here
        # That can call out to methods on the subclasses
        initial_free = self.params.free

        # Minimize the objective function
        result = minimize(self.err_func, initial_free, method=method)

        # Assign fitted attributes
        self.result_ = result
        self.ll_ = -result["fun"]
        self.params_ = self.params
        self.success_ = result["success"]

        # Numerically estimate the hessian for standard error on params
        # TODO this doesn't match the param vector because of fixed params
        # hessian = numdiff.approx_hess(self.params.free, self.err_func)
        # stderr = np.diag(np.sqrt(np.linalg.inv(hessian)))
        # self.hessian_ = hessian
        # self.stderrs_ = stderr

        return self

    def bootstrap(self, n=100):
        """Resamplea and re-fit to estimate parameter confidence intervals."""
        raise NotImplementedError

    def _plot(self, y_var, pred_func, plot_zero, color, label, logx,
              data_kws, line_kws, ax):

        if ax is None:
            ax = plt.gca()

        if data_kws is None:
            data_kws = {}
        else:
            data_kws = data_kws.copy()

        if line_kws is None:
            line_kws = {}
        else:
            line_kws = line_kws.copy()

        if color is None:
            line, = ax.plot([])
            color = line.get_color()
            line.remove()

        data_kws.setdefault("color", color)
        line_kws.setdefault("color", color)
        line_kws.setdefault("label", label)

        plot_data = self.fit_data.copy()

        if logx:
            x_ticks, x_labels = log0_safe_xticks(plot_data["x"])
            x_lim = plot_limits(plot_data["x"])
            xx = np.logspace(*np.log(x_lim), base=np.e, num=100)
        else:
            x_lim = plot_limits(plot_data["x"], logsteps=False)
            xx = np.linspace(*x_lim, num=100)

        if not plot_zero:
            plot_data = plot_data.query("x > 0")
        else:
            plot_data["x"] = plot_data["x"].replace({0: x_ticks[0]})

        sns.regplot("x", y_var,
                    data=plot_data,
                    x_estimator=np.mean,
                    fit_reg=False,
                    ax=ax,
                    **data_kws)

        yy = pred_func(xx)
        ax.plot(xx, yy, **line_kws)

        if logx:
            ax.set_xscale("log")
            ax.set_xticks([], minor=True)
            ax.set(xlim=x_lim,
                   xticks=x_ticks,
                   xticklabels=x_labels,
                   xlabel=self._x_name)
        else:
            ax.set(xlim=x_lim,
                   xlabel=self._x_name)

        return ax


class PsychometricModel(PsychophysicsModel):

    fit_which = "pc"

    def __init__(self, data, x, pc):
        """Initialize the model with relevant data parameters."""
        self._full_data = data
        self.fit_data = data[[x, pc]].copy()
        self.fit_data.columns = ["x", "pc"]
        self._x_name = x

    @staticmethod
    def pc_func(x, *params):
        """Bernoulli likelihood given model parameters."""
        raise NotImplementedError

    def predict_pc(self, x):
        """Use fitted parameters to predict proportion correct."""
        raise NotImplementedError

    def plot(self, color=None, label=None, logx=False,
             data_kws=None, line_kws=None, ax=None):

        ax = self._plot("pc",
                        pred_func=self.predict_pc,
                        plot_zero=False,
                        color=color,
                        label=label,
                        logx=logx,
                        data_kws=data_kws,
                        line_kws=line_kws,
                        ax=ax)

        ax.set(ylabel="P(Correct)", ylim=(None, 1.025))
        return ax

    @classmethod
    def fit_and_plot(cls, x, y, initial_params=None, fix=None, **kws):

        df = pd.DataFrame(dict(x=x, pc=y))
        m = cls(df, "x", "pc").fit(initial_params, fix)
        m.plot(**kws)


class ChronometricModel(PsychophysicsModel):

    fit_which = "rt"

    def __init__(self, data, x, rt):
        """Initialize the model with relevant data parameters."""
        self._full_data = data
        self.fit_data = data[[x, rt]].copy()
        self.fit_data.columns = ["x", "rt"]
        self._x_name = x

    @staticmethod
    def rt_func(x, *params):
        """Predicted RT given stimulus strength and model parameters."""
        raise NotImplementedError

    def predict_rt(self, x):
        """Use fitted parameters to predict RT."""
        raise NotImplementedError

    def plot(self, color=None, label=None, logx=False,
             data_kws=None, line_kws=None, ax=None):

        ax = self._plot("rt",
                        pred_func=self.predict_rt,
                        plot_zero=True,
                        color=color,
                        label=label,
                        logx=logx,
                        data_kws=data_kws,
                        line_kws=line_kws,
                        ax=ax)

        ax.set(ylabel="RT")
        return ax

    @classmethod
    def fit_and_plot(cls, x, y, initial_params=None, fix=None, **kws):

        df = pd.DataFrame(dict(x=x, pc=y))
        m = cls(df, "x", "pc").fit(initial_params, fix)
        m.plot(**kws)


class DiffusionModel(PsychophysicsModel):

    def __init__(self, data, x, pc="correct", rt="rt"):
        """Initialize the model with relevant data parameters."""
        self._full_data = data
        if pc is None and rt is not None:
            fit_which = "rt"
            cols = [x, rt]
            col_names = ["x", "rt"]
        elif rt is None and pc is not None:
            fit_which = "pc"
            cols = [x, pc]
            col_names = ["x", "pc"]
        else:
            fit_which = "pcrt"
            cols = [x, pc, rt]
            col_names = ["x", "pc", "rt"]
        self.fit_data = data[cols].copy()
        self.fit_data.columns = col_names
        self.fit_which = fit_which
        self._x_name = x

    pc_func = PsychometricModel.pc_func
    predict_pc = PsychometricModel.predict_pc

    rt_func = ChronometricModel.rt_func
    predict_rt = ChronometricModel.predict_rt


class Logistic(PsychometricModel):

    default_params = pd.Series([1, 0], ["alpha", "lapse"])
    default_fixed = ["lapse"]

    @staticmethod
    def pc_func(x, alpha, lapse):
        y = lapse + (1 - 2 * lapse) * (1 / (1 + np.exp(-alpha * x)))
        return y

    def err_func(self, param_vector):

        p = self.params.update(param_vector)

        x = np.array(self.fit_data["x"])
        pc = np.array(self.fit_data["pc"], np.bool)
        pred_pc = self.pc_func(x, p.alpha, p.lapse)
        pred_pc[~pc & (pred_pc == 1)] = 1e-8
        ll = np.sum(np.log(pred_pc[pc])) + np.sum(np.log(1 - pred_pc[~pc]))

        return -ll

    def predict_pc(self, x):
        # TODO could this be abstracted?
        x = np.asarray(x)
        p = self.params
        return self.pc_func(x, p.alpha, p.lapse)


class Weibull(PsychometricModel):

    # TODO implement infering default alpha value
    default_params = pd.Series([1, 1, 0], ["alpha", "beta", "lapse"])
    default_fixed = ["lapse"]

    @staticmethod
    def pc_func(x, alpha, beta, lapse):
        y = 0.5 + (0.5 - lapse) * (1 - np.exp(-(x / alpha) ** beta))
        return y

    def err_func(self, param_vector):

        p = self.params.update(param_vector)

        x = np.array(self.fit_data["x"])
        pc = np.array(self.fit_data["pc"], np.bool)

        # TODO implement > 2AFC (makes sense?)
        pred_pc = self.pc_func(x, p.alpha, p.beta, p.lapse)

        # Avoid log(0)
        # TODO candidate for abstraction
        pred_pc[~pc & (pred_pc == 1)] = 1e-8

        # TODO candidate for abstraction
        ll = (np.sum(np.log(pred_pc[pc]))
              + np.sum(np.log(1 - pred_pc[~pc])))

        return -ll

    def predict_pc(self, x):
        # TODO could this be abstracted?
        x = np.asarray(x)
        p = self.params
        return self.pc_func(x, p.alpha, p.beta, p.lapse)


class HyperbolicTangent(ChronometricModel):

    default_params = pd.Series([.1, 20, 500], ["k", "B", "T0"])
    default_fixed = []

    @staticmethod
    def rt_func(x, k, B, T0):
        s = x != 0
        y = np.empty_like(x)
        y[s] = (B / (k * x[s])) * np.tanh(B * k * x[s]) + T0
        y[~s] = B ** 2 + T0
        return y

    def err_func(self, param_vector):

        p = self.params.update(param_vector)

        rt_bins = self.fit_data.groupby("x").rt.agg([np.mean, stats.sem])
        x_vals = rt_bins.index.values
        pred_rt = self.rt_func(x_vals, p.k, p.B, p.T0)
        rt_like = _lognormpdf(pred_rt,
                              rt_bins["mean"].values,
                              rt_bins["sem"].values)
        ll = rt_like.sum()

        return -ll

    def predict_rt(self, x):
        x = np.asarray(x)
        p = self.params
        return self.rt_func(x, p.k, p.B, p.T0)


class IndependentDiffusion(DiffusionModel):

    default_params = pd.Series([.1, 20, 500, 0], ["k", "B", "T0", "lapse"])
    default_fixed = ["lapse"]

    @staticmethod
    def pc_func(x, k, B, lapse):
        return lapse + (1 - 2 * lapse) * (1 / (1 + np.exp(-2 * B * k * x)))

    @staticmethod
    def rt_func(x, k, B, T0):
        s = x != 0
        y = np.empty_like(x)
        y[s] = (B / (k * x[s])) * np.tanh(B * k * x[s]) + T0
        y[~s] = B ** 2 + T0
        return y

    def err_func(self, param_vector):

        p = self.params.update(param_vector)
        x = np.array(self.fit_data["x"])
        ll = 0

        if "pc" in self.fit_which:
            pc = np.array(self.fit_data["pc"], np.bool)
            pred_pc = self.pc_func(x, p.k, p.B, p.lapse)
            pred_pc[~pc & (pred_pc == 1)] = 1e-8
            pc_like = np.r_[pred_pc[pc], 1 - pred_pc[~pc]]
            ll_pc = np.log(pc_like).sum()
            ll += ll_pc

        if "rt" in self.fit_which:
            rt_bins = self.fit_data.groupby("x").rt.agg([np.mean, stats.sem])
            x_vals = rt_bins.index.values
            pred_rt = self.rt_func(x_vals, p.k, p.B, p.T0)
            rt_like = _lognormpdf(pred_rt,
                                  rt_bins["mean"].values,
                                  rt_bins["sem"].values)
            ll_rt = rt_like.sum()
            ll += ll_rt

        return -ll

    def predict_pc(self, x):
        x = np.asarray(x)
        p = self.params
        return self.pc_func(x, p.k, p.B, p.lapse)

    def predict_rt(self, x):
        x = np.asarray(x)
        p = self.params
        return self.rt_func(x, p.k, p.B, p.T0)


class ProportionalRate(IndependentDiffusion):

    @staticmethod
    def rt_var_func(x, k, B, T0):
        s = x != 0
        y = np.empty_like(x)
        mu = x * k
        y[s] = ((B * (np.tanh(B * mu[s]) - B * mu[s]
                 / np.cosh(B * mu[s]) ** 2))
                / mu[s] ** 3) + .1 * T0
        y[~s] = (2 / 3) * B ** 4 + .1 * T0
        return y

    def err_func(self, param_vector):

        p = self.params.update(param_vector)
        x = np.array(self.fit_data["x"])
        ll = 0

        if "pc" in self.fit_which:
            pc = np.array(self.fit_data["pc"], np.bool)
            pred_pc = self.pc_func(x, p.k, p.B, p.lapse)
            pred_pc[~pc & (pred_pc == 1)] = 1e-8
            pc_like = np.r_[pred_pc[pc], 1 - pred_pc[~pc]]
            ll_pc = np.log(pc_like).sum()
            ll += ll_pc

        if "rt" in self.fit_which:
            rt_bins = self.fit_data.groupby("x").rt.agg([np.mean, np.size])
            x_vals = rt_bins.index.values
            pred_rt = self.rt_func(x_vals, p.k, p.B, p.T0)
            pred_rt_var = self.rt_var_func(x_vals, p.k, p.B, p.T0)
            pred_rt_sem = np.sqrt(pred_rt_var / rt_bins["size"])
            rt_like_bins = _lognormpdf(rt_bins["mean"], pred_rt, pred_rt_sem)
            rt_like_bins = pd.Series(rt_like_bins, index=rt_bins.index)
            rt_like = self.fit_data.x.map(rt_like_bins)
            ll_rt = rt_like.sum()
            ll += ll_rt

        return -ll


def _lognormpdf(x, mu, sigma):
    """Log likelihood of normal distribution."""
    d = np.sqrt(2 * np.pi)
    return np.log(1 / (d * sigma)) - (x - mu) ** 2 / (2 * sigma ** 2)
