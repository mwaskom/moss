"""Assorted functions for statistical calculations."""
from __future__ import division
import numpy as np
from scipy import stats
from scipy.interpolate import interp1d
import pandas as pd
import statsmodels.api as sm
from six.moves import range


def bootstrap(*args, **kwargs):
    """Resample one or more arrays with replacement and store aggregate values.

    Positional arguments are a sequence of arrays to bootstrap along the first
    axis and pass to a summary function.

    Keyword arguments:
        n_boot : int, default 10000
            Number of iterations
        axis : int, default None
            Will pass axis to ``func`` as a keyword argument.
        units : array, default None
            Array of sampling unit IDs. When used the bootstrap resamples units
            and then observations within units instead of individual
            datapoints.
        smooth : bool, default False
            If True, performs a smoothed bootstrap (draws samples from a kernel
            destiny estimate); only works for one-dimensional inputs and cannot
            be used `units` is present.
        func : callable, default np.mean
            Function to call on the args that are passed in.
        random_seed : int | None, default None
            Seed for the random number generator; useful if you want
            reproducible resamples.

    Returns
    -------
    boot_dist: array
        array of bootstrapped statistic values

    """
    # Ensure list of arrays are same length
    if len(np.unique(list(map(len, args)))) > 1:
        raise ValueError("All input arrays must have the same length")
    n = len(args[0])

    # Default keyword arguments
    n_boot = kwargs.get("n_boot", 10000)
    func = kwargs.get("func", np.mean)
    axis = kwargs.get("axis", None)
    units = kwargs.get("units", None)
    smooth = kwargs.get("smooth", False)
    random_seed = kwargs.get("random_seed", None)
    if axis is None:
        func_kwargs = dict()
    else:
        func_kwargs = dict(axis=axis)

    # Initialize the resampler
    rs = np.random.RandomState(random_seed)

    # Coerce to arrays
    args = list(map(np.asarray, args))
    if units is not None:
        units = np.asarray(units)

    # Do the bootstrap
    if smooth:
        return _smooth_bootstrap(args, n_boot, func, func_kwargs)

    if units is not None:
        return _structured_bootstrap(args, n_boot, units,
                                     func, func_kwargs, rs)

    boot_dist = []
    for i in range(int(n_boot)):
        resampler = rs.randint(0, n, n)
        sample = [a.take(resampler, axis=0) for a in args]
        boot_dist.append(func(*sample, **func_kwargs))
    return np.array(boot_dist)


def _structured_bootstrap(args, n_boot, units, func, func_kwargs, rs):
    """Resample units instead of datapoints."""
    unique_units = np.unique(units)
    n_units = len(unique_units)

    args = [[a[units == unit] for unit in unique_units] for a in args]

    boot_dist = []
    for i in range(int(n_boot)):
        resampler = rs.randint(0, n_units, n_units)
        sample = [np.take(a, resampler, axis=0) for a in args]
        lengths = map(len, sample[0])
        resampler = [rs.randint(0, n, n) for n in lengths]
        sample = [[c.take(r, axis=0) for c, r in zip(a, resampler)]
                  for a in sample]
        sample = list(map(np.concatenate, sample))
        boot_dist.append(func(*sample, **func_kwargs))
    return np.array(boot_dist)


def _smooth_bootstrap(args, n_boot, func, func_kwargs):
    """Bootstrap by resampling from a kernel density estimate."""
    n = len(args[0])
    boot_dist = []
    kde = [stats.gaussian_kde(np.transpose(a)) for a in args]
    for i in range(int(n_boot)):
        sample = [a.resample(n).T for a in kde]
        boot_dist.append(func(*sample, **func_kwargs))
    return np.array(boot_dist)


def percentiles(a, pcts, axis=None):
    """Like scoreatpercentile but can take and return array of percentiles.

    Parameters
    ----------
    a : array
        data
    pcts : sequence of percentile values
        percentile or percentiles to find score at
    axis : int or None
        if not None, computes scores over this axis

    Returns
    -------
    scores: array
        array of scores at requested percentiles
        first dimension is length of object passed to ``pcts``

    """
    scores = []
    try:
        n = len(pcts)
    except TypeError:
        pcts = [pcts]
        n = 0
    for i, p in enumerate(pcts):
        if axis is None:
            score = stats.scoreatpercentile(a.ravel(), p)
        else:
            score = np.apply_along_axis(stats.scoreatpercentile, axis, a, p)
        scores.append(score)
    scores = np.asarray(scores)
    if not n:
        scores = scores.squeeze()
    return scores


def vector_reject(x, y):
    """Remove y from x using vector rejection."""
    x = np.asarray(x).astype(float).reshape((-1, 1))
    y = np.asarray(y).astype(float).reshape((-1, 1))
    x_ = x - np.dot(x.T, y).T * (y / np.dot(y.T, y))
    return x_.ravel()


def ci(a, which=95, axis=None):
    """Return a percentile range from an array of values."""
    p = 50 - which / 2, 50 + which / 2
    return percentiles(a, p, axis)


def add_constant(a):
    """Add a constant term to a design matrix.

    Parameters
    ----------
    a : array
        original design matrix

    Returns
    -------
    a : array
        design matrix with constant as final column

    """
    return np.column_stack((a, np.ones(len(a))))


def randomize_onesample(a, n_iter=10000, h_0=0, corrected=True,
                        random_seed=None, return_dist=False):
    """Nonparametric one-sample T test through randomization.

    On each iteration, randomly flip the signs of the values in ``a``
    and test the mean against 0.

    If ``a`` is two-dimensional, it is assumed to be shaped as
    (n_observations, n_tests), and a max-statistic based approach
    is used to correct the p values for multiple comparisons over tests.

    Parameters
    ----------
    a : array-like
        input data to test
    n_iter : int
        number of randomization iterations
    h_0 : float, broadcastable to tests in a
        null hypothesis for the group mean
    corrected : bool
        correct the p values in the case of multiple tests
    random_seed : int or None
        seed to use for random number generator
    return_dist : bool
        if True, return the null distribution of t statistics

    Returns
    -------
    obs_t : float or array of floats
        group mean T statistic(s) corresponding to tests in input
    obs_p : float or array of floats
        one-tailed p value that the population mean is greater than h_0
        (1 - percentile under the null)
    dist : ndarray, optional
        if return_dist is True, the null distribution of t statistics

    """
    a = np.asarray(a, np.float)
    if a.ndim < 2:
        a = a.reshape(-1, 1)
    n_samp, n_test = a.shape

    a -= h_0

    rs = np.random.RandomState(random_seed)
    flipper = (rs.uniform(size=(n_samp, n_iter)) > 0.5) * 2 - 1
    flipper = (flipper.reshape(n_samp, 1, n_iter) *
               np.ones((n_samp, n_test, n_iter), int))
    rand_dist = a[:, :, None] * flipper

    err_denom = np.sqrt(n_samp - 1)
    std_err = rand_dist.std(axis=0) / err_denom
    t_dist = rand_dist.mean(axis=0) / std_err

    obs_t = a.mean(axis=0) / (a.std(axis=0) / err_denom)
    if corrected:
        cdf = sm.distributions.ECDF(t_dist.max(axis=0))
        obs_p = 1 - cdf(obs_t)
    else:
        obs_p = []
        for obs_i, null_i in zip(obs_t, t_dist):
            cdf = sm.distributions.ECDF(null_i)
            obs_p.append(1 - cdf(obs_i))
        obs_p = np.array(obs_p)

    if a.shape[1] == 1:
        obs_t = np.asscalar(obs_t)
        obs_p = np.asscalar(obs_p)
        t_dist = t_dist.squeeze()

    if return_dist:
        return obs_t, obs_p, t_dist
    return obs_t, obs_p


def randomize_corrmat(a, tail="both", corrected=True, n_iter=1000,
                      random_seed=None, return_dist=False):
    """Test the significance of set of correlations with permutations.

    By default this corrects for multiple comparisons across one side
    of the matrix.

    Parameters
    ----------
    a : n_vars x n_obs array
        array with variables as rows
    tail : both | upper | lower
        whether test should be two-tailed, or which tail to integrate over
    corrected : boolean
        if True reports p values with respect to the max stat distribution
    n_iter : int
        number of permutation iterations
    random_seed : int or None
        seed for RNG
    return_dist : bool
        if True, return n_vars x n_vars x n_iter

    Returns
    -------
    p_mat : float
        array of probabilites for actual correlation from null CDF

    """
    if tail not in ["upper", "lower", "both"]:
        raise ValueError("'tail' must be 'upper', 'lower', or 'both'")

    rs = np.random.RandomState(random_seed)

    a = np.asarray(a)
    flat_a = a.ravel()
    n_vars, n_obs = a.shape

    # Do the permutations to establish a null distribution
    null_dist = np.empty((n_vars, n_vars, n_iter))
    for i_i in range(n_iter):
        perm_i = np.concatenate([rs.permutation(n_obs) + (v * n_obs)
                                 for v in range(n_vars)])
        a_i = flat_a[perm_i].reshape(n_vars, n_obs)
        null_dist[..., i_i] = np.corrcoef(a_i)

    # Get the observed correlation values
    real_corr = np.corrcoef(a)

    # Figure out p values based on the permutation distribution
    p_mat = np.zeros((n_vars, n_vars))
    upper_tri = np.triu_indices(n_vars, 1)

    if corrected:
        if tail == "both":
            max_dist = np.abs(null_dist[upper_tri]).max(axis=0)
        elif tail == "lower":
            max_dist = null_dist[upper_tri].min(axis=0)
        elif tail == "upper":
            max_dist = null_dist[upper_tri].max(axis=0)

        cdf = sm.distributions.ECDF(max_dist)

        for i, j in zip(*upper_tri):
            observed = real_corr[i, j]
            if tail == "both":
                p_ij = 1 - cdf(abs(observed))
            elif tail == "lower":
                p_ij = cdf(observed)
            elif tail == "upper":
                p_ij = 1 - cdf(observed)
            p_mat[i, j] = p_ij

    else:
        for i, j in zip(*upper_tri):

            null_corrs = null_dist[i, j]
            cdf = sm.distributions.ECDF(null_corrs)

            observed = real_corr[i, j]
            if tail == "both":
                p_ij = 2 * (1 - cdf(abs(observed)))
            elif tail == "lower":
                p_ij = cdf(observed)
            elif tail == "upper":
                p_ij = 1 - cdf(observed)
            p_mat[i, j] = p_ij

    # Make p matrix symettrical with nans on the diagonal
    p_mat += p_mat.T
    p_mat[np.diag_indices(n_vars)] = np.nan

    if return_dist:
        return p_mat, null_dist
    return p_mat


def randomize_classifier(data, model, n_iter=1000, cv_method="run",
                         random_seed=None, return_dist=False, dv=None):
    """Randomly shuffle class labels to build a null distribution of accuracy.

    Randimization can be distributed over an IPython cluster using the ``dv``
    argument. Otherwise, it runs in serial.

    Parameters
    ----------
    data : dict
        single-subject dataset dictionary
    model : scikit-learn estimator
        model object to fit
    n_iter : int
        number of permutation iterations
    cv_method : run | sample | cv arg for cross_val_score
        cross validate over runs, over samples (leave-one-out)
        or otherwise something that can be provided to the cv
        argument for sklearn.cross_val_score
    random_state : int
        seed for random state to obtain stable permutations
    return_dist : bool
        if True, return null distribution
    dv : IPython direct view
        view onto IPython cluster for parallel execution over iterations

    Returns
    -------
    p_vals : n_tp array
        array of one-sided p values for observed classification scores
        against the empirical null distribution
    null_dist : n_iter x n_tp array
        array of null model scores, only if asked for it

    """
    # Import sklearn here to relieve moss dependency on it
    from sklearn.cross_validation import (cross_val_score,
                                          LeaveOneOut, LeaveOneLabelOut)
    if dv is None:
        from six.moves import builtins
        _map = builtins.map
    else:
        _map = dv.map_sync

    # Set up the data properly
    X = data["X"]
    y = data["y"]
    runs = data["runs"]
    if cv_method == "run":
        cv = LeaveOneLabelOut(runs)
    elif cv_method == "sample":
        cv = LeaveOneOut(len(y))
    else:
        cv = cv_method
    if X.ndim < 3:
        X = [X]

    def _perm_decode(model, X, y, cv, perm):
        """Internal func for parallel purposes."""
        y_perm = y[perm]
        perm_acc = cross_val_score(model, X, y_perm, cv=cv).mean()
        return perm_acc

    # Make lists to send into map()
    model_p = [model for i in range(n_iter)]
    y_p = [y for i in range(n_iter)]
    cv_p = [cv for i in range(n_iter)]

    # Permute within run
    rs = np.random.RandomState(random_seed)
    perms = []
    for i in range(n_iter):
        perm_i = []
        for run in np.unique(runs):
            perm_r = rs.permutation(np.sum(runs == run))
            perm_r += np.sum(runs == run - 1)
            perm_i.append(perm_r)
        perms.append(np.concatenate(perm_i))

    # Actually do the permutations, possibly in parallel
    null_dist = []
    for X_i in X:
        X_p = [X_i for i in range(n_iter)]
        tr_scores = list(_map(_perm_decode, model_p, X_p, y_p, cv_p, perms))
        null_dist.append(tr_scores)
    null_dist = np.array(null_dist).T

    # Calculate a p value for each TR
    p_vals = []
    for i, dist_i in enumerate(null_dist.T):
        acc_i = cross_val_score(model, X[i], y, cv=cv).mean()
        cdf_i = sm.distributions.ECDF(dist_i)
        p_vals.append(1 - cdf_i(acc_i))
    p_vals = np.array(p_vals)

    if return_dist:
        return p_vals, null_dist
    return p_vals


def transition_probabilities(sched):
    """Return probability of moving from row trial to col trial.

    Parameters
    ----------
    sched : array or pandas Series
        event schedule

    Returns
    -------
    trans_probs : pandas DataFrame

    """
    sched = np.asarray(sched)
    trial_types = np.sort(np.unique(sched))
    n_types = len(trial_types)
    trans_probs = pd.DataFrame(columns=trial_types, index=trial_types)

    for type in trial_types:
        type_probs = pd.Series(np.zeros(n_types), index=trial_types)
        idx, = np.nonzero(sched[:-1] == type)
        idx += 1
        type_probs.update(pd.value_counts(sched[idx]))
        trans_probs[type] = type_probs

    trans_probs = trans_probs.divide(pd.value_counts(sched[:-1]))
    return trans_probs.T


def upsample(y, by):
    """Upsample a timeseries by some factor using cubic spline interpolation.

    y : array-like
        Input data. Assumes that y is regularly sampled.
    by : int
        Factor to upsample by.

    """
    y = np.asarray(y)
    ntp = len(y)
    x = np.linspace(0, ntp - 1, ntp)
    xx = np.linspace(0, ntp, ntp * by + 1)[:-by]
    yy = interp1d(x, y, "cubic", axis=0)(xx)
    return yy


def remove_unit_variance(df, col, unit, group=None, suffix="_within"):
    """Remove variance between sampling units.

    This is useful for plotting repeated-measures data using within-unit
    error bars.

    Parameters
    ----------
    df : DataFrame
        Input data. Will have a new column added.
    col : column name
        Column in dataframe with quantitative measure to modify.
    unit : column name
        Column in dataframe defining sampling units (e.g., subjects).
    group : column name(s), optional
        Columns defining groups to remove unit variance within.
    suffix : string, optional
        Suffix appended to ``col`` name to create new column.

    Returns
    -------
    df : DataFrame
        Returns modified dataframe.

    """
    new_col = col + suffix

    def demean(x):
        return x - x.mean()

    if group is None:
        new = df.groupby(unit)[col].transform(demean)
        new += df[col].mean()
        df.loc[:, new_col] = new
    else:
        df.loc[:, new_col] = np.nan
        for level, df_level in df.groupby(group):
            new = df_level.groupby(unit)[col].transform(demean)
            new += df_level[col].mean()
            df.loc[new.index, new_col] = new

    return df


def vectorized_correlation(x, y):
    """Compute correlation coefficient between arrays with vectorization.

    Parameters
    ----------
    x, y : array-like
        Dimensions on the final axis should match, computation will be
        vectorized over preceding axes. Dimensions will be matched, or
        broadcasted, depending on shapes. In other words, passing two (m x n)
        arrays will compute the correlation between each pair of rows and
        return a vector of length n. Passing one vector of length n and one
        array of shape (m x n) will compute the correlation between the vector
        and each row in the array, also returning a vector of length n.

    Returns
    -------
    r : array
        Correlation coefficient(s).

    """
    x, y = np.asarray(x), np.asarray(y)
    mx = x.mean(axis=-1)
    my = y.mean(axis=-1)
    xm, ym = x - mx[..., None], y - my[..., None]
    r_num = np.add.reduce(xm * ym, axis=-1)
    r_den = np.sqrt(stats.ss(xm, axis=-1) * stats.ss(ym, axis=-1))
    r = r_num / r_den
    return r


def percent_change(ts, n_runs=1):
    """Convert to percent signal change by run.

    Assumes all runs have the same length.

    Parameters
    ----------
    ts : array or DataFrame
        Timeseries data with timepoints in the columns.
    n_runs : int
        Number of runs to split the timeseries into.

    Returns
    -------
    out_ts : array or DataFrame
        Rescaled timeseries with type of input.

    """
    if not isinstance(ts, pd.DataFrame):
        ts = pd.DataFrame(np.atleast_2d(ts))
        dataframe = False
    else:
        dataframe = True

    run_tps = np.split(ts.columns, n_runs)

    # Iterate over runs
    run_data = []
    for run in range(n_runs):
        run_ts = ts[run_tps[run]]
        run_ts = (run_ts.divide(run_ts.mean(axis=1), axis=0) - 1) * 100
        run_data.append(run_ts)

    out_ts = pd.concat(run_data, axis=1)

    # Return input type
    if not dataframe:
        out_ts = out_ts.values

    return out_ts
