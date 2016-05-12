import numpy as np
import scipy as sp
from scipy import stats as spstats
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from six.moves import range

from numpy.testing import assert_array_equal, assert_array_almost_equal
import numpy.testing as npt
import nose.tools
import nose.tools as nt
from nose.tools import assert_equal, assert_almost_equal, raises
import pandas.util.testing as pdt

from .. import statistical as stat

rs = np.random.RandomState(sum(map(ord, "moss_stats")))

a_norm = rs.randn(100)

a_range = np.arange(101)

datasets = [dict(X=spstats.norm(0, 1).rvs((24, 12)),
                 y=spstats.bernoulli(.5).rvs(24),
                 runs=np.repeat([0, 1], 12)) for i in range(3)]

datasets_3d = [dict(X=spstats.norm(0, 1).rvs((4, 24, 12)),
                    y=spstats.bernoulli(.5).rvs(24),
                    runs=np.repeat([0, 1], 12)) for i in range(3)]


def test_bootstrap():
    """Test that bootstrapping gives the right answer in dumb cases."""
    a_ones = np.ones(10)
    n_boot = 5
    out1 = stat.bootstrap(a_ones, n_boot=n_boot)
    assert_array_equal(out1, np.ones(n_boot))
    out2 = stat.bootstrap(a_ones, n_boot=n_boot, func=np.median)
    assert_array_equal(out2, np.ones(n_boot))


def test_bootstrap_length():
    """Test that we get a bootstrap array of the right shape."""
    out = stat.bootstrap(a_norm)
    assert_equal(len(out), 10000)

    n_boot = 100
    out = stat.bootstrap(a_norm, n_boot=n_boot)
    assert_equal(len(out), n_boot)


def test_bootstrap_range():
    """Test that boostrapping a random array stays within the right range."""
    min, max = a_norm.min(), a_norm.max()
    out = stat.bootstrap(a_norm)
    nose.tools.assert_less(min, out.min())
    nose.tools.assert_greater_equal(max, out.max())


def test_bootstrap_multiarg():
    """Test that bootstrap works with multiple input arrays."""
    x = np.vstack([[1, 10] for i in range(10)])
    y = np.vstack([[5, 5] for i in range(10)])

    def test_func(x, y):
        return np.vstack((x, y)).max(axis=0)

    out_actual = stat.bootstrap(x, y, n_boot=2, func=test_func)
    out_wanted = np.array([[5, 10], [5, 10]])
    assert_array_equal(out_actual, out_wanted)


def test_bootstrap_axis():
    """Test axis kwarg to bootstrap function."""
    x = rs.randn(10, 20)
    n_boot = 100
    out_default = stat.bootstrap(x, n_boot=n_boot)
    assert_equal(out_default.shape, (n_boot,))
    out_axis = stat.bootstrap(x, n_boot=n_boot, axis=0)
    assert_equal(out_axis.shape, (n_boot, 20))


def test_bootstrap_random_seed():
    """Test that we can get reproducible resamples by seeding the RNG."""
    data = rs.randn(50)
    seed = 42
    boots1 = stat.bootstrap(data, random_seed=seed)
    boots2 = stat.bootstrap(data, random_seed=seed)
    assert_array_equal(boots1, boots2)


def test_smooth_bootstrap():
    """Test smooth bootstrap."""
    x = rs.randn(15)
    n_boot = 100
    out_normal = stat.bootstrap(x, n_boot=n_boot, func=np.median)
    out_smooth = stat.bootstrap(x, n_boot=n_boot,
                                smooth=True, func=np.median)
    assert(np.median(out_normal) in x)
    assert(not np.median(out_smooth) in x)


def test_bootstrap_ols():
    """Test bootstrap of OLS model fit."""
    def ols_fit(X, y):
        return np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)
    X = np.column_stack((rs.randn(50, 4), np.ones(50)))
    w = [2, 4, 0, 3, 5]
    y_noisy = np.dot(X, w) + rs.randn(50) * 20
    y_lownoise = np.dot(X, w) + rs.randn(50)

    n_boot = 500
    w_boot_noisy = stat.bootstrap(X, y_noisy,
                                  n_boot=n_boot,
                                  func=ols_fit)
    w_boot_lownoise = stat.bootstrap(X, y_lownoise,
                                     n_boot=n_boot,
                                     func=ols_fit)

    assert_equal(w_boot_noisy.shape, (n_boot, 5))
    assert_equal(w_boot_lownoise.shape, (n_boot, 5))
    nose.tools.assert_greater(w_boot_noisy.std(),
                              w_boot_lownoise.std())


def test_bootstrap_units():
    """Test that results make sense when passing unit IDs to bootstrap."""
    data = rs.randn(50)
    ids = np.repeat(range(10), 5)
    bwerr = rs.normal(0, 2, 10)
    bwerr = bwerr[ids]
    data_rm = data + bwerr
    seed = 77

    boots_orig = stat.bootstrap(data_rm, random_seed=seed)
    boots_rm = stat.bootstrap(data_rm, units=ids, random_seed=seed)
    nose.tools.assert_greater(boots_rm.std(), boots_orig.std())


@raises(ValueError)
def test_bootstrap_arglength():
    """Test that different length args raise ValueError."""
    stat.bootstrap(np.arange(5), np.arange(10))


@raises(TypeError)
def test_bootstrap_noncallable():
    """Test that we get a TypeError with noncallable statfunc."""
    non_func = "mean"
    stat.bootstrap(a_norm, 100, non_func)


def test_percentiles():
    """Test function to return sequence of percentiles."""
    single_val = 5
    single = stat.percentiles(a_range, single_val)
    assert_equal(single, single_val)

    multi_val = [10, 20]
    multi = stat.percentiles(a_range, multi_val)
    assert_array_equal(multi, multi_val)

    array_val = rs.randint(0, 101, 5).astype(float)
    array = stat.percentiles(a_range, array_val)
    assert_array_almost_equal(array, array_val)


def test_percentiles_acc():
    """Test accuracy of calculation."""
    # First a basic case
    data = np.array([10, 20, 30])
    val = 20
    perc = stat.percentiles(data, 50)
    assert_equal(perc, val)

    # Now test against scoreatpercentile
    percentiles = rs.randint(0, 101, 10)
    out = stat.percentiles(a_norm, percentiles)
    for score, pct in zip(out, percentiles):
        assert_equal(score, sp.stats.scoreatpercentile(a_norm, pct))


def test_percentiles_axis():
    """Test use of axis argument to percentils."""
    data = rs.randn(10, 10)

    # Test against the median with 50th percentile
    median1 = np.median(data)
    out1 = stat.percentiles(data, 50)
    assert_array_almost_equal(median1, out1)

    for axis in range(2):
        median2 = np.median(data, axis=axis)
        out2 = stat.percentiles(data, 50, axis=axis)
        assert_array_almost_equal(median2, out2)

    median3 = np.median(data, axis=0)
    out3 = stat.percentiles(data, [50, 95], axis=0)
    assert_array_almost_equal(median3, out3[0])
    assert_equal(2, len(out3))


def test_ci():
    """Test ci against percentiles."""
    a = rs.randn(100)
    p = stat.percentiles(a, [2.5, 97.5])
    c = stat.ci(a, 95)
    assert_array_equal(p, c)


def test_vector_reject():
    """Test vector rejection function."""
    x = rs.randn(30)
    y = x + rs.randn(30) / 2
    x_ = stat.vector_reject(x, y)
    assert_almost_equal(np.dot(x_, y), 0)


def test_add_constant():
    """Test the add_constant function."""
    a = rs.randn(10, 5)
    wanted = np.column_stack((a, np.ones(10)))
    got = stat.add_constant(a)
    assert_array_equal(wanted, got)


def test_randomize_onesample():
    """Test performance of randomize_onesample."""
    a_zero = rs.normal(0, 1, 50)
    t_zero, p_zero = stat.randomize_onesample(a_zero)
    nose.tools.assert_greater(p_zero, 0.05)

    a_five = rs.normal(5, 1, 50)
    t_five, p_five = stat.randomize_onesample(a_five)
    nose.tools.assert_greater(0.05, p_five)

    t_scipy, p_scipy = sp.stats.ttest_1samp(a_five, 0)
    nose.tools.assert_almost_equal(t_scipy, t_five)


def test_randomize_onesample_range():
    """Make sure that output is bounded between 0 and 1."""
    for i in range(100):
        a = rs.normal(rs.randint(-10, 10),
                      rs.uniform(.5, 3), 100)
        t, p = stat.randomize_onesample(a, 100)
        nose.tools.assert_greater_equal(1, p)
        nose.tools.assert_greater_equal(p, 0)


def test_randomize_onesample_getdist():
    """Test that we can get the null distribution if we ask for it."""
    a = rs.normal(0, 1, 20)
    out = stat.randomize_onesample(a, return_dist=True)
    assert_equal(len(out), 3)


def test_randomize_onesample_iters():
    """Make sure we get the right number of samples."""
    a = rs.normal(0, 1, 20)
    t, p, samples = stat.randomize_onesample(a, return_dist=True)
    assert_equal(len(samples), 10000)
    for n in rs.randint(5, 1e4, 5):
        t, p, samples = stat.randomize_onesample(a, n, return_dist=True)
        assert_equal(len(samples), n)


def test_randomize_onesample_seed():
    """Test that we can seed the random state and get the same distribution."""
    a = rs.normal(0, 1, 20)
    seed = 42
    t_a, p_a, samples_a = stat.randomize_onesample(a, 1000,
                                                   random_seed=seed,
                                                   return_dist=True)
    t_b, t_b, samples_b = stat.randomize_onesample(a, 1000,
                                                   random_seed=seed,
                                                   return_dist=True)
    assert_array_equal(samples_a, samples_b)


def test_randomize_onesample_multitest():
    """Test that randomizing over multiple tests works."""
    a = rs.normal(0, 1, (20, 5))
    t, p = stat.randomize_onesample(a, 1000)
    assert_equal(len(t), 5)
    assert_equal(len(p), 5)

    t, p, dist = stat.randomize_onesample(a, 1000, return_dist=True)
    assert_equal(dist.shape, (5, 1000))


def test_randomize_onesample_correction():
    """Test that maximum based correction (seems to) work."""
    a = rs.normal(0, 1, (100, 10))
    t_un, p_un = stat.randomize_onesample(a, 1000, corrected=False)
    t_corr, p_corr = stat.randomize_onesample(a, 1000, corrected=True)
    assert_array_equal(t_un, t_corr)
    npt.assert_array_less(p_un, p_corr)


def test_randomize_onesample_h0():
    """Test that we can supply a null hypothesis for the group mean."""
    a = rs.normal(4, 1, 100)
    t, p = stat.randomize_onesample(a, 1000, h_0=0)
    assert p < 0.01

    t, p = stat.randomize_onesample(a, 1000, h_0=4)
    assert p > 0.01


def test_randomize_onesample_scalar():
    """Single values returned from randomize_onesample should be scalars."""
    a = rs.randn(40)
    t, p = stat.randomize_onesample(a)
    assert np.isscalar(t)
    assert np.isscalar(p)

    a = rs.randn(40, 3)
    t, p = stat.randomize_onesample(a)
    assert not np.isscalar(t)
    assert not np.isscalar(p)


def test_randomize_corrmat():
    """Test the correctness of the correlation matrix p values."""
    a = rs.randn(30)
    b = a + rs.rand(30) * 3
    c = rs.randn(30)
    d = [a, b, c]

    p_mat, dist = stat.randomize_corrmat(d, tail="upper", corrected=False,
                                         return_dist=True)
    nose.tools.assert_greater(p_mat[2, 0], p_mat[1, 0])

    corrmat = np.corrcoef(d)
    pctile = 100 - spstats.percentileofscore(dist[2, 1], corrmat[2, 1])
    nose.tools.assert_almost_equal(p_mat[2, 1] * 100, pctile)

    d[1] = -a + rs.rand(30)
    p_mat = stat.randomize_corrmat(d)
    nose.tools.assert_greater(0.05, p_mat[1, 0])


def test_randomize_corrmat_dist():
    """Test that the distribution looks right."""
    a = rs.randn(3, 20)
    for n_i in [5, 10]:
        p_mat, dist = stat.randomize_corrmat(a, n_iter=n_i, return_dist=True)
        assert_equal(n_i, dist.shape[-1])

    p_mat, dist = stat.randomize_corrmat(a, n_iter=10000, return_dist=True)

    diag_mean = dist[0, 0].mean()
    assert_equal(diag_mean, 1)

    off_diag_mean = dist[0, 1].mean()
    nose.tools.assert_greater(0.05, off_diag_mean)


def test_randomize_corrmat_correction():
    """Test that FWE correction works."""
    a = rs.randn(3, 20)
    p_mat = stat.randomize_corrmat(a, "upper", False)
    p_mat_corr = stat.randomize_corrmat(a, "upper", True)
    triu = np.triu_indices(3, 1)
    npt.assert_array_less(p_mat[triu], p_mat_corr[triu])


def test_randimoize_corrmat_tails():
    """Test that the tail argument works."""
    a = rs.randn(30)
    b = a + rs.rand(30) * 8
    c = rs.randn(30)
    d = [a, b, c]

    p_mat_b = stat.randomize_corrmat(d, "both", False, random_seed=0)
    p_mat_u = stat.randomize_corrmat(d, "upper", False, random_seed=0)
    p_mat_l = stat.randomize_corrmat(d, "lower", False, random_seed=0)
    assert_equal(p_mat_b[0, 1], p_mat_u[0, 1] * 2)
    assert_equal(p_mat_l[0, 1], 1 - p_mat_u[0, 1])


def test_randomise_corrmat_seed():
    """Test that we can seed the corrmat randomization."""
    a = rs.randn(3, 20)
    _, dist1 = stat.randomize_corrmat(a, random_seed=0, return_dist=True)
    _, dist2 = stat.randomize_corrmat(a, random_seed=0, return_dist=True)
    assert_array_equal(dist1, dist2)


@raises(ValueError)
def test_randomize_corrmat_tail_error():
    """Test that we are strict about tail paramete."""
    a = rs.randn(3, 30)
    stat.randomize_corrmat(a, "hello")


def test_randomize_classifier():
    """Test basic functions of randomize_classifier."""
    data = dict(X=spstats.norm(0, 1).rvs((100, 12)),
                y=spstats.bernoulli(.5).rvs(100),
                runs=np.repeat([0, 1], 50))
    model = GaussianNB()
    p_vals, perm_vals = stat.randomize_classifier(data, model,
                                                  return_dist=True)
    p_min, p_max = p_vals.min(), p_vals.max()
    perm_mean = perm_vals.mean()

    # Test that the p value are well behaved
    nose.tools.assert_greater_equal(1, p_max)
    nose.tools.assert_greater_equal(p_min, 0)

    # Test that the mean is close to chance (this is probabilistic)
    nose.tools.assert_greater(.1, np.abs(perm_mean - 0.5))

    # Test that the distribution looks normal (this is probabilistic)
    val, p = spstats.normaltest(perm_vals)
    nose.tools.assert_greater(p, 0.001)


def test_randomize_classifier_dimension():
    """Test that we can have a time dimension and it's where we expect."""
    data = datasets_3d[0]
    n_perm = 30
    model = GaussianNB()
    p_vals, perm_vals = stat.randomize_classifier(data, model, n_perm,
                                                  return_dist=True)
    nose.tools.assert_equal(len(p_vals), len(data["X"]))
    nose.tools.assert_equal(perm_vals.shape, (n_perm, len(data["X"])))


def test_randomize_classifier_seed():
    """Test that we can give a particular random seed to the permuter."""
    data = datasets[0]
    model = GaussianNB()
    seed = 1
    out_a = stat.randomize_classifier(data, model, random_seed=seed)
    out_b = stat.randomize_classifier(data, model, random_seed=seed)
    assert_array_equal(out_a, out_b)


def test_randomize_classifier_number():
    """Test size of randomize_classifier vectors."""
    data = datasets[0]
    model = GaussianNB()
    for n_iter in rs.randint(10, 250, 5):
        p_vals, perm_dist = stat.randomize_classifier(data, model, n_iter,
                                                      return_dist=True)
        nose.tools.assert_equal(len(perm_dist), n_iter)


def test_transition_probabilities():

    # Test basic
    sched = [0, 1, 0, 1]
    expected = pd.DataFrame([[0, 1], [1, 0]])
    actual = stat.transition_probabilities(sched)
    npt.assert_array_equal(expected, actual)

    sched = [0, 0, 1, 1]
    expected = pd.DataFrame([[.5, .5], [0, 1]])
    actual = stat.transition_probabilities(sched)
    npt.assert_array_equal(expected, actual)

    a = rs.rand(100) < .5
    a = np.where(a, "foo", "bar")
    out = stat.transition_probabilities(a)
    npt.assert_array_equal(out.columns.tolist(), ["bar", "foo"])
    npt.assert_array_equal(out.columns, out.index)


def test_upsample():

    y = np.cumsum(rs.randn(100))

    yy1 = stat.upsample(y, 1)
    assert_equal(len(yy1), 100)
    npt.assert_array_almost_equal(y, yy1)

    yy2 = stat.upsample(y, 2)
    assert_equal(len(yy2), 199)
    npt.assert_array_almost_equal(y, yy2[::2])


class TestRemoveUnitVariance(object):

    rs = np.random.RandomState(93)
    df = pd.DataFrame(dict(value=rs.rand(8),
                           group=np.repeat(np.tile(["m", "n"], 2), 2),
                           cond=np.tile(["x", "y"], 4),
                           unit=np.repeat(["a", "b"], 4)))

    def test_remove_all(self):

        df = stat.remove_unit_variance(self.df, "value", "unit")
        nt.assert_in("value_within", df)

        nt.assert_equal(self.df.value.mean(), self.df.value_within.mean())
        nt.assert_equal(self.df.groupby("unit").value_within.mean().var(), 0)

    def test_remove_by_group(self):

        df = stat.remove_unit_variance(self.df, "value", "unit", "group")
        grp = df.groupby("group")
        pdt.assert_series_equal(grp.value.mean(), grp.value_within.mean(),
                                check_names=False)

        for _, g in grp:
            nt.assert_equal(g.groupby("unit").value_within.mean().var(), 0)

    def test_suffix(self):

        df = stat.remove_unit_variance(self.df, "value", "unit", suffix="_foo")
        nt.assert_in("value_foo", df)


class TestVectorizedCorrelation(object):

    rs = np.random.RandomState()
    a = rs.randn(50)
    b = rs.randn(50)
    c = rs.randn(5, 50)
    d = rs.randn(5, 50)

    def test_vector_to_vector(self):

        r_got = stat.vectorized_correlation(self.a, self.b)
        r_want, _ = spstats.pearsonr(self.a, self.b)
        npt.assert_almost_equal(r_got, r_want)

    def test_vector_to_matrix(self):

        r_got = stat.vectorized_correlation(self.a, self.c)
        nt.assert_equal(r_got.shape, (self.c.shape[0],))

        for i, r_got_i in enumerate(r_got):
            r_want_i, _ = spstats.pearsonr(self.a, self.c[i])
            npt.assert_almost_equal(r_got_i, r_want_i)

    def test_matrix_to_matrix(self):

        r_got = stat.vectorized_correlation(self.c, self.d)
        nt.assert_equal(r_got.shape, (self.c.shape[0],))

        for i, r_got_i in enumerate(r_got):
            r_want_i, _ = spstats.pearsonr(self.c[i], self.d[i])
            npt.assert_almost_equal(r_got_i, r_want_i)


class TestPercentChange(object):

    ts_array = np.arange(6).reshape(1, 6)
    ts = pd.DataFrame(ts_array)

    def test_df(self):

        out = stat.percent_change(self.ts)
        want = pd.DataFrame([[-100, -60, -20, 20, 60, 100]], dtype=np.float)
        pdt.assert_frame_equal(out, want)

    def test_df_multirun(self):

        out = stat.percent_change(self.ts, 2)
        want = pd.DataFrame([[-100, 0, 100, -25, 0, 25]], dtype=np.float)
        pdt.assert_frame_equal(out, want)

    def test_array(self):

        out = stat.percent_change(self.ts_array, 2)
        want = np.array([[-100, 0, 100, -25, 0, 25]], np.float)
        npt.assert_array_equal(out, want)
