from textwrap import dedent
import numpy as np
import pandas as pd
import pandas.util.testing as pdt
import pytest

from .. import params


@pytest.fixture
def initial_params_dict():
    return dict(alpha=1.0, beta=2.0)


@pytest.fixture
def initial_params_series():
    return pd.Series([1.0, 2.0], ["alpha", "beta"])


@pytest.fixture
def basic_paramset():
    p = pd.Series([1.0, 2.0], ["alpha", "beta"])
    return params.ParamSet(p)


@pytest.fixture
def fixed_paramset():
    p = pd.Series([1.0, 2.0], ["alpha", "beta"])
    return params.ParamSet(p, ["beta"])


def test_paramset_series_initialization(initial_params_series):

    p = params.ParamSet(initial_params_series)
    pdt.assert_series_equal(p.params, initial_params_series)
    pdt.assert_series_equal(p.free, initial_params_series)
    pdt.assert_series_equal(p.fixed, pd.Series([]))


def test_paramset_dict_initialization(initial_params_dict):

    p = params.ParamSet(initial_params_dict)
    params_series = pd.Series(initial_params_dict)
    pdt.assert_series_equal(p.params, params_series)
    pdt.assert_series_equal(p.free, params_series)
    pdt.assert_series_equal(p.fixed, pd.Series([]))


def test_paramset_fixed_initialization(initial_params_series):

    free_names = ["alpha"]
    fixed_names = ["beta"]
    p = params.ParamSet(initial_params_series, fixed_names)
    pdt.assert_series_equal(p.params, initial_params_series)
    pdt.assert_series_equal(p.free, initial_params_series[free_names])
    pdt.assert_series_equal(p.fixed, initial_params_series[fixed_names])
    assert p.free_names == free_names
    assert p.fixed_names == fixed_names

    with pytest.raises(ValueError):
        p = params.ParamSet(initial_params_series, ["gamma"])


def test_paramset_repr(basic_paramset, fixed_paramset):

    expected_repr_without_fixed = dedent("""\
    Free Parameters:
      alpha: 1
      beta: 2
    """)
    assert basic_paramset.__repr__() == expected_repr_without_fixed

    expected_repr_with_fixed = dedent("""\
    Free Parameters:
      alpha: 1
    Fixed Parameters:
      beta: 2
    """)
    assert fixed_paramset.__repr__() == expected_repr_with_fixed


def test_paramset_paramset_update(basic_paramset):

    update_series = pd.Series([3.0, 4.0], ["alpha", "beta"])
    update_paramset = params.ParamSet(update_series)
    new_paramset = basic_paramset.update(update_paramset)

    pdt.assert_series_equal(basic_paramset.params, new_paramset.params)
    pdt.assert_series_equal(basic_paramset.params, update_series)


def test_paramset_series_update(basic_paramset):

    update = pd.Series([3.0, 4.0], ["alpha", "beta"])
    new_paramset = basic_paramset.update(update)

    pdt.assert_series_equal(basic_paramset.params, new_paramset.params)
    pdt.assert_series_equal(basic_paramset.params, update)


def test_paramset_dict_update(basic_paramset):

    update_dict = dict(alpha=3.0, beta=4.0)
    update_series = pd.Series(update_dict, index=basic_paramset.names)

    new_paramset = basic_paramset.update(update_dict)

    pdt.assert_series_equal(basic_paramset.params, new_paramset.params)
    pdt.assert_series_equal(basic_paramset.params, update_series)


def test_paramset_vector_update(basic_paramset):

    update_series = pd.Series([3.0, 4.0], ["alpha", "beta"])
    update_vector = np.asarray(update_series)

    new_paramset = basic_paramset.update(update_vector)

    pdt.assert_series_equal(basic_paramset.params, new_paramset.params)
    pdt.assert_series_equal(basic_paramset.params, update_series)


def test_paramset_fixed_series_update(fixed_paramset):

    update_series = pd.Series([3.0], ["alpha"])
    new_series = pd.Series([3.0, 2.0], ["alpha", "beta"])
    new_paramset = fixed_paramset.update(update_series)

    pdt.assert_series_equal(fixed_paramset.params, new_paramset.params)
    pdt.assert_series_equal(fixed_paramset.params, new_series)


def test_paramset_update_fixed_exception(fixed_paramset, basic_paramset):

    update_dict = dict(beta=4.0)
    with pytest.raises(ValueError):
        fixed_paramset.update(update_dict)

    update_series = pd.Series(update_dict)
    with pytest.raises(ValueError):
        fixed_paramset.update(update_series)

    with pytest.raises(ValueError):
        fixed_paramset.update(basic_paramset)


def test_paramset_attribute_access(basic_paramset):

    assert basic_paramset.alpha == 1
    assert basic_paramset.beta == 2

    basic_paramset.update(dict(alpha=3, beta=4))
    assert basic_paramset.alpha == 3
    assert basic_paramset.beta == 4


def test_paramset_attribute_setting(basic_paramset):

    # TODO commented out until I can figure out the infinite recursion issues
    # basic_paramset.alpha = 3
    # assert basic_paramset.params["alpha"] == 3
    pass
