"""Utilities for experimental design."""
from __future__ import division
import numpy as np
from numpy.random import permutation, multinomial


def optimize_event_schedule(n_cat, n_total, max_repeat,
                            n_search=1000, enforce_balance=False):
    """Generate an event schedule optimizing CB1 and even conditions.

    Parameters
    ----------
    n_cat: int
        Total number of event types
    n_total: int
        Total number of events
    max_repeat: int
        Maximum number of event repetitions allowed
    n_search: int
        Size of the searc space
    enforce_balance: bool
        If true, raises a ValueError if event types are not balanced

    Returns
    -------
    schedule: numpy array
        Optimal event schedule with 0-based event ids

    """
    # Determine the idea transition matrix
    ev_count = [n_total / n_cat] * n_cat
    ideal = cb1_ideal(ev_count)

    # Generate the space of schedules
    schedules = []
    bal_costs = np.zeros(n_search)
    cb1_costs = np.zeros(n_search)
    for i in xrange(n_search):
        sched = make_schedule(n_cat, n_total, max_repeat)
        schedules.append(sched)

        # Determine balance cost
        hist = np.histogram(sched, n_cat)[0]
        bal_costs[i] = np.sum(np.abs(hist - hist.mean()))

        # Determine CB1 cost
        cb1_mat = cb1_prob(sched, ev_count)
        cb1_costs[i] = cb1_cost(ideal, cb1_mat)

    # Possibly error out if schedules are not balanced
    if enforce_balance and bal_costs.min():
        raise ValueError("Failed to generate balanced schedule")

    # Zscore the two costs and sum
    zscore = lambda x: (x - x.mean()) / x.std()
    bal_costs = zscore(bal_costs)
    cb1_costs = zscore(cb1_costs)
    costs = bal_costs + cb1_costs

    # Return the best schdule
    return np.array(schedules[np.argmin(costs)])


def make_schedule(n_cat, n_total, max_repeat):
    """Generate an event schedule subject to a repeat constraint."""

    # Make the uniform transition matrix
    ideal_tmat = [1 / n_cat] * n_cat
    # Build the transition matrices for when we've exceeded our repeat limit
    const_mat_list = []
    for i in range(n_cat):
        const_mat_list.append([1 / (n_cat - 1)] * n_cat)
        const_mat_list[-1][i] = 0

    # Convenience function to make the transitions
    cat_range = np.arange(n_cat)
    draw = lambda x: np.asscalar(cat_range[multinomial(1, x).astype(bool)])

    # Generate the schedule
    schedule = []
    for i in xrange(n_total):
        trailing_set = set(schedule[-max_repeat:])
        # Check if we're at our repeat limit
        if len(trailing_set) == 1:
            tdist = const_mat_list[trailing_set.pop()]
        else:
            tdist = ideal_tmat
        # Assign this iteration's state
        schedule.append(draw(tdist))

    return schedule


def cb1_optimize(ev_count, n_search=1000, constraint=None):
    """Given event counts, return a first order counterbalanced schedule.

    Note that this is a stupid brute force algorithm. It's bascially a Python
    port of Doug Greve's C implementation of this in optseq with the addition
    of a constraint option.

    Inputs
    ------
    ev_count: sequence
        desired number of appearences for each event t
    constraint: callable
        arbitrary function that takes a squence and returns a boolean
    n_search: int
        iterations of search algorithm

    """
    # Figure the total event count
    ev_count = np.asarray(ev_count)
    n_total = ev_count.sum()

    # Set up a default constraint function
    if constraint is None:
        constraint = lambda x: True

    # Create the ideal FOCB matrix
    ideal = cb1_ideal(ev_count)

    # Make an unordered schedule
    sched_list = []
    for i, n in enumerate(ev_count, 1):
        sched_list.append(np.ones(n, int) * i)
    sched = np.hstack(sched_list)

    # Create n_search random schedules and pick the best one
    sched_costs = np.zeros(n_search)
    best_sched = sched
    for i in xrange(n_search):
        iter_sched = sched[permutation(int(n_total))]
        if not constraint(iter_sched):
            continue
        iter_cb1_mat = cb1_prob(iter_sched, ev_count)
        iter_cost = cb1_cost(ideal, iter_cb1_mat)
        sched_costs[i] = iter_cost
        if (not i) or (cb1_cost == sched_costs[:i].min()):
            best_sched = iter_sched

    # Make sure we could permute
    if np.array_equal(best_sched, sched):
        raise ValueError("Could not satisfy constraint")

    return best_sched


def max_three_in_a_row(seq):
    """Only allow sequences with 3 or fewer tokens in a row.

    This assumes the tokens are represnted by integers in [0, 9].

    """
    seq_str = "".join(map(str, seq))
    for item in np.unique(seq):
        item_str = str(item)
        check_str = "".join([item_str for i in range(4)])
        if check_str in seq_str:
            return False
    return True


def max_four_in_a_row(seq):
    """Only allow sequences with 4 or fewer tokens in a row.

    This assumes the tokens are represnted by integers in [0, 9].

    """
    seq_str = "".join(map(str, seq))
    for item in np.unique(seq):
        item_str = str(item)
        check_str = "".join([item_str for i in range(5)])
        if check_str in seq_str:
            return False
    return True


def cb1_ideal(ev_count):
    """Calculate the ideal FOCB matrix"""
    n_events = len(ev_count)
    ideal = np.zeros((n_events, n_events))
    ideal[:] = ev_count / np.sum(ev_count)
    return ideal


def cb1_prob(sched, ev_count):
    """Calculate the empirical FOCB matrix from a schedule."""
    n_events = len(ev_count)
    cb_mat = np.zeros((n_events, n_events))
    for i, event in enumerate(sched[:-1]):
        next_event = sched[i + 1]
        cb_mat[event - 1, next_event - 1] += 1

    for i, count in enumerate(ev_count):
        cb_mat[i] /= count

    return cb_mat


def cb1_cost(ideal_mat, test_mat):
    """Calculate the error between ideal and empirical FOCB matricies."""
    cb1err = np.abs(ideal_mat - test_mat)
    cb1err /= ideal_mat
    cb1err = cb1err.sum()
    cb1err /= ideal_mat.shape[0] ** 2
    return cb1err


def build_simple_ev(data, onset, name, duration=None):
    """Make design info for a single-column constant-value ev.

    Parameters
    ----------
    data : DataFrame
        Input data; must have "run" column and any others specified.
    onset : string
        Column name containing event onset information.
    name : string
        Condition name to use for this ev.
    duration : string, float, or ``None``
        Column name containing event duration information, or a value
        to use for all events, or ``None`` to model events as impulses.

    Returns
    -------
    ev : DataFrame
        Returned DataFrame will have "run", "onset", "duration", "value",
        and "condition" columns.

    """
    ev = data[["run", onset]].copy()
    ev.columns = ["run", "onset"]

    # Set a constant amplitude for all events
    ev.loc[:, "value"] = 1

    # Use the same condition name for all events
    ev.loc[:, "condition"] = name

    # Determine the event duration
    ev = _add_duration_information(data, ev, duration)

    return ev


def build_condition_ev(data, onset, condition, duration=None, prefix=None):
    """Make design info for a multi-column constant-value ev.

    Parameters
    ----------
    data : DataFrame
        Input data; must have "run" column and any others specified.
    onset : string
        Column name containing event onset information.
    condition : string
        Column name containing condition information.
    duration : string, float, or ``None``
        Column name containing event duration information, or a value
        to use for all events, or ``None`` to model events as impulses.
    prefix : string or ``None``
        Prefix to add to all condition names.

    Returns
    -------
    ev : DataFrame
        Returned DataFrame will have "run", "onset", "duration", "value",
        and "condition" columns.

    """
    ev = data[["run", onset, condition]].copy()
    ev.columns = ["run", "onset", "condition"]

    if prefix is not None:
        ev["condition"] = prefix + ev["condition"]

    # Set a constant amplitude for all events
    ev["value"] = 1

    # Determine the event duration
    ev = _add_duration_information(data, ev, duration)

    return ev


def build_parametric_ev(data, onset, name, value, duration=None,
                        center=None, scale=None):
    """Make design info for a multi-column constant-value ev.

    Parameters
    ----------
    data : DataFrame
        Input data; must have "run" column and any others specified.
    onset : string
        Column name containing event onset information.
    name : string
        Condition name to use for this ev.
    value : string
        Column name containing event amplitude information.
    duration : string, float, or ``None``
        Column name containing event duration information, or a value
        to use for all events, or ``None`` to model events as impulses.
    center : float, optional
        Value to center the ``value`` column at before scaling. If absent,
        center at the mean across runs.
    scale : callable, optional
        Function to scale the centered value column with.

    Returns
    -------
    ev : DataFrame
        Returned DataFrame will have "run", "onset", "duration", "value",
        and "condition" columns.

    """
    ev = data[["run", onset, value]].copy()
    ev.columns = ["run", "onset", "value"]

    # Center the event amplitude
    if center is None:
        ev["value"] -= ev.value.mean()
    else:
        ev["value"] = ev.value - center

    # (Possibly) scale the event amplitude
    if scale is not None:
        ev["value"] = scale(ev["value"])

    # Set a condition name for all events
    ev["condition"] = name

    # Determine the event duration
    ev = _add_duration_information(data, ev, duration)

    return ev


def _add_duration_information(data, ev, duration):
    """Determine event duration in one of a few ways."""
    if duration is None:
        # All events modeled as an impulse
        duration = 0
    elif duration in data:
        # Each event gets its own duration
        duration = data[duration]
    ev["duration"] = duration

    return ev
