import numpy as np
import pytest

from spml.ensemble import (
    BaggedSpatialIndexEnsemble,
    KDTreeEnsembleRegressor,
    QuadtreeEnsembleRegressor,
)
from spml.ensemble._bagged import (
    _softmax_weights,
    _weighted_mean,
    _weighted_median,
)


# --------------------------------------------------------------------------
# unit tests on the pure combiners (deterministic, no fitting)
# --------------------------------------------------------------------------

def test_weighted_median_known():
    # one column: values 1, 2, 100 with weights .4, .4, .2 -> lower weighted
    # median crosses 0.5 at the value 2 (mean would be 21.2).
    P = np.array([[1.0], [2.0], [100.0]])
    w = np.array([[0.4], [0.4], [0.2]])
    assert _weighted_median(P, w)[0] == 2.0


def test_weighted_median_equals_plain_median_when_uniform():
    P = np.array([[1.0], [2.0], [3.0]])
    w = np.full((3, 1), 1 / 3)
    assert _weighted_median(P, w)[0] == 2.0


def test_weighted_mean_known():
    P = np.array([[1.0], [2.0], [100.0]])
    w = np.array([[0.4], [0.4], [0.2]])
    assert np.isclose(_weighted_mean(P, w)[0], 0.4 + 0.8 + 20.0)


def test_weighted_mean_ignores_nan_members():
    # a NaN prediction carries zero weight and must not poison the mean.
    P = np.array([[1.0], [np.nan], [3.0]])
    w = np.array([[0.5], [0.0], [0.5]])
    assert np.isclose(_weighted_mean(P, w)[0], 2.0)


def test_softmax_weights_uniform_on_zero_skill():
    skill = np.zeros((4, 3))
    valid = np.ones((4, 3), dtype=bool)
    w = _softmax_weights(skill, valid, temperature=1.0)
    assert np.allclose(w, 0.25)
    assert np.allclose(w.sum(axis=0), 1.0)


def test_softmax_weights_large_temperature_approaches_uniform():
    rng = np.random.default_rng(0)
    skill = rng.normal(size=(5, 7))
    valid = np.ones((5, 7), dtype=bool)
    w = _softmax_weights(skill, valid, temperature=1e6)
    assert np.allclose(w, 1 / 5, atol=1e-3)


def test_softmax_weights_zero_for_invalid_and_columns_sum_to_one():
    skill = np.array([[1.0, 2.0], [3.0, 4.0]])
    valid = np.array([[True, False], [True, True]])
    w = _softmax_weights(skill, valid, temperature=1.0)
    assert w[1, 0] == pytest.approx(1 - w[0, 0])  # column 0 splits over 2 valid
    assert w[0, 1] == 0.0                          # invalid entry -> zero
    assert w[1, 1] == pytest.approx(1.0)           # lone valid entry -> all mass


# --------------------------------------------------------------------------
# integration tests on small real data
# --------------------------------------------------------------------------

def _coords(geometry):
    return np.c_[geometry.x, geometry.y]


def test_invalid_params_raise(sample_regression_data):
    X, y, geometry = sample_regression_data
    coords = _coords(geometry)
    base = KDTreeEnsembleRegressor(min_leaf_size=10, pbar=False)
    with pytest.raises(ValueError):
        BaggedSpatialIndexEnsemble(base, weighting="bogus").fit(X, y, coords)
    with pytest.raises(ValueError):
        BaggedSpatialIndexEnsemble(base, reduction="bogus").fit(X, y, coords)


def test_empty_oob_leaf_warns(sample_regression_data):
    # small leaves + few members guarantee some leaf catches no out-of-bag row,
    # which must warn loudly (never degrade silently).
    X, y, geometry = sample_regression_data
    coords = _coords(geometry)
    base = KDTreeEnsembleRegressor(min_leaf_size=5, pbar=False)
    with pytest.warns(UserWarning, match="no out-of-bag"):
        BaggedSpatialIndexEnsemble(
            base, n_estimators=3, weighting="leaf", random_state=0
        ).fit(X, y, coords)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_uniform_softmax_equals_plain_mean(sample_regression_data):
    X, y, geometry = sample_regression_data
    coords = _coords(geometry)
    base = KDTreeEnsembleRegressor(min_leaf_size=10, pbar=False)
    bag = BaggedSpatialIndexEnsemble(
        base, n_estimators=5, weighting="uniform", reduction="softmax",
        random_state=0,
    ).fit(X, y, coords)

    pred = bag.predict(X, coords)
    manual = np.nanmean(
        np.vstack([m.predict(X, coords) for m in bag.members_]), axis=0
    )
    assert np.allclose(pred, manual, equal_nan=True)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_base_estimator_not_mutated(sample_regression_data):
    X, y, geometry = sample_regression_data
    coords = _coords(geometry)
    base = KDTreeEnsembleRegressor(min_leaf_size=10, pbar=False)
    BaggedSpatialIndexEnsemble(base, n_estimators=3, random_state=0).fit(
        X, y, coords
    )
    # the original template must stay unfitted (deepcopy isolation)
    assert not hasattr(base, "leaves_")


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_tree_weight_large_temperature_matches_uniform(sample_regression_data):
    X, y, geometry = sample_regression_data
    coords = _coords(geometry)
    base = KDTreeEnsembleRegressor(min_leaf_size=10, pbar=False)
    kwargs = dict(n_estimators=5, reduction="softmax", random_state=0)
    tree = BaggedSpatialIndexEnsemble(base, weighting="tree", **kwargs).fit(
        X, y, coords
    )
    uniform = BaggedSpatialIndexEnsemble(base, weighting="uniform", **kwargs).fit(
        X, y, coords
    )
    # same seed -> same members; huge temperature flattens the tree weights.
    assert np.allclose(
        tree.predict(X, coords, temperature=1e9),
        uniform.predict(X, coords),
        equal_nan=True,
    )


@pytest.mark.parametrize("weighting", ["uniform", "tree", "leaf"])
@pytest.mark.parametrize("reduction", ["softmax", "median"])
@pytest.mark.parametrize(
    "base_cls", [KDTreeEnsembleRegressor, QuadtreeEnsembleRegressor]
)
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_all_strategies_fit_and_predict(
    sample_regression_data, weighting, reduction, base_cls
):
    X, y, geometry = sample_regression_data
    coords = _coords(geometry)
    base = base_cls(min_leaf_size=10, pbar=False)
    bag = BaggedSpatialIndexEnsemble(
        base, n_estimators=5, weighting=weighting, reduction=reduction,
        random_state=0,
    ).fit(X, y, coords)

    pred = bag.predict(X, coords)
    assert pred.shape == (len(y),)
    assert np.isfinite(pred).all()          # every training point is routed
    assert np.isfinite(bag.score(X, y, coords))
    assert bag.n_leaves == sum(m.n_leaves for m in bag.members_)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_predict_temperature_override_changes_output(sample_regression_data):
    X, y, geometry = sample_regression_data
    coords = _coords(geometry)
    base = KDTreeEnsembleRegressor(min_leaf_size=10, pbar=False)
    bag = BaggedSpatialIndexEnsemble(
        base, n_estimators=8, weighting="leaf", reduction="softmax",
        temperature=1.0, random_state=1,
    ).fit(X, y, coords)
    hot = bag.predict(X, coords, temperature=1e9)   # ~uniform
    cold = bag.predict(X, coords, temperature=0.05)  # ~argmax
    assert not np.allclose(hot, cold, equal_nan=True)
