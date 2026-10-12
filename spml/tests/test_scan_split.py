import warnings

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor

from spml.ensemble import KDTreeBoostingRegressor, KDTreeEnsembleRegressor
from spml.ensemble._spatial_index_utils import (
    _neg_mean_squared_error,
    _ScanSplitNode,
)


def _regime_data(n=120, seed=0):
    """Two coefficient regimes separated at x = 0.3, plus noise."""
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0, 1, size=(n, 2)) * [2, 1]  # x is the longer axis
    X = rng.normal(size=(n, 2))
    slope = np.where(coords[:, 0] < 0.6, 1.0, -2.0)
    y = slope * X[:, 0] + 0.5 * X[:, 1] + rng.normal(scale=0.3, size=n)
    return X, y, coords


def _root(X, y, coords, model=None):
    return _ScanSplitNode(
        index=0,
        parent=None,
        parent_mask=np.ones(len(y), dtype=bool),
        coordinates=coords,
        bounds=[*coords.min(axis=0), *coords.max(axis=0)],
        X=X,
        y=y,
        ids=np.arange(len(y)),
        score_function=_neg_mean_squared_error,
        model=LinearRegression() if model is None else model,
    )


def _rss(X, y):
    return float(((y - LinearRegression().fit(X, y).predict(X)) ** 2).sum())


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_ols_scan_matches_brute_force(seed):
    """The exact path finds the admissible plane with the lowest summed RSS."""
    X, y, coords = _regime_data(seed=seed)
    min_leaf_size = 9
    left, right = _root(X, y, coords).split(min_leaf_size=min_leaf_size)
    scan_rss = _rss(left.X, left.y) + _rss(right.X, right.y)

    values = np.sort(coords[:, 0])
    brute = []
    for k in range(min_leaf_size + 1, len(y) - min_leaf_size):
        if values[k - 1] == values[k]:
            continue
        is_left = coords[:, 0] <= (values[k - 1] + values[k]) / 2
        brute.append(_rss(X[is_left], y[is_left]) + _rss(X[~is_left], y[~is_left]))

    np.testing.assert_allclose(scan_rss, min(brute), rtol=1e-8)


@pytest.mark.parametrize("min_leaf_size", [5, 20, 50])
def test_scan_children_clear_floor(min_leaf_size):
    X, y, coords = _regime_data()
    left, right = _root(X, y, coords).split(min_leaf_size=min_leaf_size)
    assert left.size > min_leaf_size
    assert right.size > min_leaf_size
    assert left.size + right.size == len(y)


def test_scan_keeps_ties_together():
    X, y, _ = _regime_data()
    rng = np.random.default_rng(3)
    coords = np.column_stack(
        [rng.integers(0, 6, size=len(y)) * 2.0, rng.uniform(0, 1, size=len(y))]
    )
    left, right = _root(X, y, coords).split(min_leaf_size=5)
    assert left.coordinates[:, 0].max() < right.coordinates[:, 0].min()


def test_scan_without_admissible_plane_keeps_root():
    X, y, coords = _regime_data(n=30)
    model = KDTreeEnsembleRegressor(
        split_method="scan", min_leaf_size=20, pbar=False
    ).fit(X, y, coords)
    assert model.n_leaves == 1


def test_exact_scan_does_not_warn():
    X, y, coords = _regime_data()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        KDTreeEnsembleRegressor(
            split_method="scan", min_leaf_size=10, pbar=False
        ).fit(X, y, coords)


@pytest.mark.parametrize(
    "model", [Ridge(), DecisionTreeRegressor(max_depth=2, random_state=0)]
)
def test_grid_scan_warns_and_clears_floor(model):
    X, y, coords = _regime_data()
    with pytest.warns(UserWarning, match="split_method='scan'"):
        fitted = KDTreeEnsembleRegressor(
            model=model, split_method="scan", min_leaf_size=10, pbar=False
        ).fit(X, y, coords)
    assert all(leaf.size > 10 for leaf in fitted.leaves_.values())


def test_scan_finds_regime_boundary():
    """With a sharp regime change, the first split lands near it."""
    X, y, coords = _regime_data(n=400)
    left, right = _root(X, y, coords).split(min_leaf_size=20)
    assert abs(left.bounds[2] - 0.6) < 0.05


def test_scan_resists_peeling():
    """A near-perfect fit at zero tolerance can't peel one point per level."""
    rng = np.random.default_rng(0)
    coords = rng.uniform(0, 1, size=(300, 2))
    X = rng.normal(size=(300, 2))
    y = X @ [1.0, -1.0]
    y[:3] += 5  # a few large residuals at arbitrary locations
    min_leaf_size = 10
    model = KDTreeEnsembleRegressor(
        split_method="scan", split_tolerance=0, min_leaf_size=min_leaf_size,
        pbar=False,
    ).fit(X, y, coords)
    assert all(leaf.size > min_leaf_size for leaf in model.leaves_.values())
    assert max(leaf.depth for leaf in model.leaves_.values()) <= 300 // (min_leaf_size + 1)


def test_boosting_scan_clears_floor():
    X, y, coords = _regime_data()
    model = KDTreeBoostingRegressor(
        split_method="scan", min_leaf_size=15, pbar=False
    ).fit(X, y, coords)
    assert all(leaf.size > 15 for leaf in model.leaves_.values())
