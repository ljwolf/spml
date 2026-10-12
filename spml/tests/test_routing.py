import sys

import numpy as np
import pytest

from spml.ensemble import (
    KDTreeBoostingClassifier,
    KDTreeBoostingRegressor,
    KDTreeEnsembleRegressor,
    QuadtreeBoostingClassifier,
    QuadtreeBoostingRegressor,
    QuadtreeEnsembleRegressor,
)
from spml.ensemble import _spatial_index_utils


def _grid_data(seed=0):
    """Points on a 17 x 17 integer grid, so many lie exactly on split lines."""
    rng = np.random.default_rng(seed)
    grid = np.arange(17.0)
    coords = np.array([(x, y) for x in grid for y in grid for _ in range(3)])
    X = rng.normal(size=(len(coords), 2))
    slope = np.where(coords[:, 0] < 8, 1.0, -1.0)
    y = slope * X[:, 0] + 0.5 * X[:, 1] + rng.normal(scale=0.5, size=len(coords))
    return X, y, coords


def _contains_half_open(bounds, root_bounds, coords):
    """True where a point is inside bounds that exclude the lower edges, except on the root's edges."""
    min_x, min_y, max_x, max_y = bounds
    x, y = coords[:, 0], coords[:, 1]
    above_x = (x > min_x) | ((x == min_x) & (min_x == root_bounds[0]))
    above_y = (y > min_y) | ((y == min_y) & (min_y == root_bounds[1]))
    return above_x & above_y & (x <= max_x) & (y <= max_y)


@pytest.mark.parametrize(
    "cls",
    [KDTreeEnsembleRegressor, QuadtreeEnsembleRegressor,
     KDTreeBoostingRegressor, QuadtreeBoostingRegressor],
)
def test_query_matches_fit_on_split_lines(cls):
    X, y, coords = _grid_data()
    model = cls(split_tolerance=0, min_leaf_size=20, pbar=False).fit(X, y, coords)
    assert model.query(coords).tolist() == model.labels_.tolist()
    if cls in (KDTreeEnsembleRegressor, QuadtreeEnsembleRegressor):
        np.testing.assert_allclose(model.predict(X, coords), model.prediction_)


def test_query_outside_root_is_minus_one():
    X, y, coords = _grid_data()
    model = KDTreeEnsembleRegressor(min_leaf_size=20, pbar=False).fit(X, y, coords)
    assert model.query(np.array([[-1.0, 5.0], [5.0, 17.5]])).tolist() == [-1, -1]


@pytest.mark.parametrize("cls", [KDTreeBoostingClassifier, QuadtreeBoostingClassifier])
def test_boosting_classifier_routes_split_lines_low(cls, monkeypatch):
    """Each point routes to the leaf that holds it with lower edges excluded."""
    # KDTreeBoostingClassifier imports weighted_median from a module outside spml.
    monkeypatch.setitem(sys.modules, "kdtree_regression", _spatial_index_utils)
    X, y, coords = _grid_data()
    labels = (y > 0).astype(int)
    model = cls(split_tolerance=-np.inf, min_leaf_size=20, max_depth=4, pbar=False)
    model.fit(X, labels, coords)
    root = model.nodes_[0].bounds
    routed = model._query(coords)
    assert (routed != -1).all()
    for leaf in set(routed.tolist()):
        assert _contains_half_open(model.nodes_[leaf].bounds, root, coords[routed == leaf]).all()
