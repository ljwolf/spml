import numpy as np
import pytest

from spml.ensemble import (
    KDTreeBoostingRegressor,
    KDTreeEnsembleRegressor,
    QuadtreeEnsembleRegressor,
)
from spml.ensemble.pruning import permutation_prune


def _data(n=1500, regimes=True, seed=0):
    """A slope that flips sign at x = 0.5, or a constant slope."""
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0, 1, size=(n, 2))
    X = rng.normal(size=(n, 2))
    slope = np.where(coords[:, 0] < 0.5, 1.0, -1.0) if regimes else 1.0
    y = slope * X[:, 0] + 0.5 * X[:, 1] + rng.normal(scale=0.5, size=n)
    return X, y, coords


def _overgrown(cls, X, y, coords):
    return cls(split_tolerance=0, min_leaf_size=30, pbar=False).fit(X, y, coords)


@pytest.fixture(scope="module", params=[KDTreeEnsembleRegressor, QuadtreeEnsembleRegressor])
def regime_fit(request):
    X, y, coords = _data()
    model = _overgrown(request.param, X, y, coords)
    return X, y, coords, model, permutation_prune(model, n_permutations=99, random_state=0)


def test_input_is_unchanged(regime_fit):
    X, y, coords, model, pruned = regime_fit
    assert pruned is not model
    assert model.n_leaves == len(model.leaves_)
    assert not hasattr(model, "pruning_")
    assert pruned.n_leaves < model.n_leaves


def test_regime_split_survives(regime_fit):
    """The split that separates the two regimes is significant."""
    X, y, coords, model, pruned = regime_fit
    left = coords[:, 0] < 0.5
    share_left = [left[pruned.labels_ == leaf].mean() for leaf in pruned.leaves_]
    assert pruned.n_leaves >= 2
    # no leaf mixes the two regimes in substantial proportions
    assert all(share <= 0.1 or share >= 0.9 for share in share_left)


def test_pruned_tree_is_connected(regime_fit):
    """Every ancestor of a surviving leaf is an accepted, unpruned node."""
    X, y, coords, model, pruned = regime_fit
    table = pruned.pruning_.set_index("child")
    for index in pruned.leaves_:
        while index != 0:
            assert not table.loc[index, "pruned"]
            index = table.loc[index, "parent"]


def test_untested_children_have_surviving_descendants(regime_fit):
    X, y, coords, model, pruned = regime_fit
    table = pruned.pruning_
    survivors = set(table.loc[~table.pruned, "child"])
    for _, row in table[~table.tested].iterrows():
        assert any(table.loc[table.parent == row.child, "child"].isin(survivors))
        assert not row.pruned
    assert table.loc[table.tested, "p_value"].between(0, 1).all()
    assert table.loc[~table.tested, "p_value"].isna().all()


def test_labels_and_predictions_follow_leaves(regime_fit):
    X, y, coords, model, pruned = regime_fit
    expected = np.empty(len(y), dtype=object)
    predicted = np.empty(len(y))
    for leaf in sorted(pruned.leaves_.values(), key=lambda leaf: leaf.depth):
        expected[leaf.ids] = leaf.index
        predicted[leaf.ids] = leaf.model_.predict(X[leaf.ids])
    assert pruned.labels_.tolist() == expected.tolist()
    np.testing.assert_allclose(pruned.prediction_, predicted)
    assert set(pruned.geoms_.loc[~pruned.geoms_.pruned, "index"]) == set(pruned.leaves_)


def test_quadtree_predict_matches_fit():
    X, y, coords = _data()
    pruned = permutation_prune(
        _overgrown(QuadtreeEnsembleRegressor, X, y, coords),
        n_permutations=49, random_state=0,
    )
    assert pruned.query(coords).tolist() == pruned.labels_.tolist()
    np.testing.assert_allclose(pruned.predict(X, coords), pruned.prediction_)


@pytest.mark.parametrize("cls", [KDTreeEnsembleRegressor, QuadtreeEnsembleRegressor])
def test_constant_relationship_prunes_most_splits(cls):
    X, y, coords = _data(regimes=False)
    model = _overgrown(cls, X, y, coords)
    pruned = permutation_prune(model, n_permutations=99, random_state=0)
    assert model.n_leaves >= 10
    assert pruned.n_leaves <= model.n_leaves // 3


def test_random_state_is_reproducible():
    X, y, coords = _data(n=600)
    model = _overgrown(KDTreeEnsembleRegressor, X, y, coords)
    first = permutation_prune(model, n_permutations=29, random_state=1)
    second = permutation_prune(model, n_permutations=29, random_state=1)
    assert first.pruning_.equals(second.pruning_)


def test_rejects_unsupported_estimators():
    X, y, coords = _data(n=300)
    boosted = KDTreeBoostingRegressor(min_leaf_size=30, pbar=False).fit(X, y, coords)
    with pytest.raises(TypeError, match="quadtree and KD-tree"):
        permutation_prune(boosted)
    with pytest.raises(ValueError, match="fitted"):
        permutation_prune(KDTreeEnsembleRegressor())
    fitted = _overgrown(KDTreeEnsembleRegressor, X, y, coords)
    with pytest.raises(ValueError, match="n_permutations"):
        permutation_prune(fitted, n_permutations=0)
    with pytest.raises(ValueError, match="alpha"):
        permutation_prune(fitted, alpha=1)


def test_pruned_query_matches_fit_on_split_lines():
    """Points on a grid, many on split planes, route to the leaves the fit used."""
    grid = np.arange(17.0)
    coords = np.array([(x, y) for x in grid for y in grid for _ in range(3)])
    rng = np.random.default_rng(0)
    X = rng.normal(size=(len(coords), 2))
    slope = np.where(coords[:, 0] < 8, 1.0, -1.0)
    y = slope * X[:, 0] + 0.5 * X[:, 1] + rng.normal(scale=0.5, size=len(coords))
    model = KDTreeEnsembleRegressor(split_tolerance=0, min_leaf_size=20, pbar=False)
    pruned = permutation_prune(model.fit(X, y, coords), n_permutations=19, random_state=0)
    assert pruned.query(coords).tolist() == pruned.labels_.tolist()
    np.testing.assert_allclose(pruned.predict(X, coords), pruned.prediction_)
