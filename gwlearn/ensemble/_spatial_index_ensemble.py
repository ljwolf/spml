"""
Spatial ensemble models using local models at spatial index leaves.

Each leaf in the spatial index has its own fitted local model.
Prediction routes each point to its deepest matching leaf and returns
that leaf's model prediction. Works with any scikit-learn compatible
estimator.
"""

import numpy, pandas, geopandas, shapely
from collections import defaultdict
from sklearn import linear_model, metrics
from sklearn.base import clone

from ._spatial_index_utils import (
    _Cell,
    _Node,
    _WeightedMedianSplitNode,
    _RandomSplitNode,
    _OptimalSplitNode,
)
from ._spatial_tree_ensemble import _holds_out, _leaf_split


def _neg_mean_squared_error(y_true, y_pred):
    """Default split score. Module-level (not a closure) so fitted ensembles pickle."""
    return -metrics.mean_squared_error(y_true, y_pred)


class SpatialIndexEnsemble:
    """
    Base class for spatial ensemble models.

    Builds a spatial index, fits a local model at each leaf, and predicts by
    routing each point to its deepest matching leaf. A split is accepted when
    the local model in the child cell improves upon the parent model's score
    on the same data by at least `split_tolerance`.

    Parameters
    ----------
    model : sklearn estimator (default: LinearRegression)
        Cloned for each node. Must implement fit and predict.
    score : callable (default: negative MSE)
        score(y_true, y_pred) -> scalar; higher is better.
    split_tolerance : float (default: 1e-5)
        Minimum improvement in score to accept a split.
    min_leaf_size : int (default: 10)
        Smallest allowed leaf.
    max_depth : int or None (default: None)
        Maximum depth of the spatial index.  None leaves growth governed only by
        `split_tolerance` and `min_leaf_size`.
    max_leaves : int or float (default: ``inf``)
        Cap on the total number of leaves.  ``inf`` leaves growth governed only by
        `split_tolerance` and `min_leaf_size`.  When finite, the cap is a ceiling
        enforced in the index's depth-first growth order.
    validation_fraction : float (default: 0.0)
        Proportion of each child leaf held out when `split_tolerance` is
        applied. With ``0.0``, the `split_tolerance` is applied in sample (a
        child is scored on its own training rows). With a value in
        ``(0.0, 1.0)``, the child's model is refit on the remaining rows and
        compared with the parent on the held-out rows, so the split must
        improve out of sample. The split
        location is always chosen in-sample; this only changes accept/reject.
        The leaf's stored model is always fit on the full leaf for prediction.
        A child that can't hold out this fraction of its rows (at least one
        row out and one row kept) counts as below `min_leaf_size`.
    random_state : int, numpy.random.Generator or None (default: None)
        Seeds the per-leaf holdout split when ``validation_fraction > 0``.
    pbar : bool (default: True)
        Show a progress bar during fitting.
    bounds : array-like or None
        [min_x, min_y, max_x, max_y] for the root cell. Derived from
        coordinates if None.
    """

    def __init__(
        self,
        model=None,
        score=None,
        split_tolerance=1e-5,
        min_leaf_size=10,
        max_depth=None,
        max_leaves=float("inf"),
        validation_fraction=0.0,
        random_state=None,
        pbar=True,
        bounds=None,
    ):
        if model is None:
            model = linear_model.LinearRegression()
        self.model = model
        if score is None:
            score = _neg_mean_squared_error
        self._score_function = score
        self.split_tolerance = split_tolerance
        self.min_leaf_size = min_leaf_size
        self.max_depth = max_depth
        self.max_leaves = max_leaves
        self.validation_fraction = validation_fraction
        self.random_state = random_state
        self.pbar = pbar
        self.bounds = bounds

    def _build_node(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement _build_node.")

    def _candidate_splits(self, cell):
        """
        Split a cell and register children in levels_. Returns all candidate
        children regardless of quality; filtering happens in _evaluate_splits.
        """
        splits = cell.split()
        self.levels_[cell.depth + 1].extend(splits)
        return splits

    def _split_gain(self, cell, split):
        """
        Improvement of the child's local model over the parent on the scored rows.

        In-sample by default. When ``validation_fraction > 0`` the child model is
        refit on a training slice and both models are scored on the held-out
        slice, so the gain reflects out-of-sample improvement.
        """
        if self.validation_fraction > 0:
            if split.model_ is None:
                return -numpy.inf
            fit_pos, val_pos = _leaf_split(
                numpy.arange(split.size), self.validation_fraction, self._rng
            )
            child = clone(split._model).fit(split.X[fit_pos], split.y[fit_pos])
            child_score = split._score_function(
                split.y[val_pos], child.predict(split.X[val_pos])
            )
            parent_score = split._score_function(
                split.y[val_pos], cell.prediction_[split.parent_mask][val_pos]
            )
            return child_score - parent_score
        parent_score = cell._score_function(
            cell.y[split.parent_mask], cell.prediction_[split.parent_mask]
        )
        return split.score_ - parent_score

    def _below_floor(self, split):
        """
        True if a child violates the depth or size floor; false otherwise.

        A child that can't hold out `validation_fraction` of its rows is below
        the size floor.
        """
        return (
            (self.max_depth is not None and split.depth > self.max_depth)
            or split.size <= self.min_leaf_size
            or not _holds_out(split.size, self.validation_fraction)
        )

    def _evaluate_splits(self, cell):
        """
        Evaluate each candidate split. A split is retained when the child's
        local model improves on the parent's predictions over the same subset
        by at least split_tolerance.

        Children below the depth/size floor are not expanded; the parent stays
        as a space-covering fallback leaf so those points are still predicted.
        This per-child rule is correct for geometric (quadtree) subdivision,
        where sparse quadrants are legitimate small leaves. Binary KD-tree
        splits need the stricter both-children rule (see KDTreeEnsemble) to
        avoid a degenerate plane peeling one point per level.
        """
        retained = []
        for split in self._candidate_splits(cell):
            if self._below_floor(split):
                self._update_leaf(cell)
                continue
            if self._split_gain(cell, split) >= self.split_tolerance:
                retained.append(split)
            else:
                self._update_leaf(cell)
        return retained

    def fit(self, X, y, coordinates):
        """
        Fit the spatial ensemble.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        coordinates : array-like of shape (n_samples, 2)

        Returns
        -------
        self
        """
        if isinstance(X, pandas.DataFrame):
            X = X.values
        if isinstance(y, (pandas.Series, pandas.DataFrame)):
            y = y.values.squeeze()

        coordinates = numpy.asarray(coordinates)
        self._rng = numpy.random.default_rng(self.random_state)

        if self.bounds is None:
            self.bounds = [*numpy.min(coordinates, axis=0), *numpy.max(coordinates, axis=0)]

        root = self._build_node(
            index=0,
            parent=None,
            parent_mask=numpy.ones(len(y), dtype=bool),
            coordinates=coordinates,
            bounds=self.bounds,
            ids=numpy.arange(len(X)),
            X=X,
            y=y,
            score_function=self._score_function,
            model=clone(self.model),
        )

        self.levels_ = defaultdict(list)
        self.levels_[0].append(root)
        self.labels_ = numpy.full(len(y), 0, dtype=object)
        self.prediction_ = root.prediction_.copy()
        self.leaves_ = {}

        if self.pbar:
            from tqdm import tqdm
            self._pbar = tqdm(unit=" leaves", desc="Evaluating splits...")

        queue = [root]
        while queue:
            cell = queue.pop()
            # `leaves_` holds finalized leaves; every cell still queued (plus the
            # one in hand) becomes at least one more leaf, so this is the leaf
            # count if growth stopped now.  Once it reaches the budget, stop
            # expanding and finalize the rest in place.
            if len(self.leaves_) + len(queue) + 1 >= self.max_leaves:
                self._update_leaf(cell)
                continue
            queue.extend(self._evaluate_splits(cell))

        if self.pbar:
            self._pbar.close()

        all_cells = [c for level in self.levels_.values() for c in level]
        self.geoms_ = geopandas.GeoDataFrame(
            pandas.DataFrame.from_records([
                dict(
                    index=c.index,
                    parent=c.parent,
                    depth=c.depth,
                    score=c.score_,
                    pruned=c.index not in self.leaves_,
                    geometry=shapely.box(*c.bounds),
                )
                for c in all_cells
            ])
        ).sort_values("depth")

        return self

    def predict(self, X, coordinates, extrapolate=False):
        """
        Predict by routing each point to its leaf and using that leaf's model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        coordinates : array-like of shape (n_samples, 2)
        extrapolate : bool (default: False)
            Use the root model for points outside the tree bounds.

        Returns
        -------
        numpy.ndarray of shape (n_samples,)
        """
        if isinstance(X, pandas.DataFrame):
            X = X.values
        leaf_labels = self.query(coordinates)
        output = numpy.full(len(X), numpy.nan)
        for leaf_name, leaf in self.leaves_.items():
            mask = leaf_labels == leaf_name
            if mask.any() and leaf.model_ is not None:
                output[mask] = leaf.model_.predict(X[mask])
        if extrapolate:
            outside = leaf_labels == -1
            if outside.any():
                output[outside] = self.levels_[0][0].model_.predict(X[outside])
        return output

    def score(self, X, y, coordinates):
        """Score on held-out data using self._score_function."""
        return self._score_function(y, self.predict(X, coordinates))

    def query(self, coordinates):
        """
        Return the leaf index for each point (deepest matching leaf, -1 if
        outside bounds).
        """
        coordinates = numpy.asarray(coordinates)
        leaf_names = list(self.leaves_.keys())
        depthsort = numpy.argsort(
            [leaf.depth for leaf in self.leaves_.values()]
        )[::-1]
        leaves_by_depth = {
            leaf_names[i]: self.leaves_[leaf_names[i]] for i in depthsort
        }
        query_result = pandas.concat(
            [
                pandas.Series(
                    pandas.arrays.SparseArray(leaf.query(coordinates)),
                    name=leaf.index,
                )
                for leaf in leaves_by_depth.values()
            ],
            axis=1,
        )
        query_result.loc[:, -1] = True
        return query_result.columns[query_result.values.argmax(axis=1)].values

    def _update_leaf(self, cell):
        self.labels_[cell.ids] = cell.index
        self.prediction_[cell.ids] = cell.prediction_
        self.leaves_[cell.index] = cell
        if self.pbar:
            self._pbar.update(1)

    @property
    def n_leaves(self):
        return len(self.leaves_)


class QuadtreeEnsemble(SpatialIndexEnsemble):
    """
    Spatial ensemble using a quadtree index.
    Splits each cell into 4 children at geographic midpoints.
    """

    def _build_node(self, *args, **kwargs):
        return _Cell(*args, **kwargs)


class KDTreeEnsemble(SpatialIndexEnsemble):
    """
    Spatial ensemble using a KD-tree index.
    Splits each cell into 2 children along the longest axis.

    Parameters
    ----------
    split_method : str or int/float (default: 'weighted')
        How to choose the split plane.
        - 'median': split at the coordinate median
        - 'weighted': split at the residual-weighted median
        - 'optimal': minimize loss over the split location
        - int/float: split at the median of a random subsample of this size
    """

    def __init__(self, *, split_method="weighted", **kwargs):
        super().__init__(**kwargs)
        self.split_method = split_method

    def _build_node(self, *args, **kwargs):
        if self.split_method == "median":
            return _Node(*args, **kwargs)
        elif self.split_method == "weighted":
            return _WeightedMedianSplitNode(*args, **kwargs)
        elif isinstance(self.split_method, (float, int)):
            return _RandomSplitNode(*args, **kwargs)
        elif self.split_method == "optimal":
            return _OptimalSplitNode(*args, **kwargs)
        else:
            raise ValueError(f"Unknown split_method: {self.split_method!r}")

    def _candidate_splits(self, cell):
        if self.split_method == "optimal":
            splits = cell.split(min_leaf_size=self.min_leaf_size)
        else:
            splits = cell.split()
        self.levels_[cell.depth + 1].extend(splits)
        return splits

    def _evaluate_splits(self, cell):
        """
        A binary KD-tree split is admissible only if *both* children clear the
        depth and size floors. Evaluating children independently let a
        degenerate split (e.g. a residual-weighted median driven onto the
        boundary when residuals vanish) peel a sub-floor sliver off one side
        and retain the almost-full other child, so the support shrank by one
        point per level and depth ran away until the base-10 path-code index
        overflowed float64. Rejecting the whole split bounds depth and matches
        min_samples_leaf semantics.
        """
        splits = self._candidate_splits(cell)
        if any(self._below_floor(split) for split in splits):
            self._update_leaf(cell)
            return []
        retained = []
        for split in splits:
            if self._split_gain(cell, split) >= self.split_tolerance:
                retained.append(split)
            else:
                self._update_leaf(cell)
        return retained


# ---------------------------------------------------------------------------
# Explicit regressor / classifier ensembles
# ---------------------------------------------------------------------------

class _EnsembleClassifierMixin:
    """
    Turns a SpatialIndexEnsemble into a classifier: each leaf fits a local
    classifier and prediction routes every point to its deepest matching leaf.

    Defaults the per-node model to LogisticRegression and the split score to
    accuracy.  Adds predict_proba, aligning each leaf model's classes_ to the
    global class set so leaves that never saw a class still produce a full
    probability row.  Leaves whose local fit was degenerate (e.g. a single
    class) fall back to predicting that leaf's majority class.
    """

    def __init__(self, model=None, score=None, **kwargs):
        if model is None:
            model = linear_model.LogisticRegression()
        if score is None:
            score = metrics.accuracy_score
        super().__init__(model=model, score=score, **kwargs)

    def fit(self, X, y, coordinates):
        y_arr = (
            y.values if isinstance(y, (pandas.Series, pandas.DataFrame))
            else numpy.asarray(y)
        )
        self.classes_ = numpy.unique(y_arr)
        return super().fit(X, y, coordinates)

    def predict_proba(self, X, coordinates, extrapolate=False):
        """Return class probabilities of shape (n_samples, n_classes)."""
        if isinstance(X, pandas.DataFrame):
            X = X.values
        leaf_labels = self.query(coordinates)
        proba = numpy.zeros((len(X), len(self.classes_)))
        col = {c: i for i, c in enumerate(self.classes_)}

        def _fill(model_, mask):
            rows = numpy.flatnonzero(mask)
            local = model_.predict_proba(X[mask])
            for j, c in enumerate(model_.classes_):
                proba[rows, col[c]] = local[:, j]

        for leaf_name, leaf in self.leaves_.items():
            mask = leaf_labels == leaf_name
            if not mask.any():
                continue
            if leaf.model_ is not None:
                _fill(leaf.model_, mask)
            elif len(leaf.y) > 0:
                vals, counts = numpy.unique(leaf.y, return_counts=True)
                proba[numpy.flatnonzero(mask), col[vals[counts.argmax()]]] = 1.0

        if extrapolate:
            outside = leaf_labels == -1
            root = self.levels_[0][0]
            if outside.any() and root.model_ is not None:
                _fill(root.model_, outside)
        return proba

    def predict(self, X, coordinates, extrapolate=False):
        proba = self.predict_proba(X, coordinates, extrapolate=extrapolate)
        out = self.classes_[proba.argmax(axis=1)]
        unrouted = ~proba.any(axis=1)
        if unrouted.any():
            out = out.astype(object)
            out[unrouted] = None
        return out

    def score(self, X, y, coordinates):
        return self._score_function(numpy.asarray(y), self.predict(X, coordinates))


class QuadtreeEnsembleRegressor(QuadtreeEnsemble):
    """Quadtree spatial ensemble for regression (one local regressor per leaf)."""


class QuadtreeEnsembleClassifier(_EnsembleClassifierMixin, QuadtreeEnsemble):
    """Quadtree spatial ensemble for classification (one local classifier per leaf)."""


class KDTreeEnsembleRegressor(KDTreeEnsemble):
    """KD-tree spatial ensemble for regression (one local regressor per leaf)."""


class KDTreeEnsembleClassifier(_EnsembleClassifierMixin, KDTreeEnsemble):
    """KD-tree spatial ensemble for classification (one local classifier per leaf)."""
