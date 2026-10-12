"""
Spatial boosting models.

Two families live here:

* Leaf-model boosting over a supervised spatial index
  (SpatialIndexBoosting, QuadtreeBoosting, KDTreeBoosting): the root fits a
  global model on raw y and every accepted split fits a local model on the
  residuals left after summing all ancestor predictions.

* Tree boosting over an externally provided tree
  (SpatialTreeBoosting): residual boosting across the depths of any
  ExternalTreeTransformer.

All future boosting work should be added to this module.
"""

import numpy
import pandas
from collections import defaultdict

from ._spatial_index_ensemble import (
    SpatialIndexEnsemble,
    QuadtreeEnsemble,
    KDTreeEnsemble,
)


def _softmax(F):
    """Row-wise softmax of a margin matrix of shape (n, n_classes)."""
    F = F - F.max(axis=1, keepdims=True)
    e = numpy.exp(F)
    return e / e.sum(axis=1, keepdims=True)


class SpatialIndexBoosting(SpatialIndexEnsemble):
    """
    Spatial boosting: additive model where each depth corrects residuals
    from all shallower levels.

    The root fits a global model on raw y.  Every accepted split fits a
    local model on the residuals left after summing all ancestor predictions.
    At prediction time, contributions from every node on the path from root
    to the deepest matching leaf are summed.

    The split criterion is the same epsilon-improvement test as the ensemble:
    a split is accepted when the *cumulative* model at the child level beats
    the cumulative model at the parent level by at least `split_tolerance`.

    Parameters are identical to SpatialIndexEnsemble.
    """

    def _candidate_splits(self, cell):
        splits = cell.split(boost=True)
        self.levels_[cell.depth + 1].extend(splits)
        return splits

    def predict(self, X, coordinates, extrapolate=False):
        """
        Predict by summing each node's model contribution along the path
        from root to the deepest matching leaf.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        coordinates : array-like of shape (n_samples, 2)
        extrapolate : bool (default: False)
            Use the root model for out-of-bounds points.

        Returns
        -------
        numpy.ndarray of shape (n_samples,)
        """
        if isinstance(X, pandas.DataFrame):
            X = X.values
        leaf_labels = self.query(coordinates)

        # Index every node (not just leaves) for parent-chain traversal
        all_nodes = {
            n.index: n
            for level in self.levels_.values()
            for n in level
        }

        # Precompute root→leaf ancestor paths for each unique leaf
        paths = {}
        for leaf_idx in numpy.unique(leaf_labels):
            if leaf_idx == -1:
                continue
            path = []
            node_idx = leaf_idx
            while node_idx is not None:
                node = all_nodes.get(node_idx)
                if node is None:
                    break
                if node.model_ is not None:
                    path.append(node)
                node_idx = node.parent
            paths[leaf_idx] = path

        output = numpy.zeros(len(X))
        for leaf_idx, path in paths.items():
            mask = leaf_labels == leaf_idx
            for node in path:
                output[mask] += node.model_.predict(X[mask])

        if not extrapolate:
            output[leaf_labels == -1] = numpy.nan
        else:
            outside = leaf_labels == -1
            if outside.any():
                root = self.levels_[0][0]
                if root.model_ is not None:
                    output[outside] = root.model_.predict(X[outside])
        return output


class QuadtreeBoostingRegressor(SpatialIndexBoosting, QuadtreeEnsemble):
    """
    Spatial boosting using a quadtree index.
    Each cell is split into 4 children at geographic midpoints.
    See SpatialIndexBoosting for the boosting description.
    """


class KDTreeBoostingRegressor(SpatialIndexBoosting, KDTreeEnsemble):
    """
    Spatial boosting using a KD-tree index.
    Each cell is split into 2 children along the longest axis.
    See SpatialIndexBoosting and KDTreeEnsemble for parameter details.
    """

    def _candidate_splits(self, cell):
        if self.split_method in ("optimal", "scan"):
            splits = cell.split(boost=True, min_leaf_size=self.min_leaf_size)
        else:
            splits = cell.split(boost=True)
        self.levels_[cell.depth + 1].extend(splits)
        return splits


class SpatialTreeBoostingRegressor:
    """
    Spatial boosting backed by any ExternalTreeTransformer.

    At depths 0, 1, …, max_depth the data are partitioned into pages and a
    local model is fit on the residuals remaining after all shallower depths.
    Prediction sums contributions from every depth.

    For transformers without depth support (``ancestor_fn=None``), set
    ``max_depth=None`` to use the leaf partition as a single boosting stage
    (equivalent to a local-regression ensemble).

    Parameters
    ----------
    transformer : ExternalTreeTransformer
        A fitted transformer.  Must support depth queries when ``max_depth``
        is not None.
    model : sklearn estimator (default: LinearRegression)
        Cloned for each page at each depth.
    max_depth : int or None (default: None)
        None  → single stage using the leaf partition.
        int d → iterate depths 0, 1, …, d (requires depth support in the
                transformer).
    min_leaf_size : int (default: 10)
        Minimum page population to fit a local model; smaller pages are
        skipped and their residuals are left for deeper levels.
    """

    def __init__(self, transformer, model=None, max_depth=None,
                 min_leaf_size=10):
        self.transformer   = transformer
        self.model         = model
        self.max_depth     = max_depth
        self.min_leaf_size = min_leaf_size

    def fit(self, X, y, coordinates):
        from sklearn.base import clone
        from sklearn.linear_model import LinearRegression

        X           = numpy.asarray(X)
        y           = numpy.asarray(y, dtype=float)
        coordinates = numpy.asarray(coordinates)

        model    = self.model if self.model is not None else LinearRegression()
        depths   = [None] if self.max_depth is None else range(self.max_depth + 1)
        residuals = y.copy()

        self.models_by_depth_ = {}

        for depth in depths:
            enc        = self.transformer.transform(coordinates, depth=depth)
            page_labels = numpy.asarray(enc.argmax(axis=1)).ravel()
            in_page    = numpy.asarray(enc.sum(axis=1) > 0).ravel()

            page_models = {}
            for page in numpy.unique(page_labels[in_page]):
                mask = in_page & (page_labels == page)
                if mask.sum() < self.min_leaf_size:
                    continue
                m = clone(model)
                m.fit(X[mask], residuals[mask])
                page_models[page] = m
                residuals[mask] -= m.predict(X[mask])

            if page_models:
                self.models_by_depth_[depth] = page_models

        return self

    def predict(self, X, coordinates):
        X           = numpy.asarray(X)
        coordinates = numpy.asarray(coordinates)
        output      = numpy.zeros(len(X))

        for depth, page_models in self.models_by_depth_.items():
            enc        = self.transformer.transform(coordinates, depth=depth)
            page_labels = numpy.asarray(enc.argmax(axis=1)).ravel()
            in_page    = numpy.asarray(enc.sum(axis=1) > 0).ravel()

            for page, model in page_models.items():
                mask = in_page & (page_labels == page)
                if mask.any():
                    output[mask] += model.predict(X[mask])

        return output

    def score(self, X, y, coordinates):
        from sklearn.metrics import mean_squared_error
        return -mean_squared_error(numpy.asarray(y),
                                   self.predict(X, coordinates))


class SpatialTreeBoostingClassifier:
    """
    Gradient-boosting classifier backed by any ExternalTreeTransformer.

    A stagewise additive model in margin (logit) space.  At depths 0, 1, …,
    max_depth the data are partitioned into pages; within each page one
    regressor per class is fit to the current pseudo-residuals -- the negative
    gradient of the multinomial deviance, ``one_hot(y) - softmax(F)``.  Page
    margins accumulate across depths and a softmax link maps the summed margins
    to probabilities (the two-class case reduces to the logistic link).

    The per-page learner is a *regressor* (it fits continuous pseudo-residuals),
    not a classifier.

    Parameters
    ----------
    transformer : ExternalTreeTransformer
        A fitted transformer.  Must support depth queries when ``max_depth``
        is not None.
    model : sklearn regressor (default: LinearRegression)
        Cloned for each class within each page at each depth.
    max_depth : int or None (default: None)
        None  → single stage using the leaf partition.
        int d → iterate depths 0, 1, …, d.
    min_leaf_size : int (default: 10)
        Minimum page population to fit; smaller pages are left to deeper levels.
    learning_rate : float (default: 1.0)
        Shrinkage applied to each stage's margin contribution.
    """

    def __init__(self, transformer, model=None, max_depth=None,
                 min_leaf_size=10, learning_rate=1.0):
        self.transformer   = transformer
        self.model         = model
        self.max_depth     = max_depth
        self.min_leaf_size = min_leaf_size
        self.learning_rate = learning_rate

    def _pages(self, coordinates, depth):
        enc         = self.transformer.transform(coordinates, depth=depth)
        page_labels = numpy.asarray(enc.argmax(axis=1)).ravel()
        in_page     = numpy.asarray(enc.sum(axis=1) > 0).ravel()
        return page_labels, in_page

    def fit(self, X, y, coordinates):
        from sklearn.base import clone
        from sklearn.linear_model import LinearRegression

        X           = numpy.asarray(X)
        y           = numpy.asarray(y)
        coordinates = numpy.asarray(coordinates)

        self.classes_ = numpy.unique(y)
        K   = len(self.classes_)
        col = {c: i for i, c in enumerate(self.classes_)}
        Y   = numpy.zeros((len(y), K))
        Y[numpy.arange(len(y)), [col[v] for v in y]] = 1.0

        model  = self.model if self.model is not None else LinearRegression()
        depths = [None] if self.max_depth is None else range(self.max_depth + 1)

        priors            = numpy.clip(Y.mean(axis=0), 1e-12, None)
        self.init_margin_ = numpy.log(priors)
        F = numpy.tile(self.init_margin_, (len(y), 1))

        self.models_by_depth_ = {}
        for depth in depths:
            page_labels, in_page = self._pages(coordinates, depth)
            residuals = Y - _softmax(F)

            page_models = {}
            for page in numpy.unique(page_labels[in_page]):
                mask = in_page & (page_labels == page)
                if mask.sum() < self.min_leaf_size:
                    continue
                class_models = {}
                for k in range(K):
                    m = clone(model)
                    m.fit(X[mask], residuals[mask, k])
                    F[mask, k] += self.learning_rate * m.predict(X[mask])
                    class_models[k] = m
                page_models[page] = class_models

            if page_models:
                self.models_by_depth_[depth] = page_models

        return self

    def decision_function(self, X, coordinates):
        X           = numpy.asarray(X)
        coordinates = numpy.asarray(coordinates)
        F = numpy.tile(self.init_margin_, (len(X), 1))

        for depth, page_models in self.models_by_depth_.items():
            page_labels, in_page = self._pages(coordinates, depth)
            for page, class_models in page_models.items():
                mask = in_page & (page_labels == page)
                if not mask.any():
                    continue
                for k, m in class_models.items():
                    F[mask, k] += self.learning_rate * m.predict(X[mask])
        return F

    def predict_proba(self, X, coordinates):
        return _softmax(self.decision_function(X, coordinates))

    def predict(self, X, coordinates):
        proba = self.predict_proba(X, coordinates)
        return self.classes_[proba.argmax(axis=1)]

    def score(self, X, y, coordinates):
        from sklearn.metrics import accuracy_score
        return accuracy_score(numpy.asarray(y), self.predict(X, coordinates))


# ---------------------------------------------------------------------------
# Supervised-index gradient-boosting classifiers
# ---------------------------------------------------------------------------

class _GBNode:
    """One node of a supervised-index gradient-boosting classifier.

    Holds this node's own margin contribution (one regressor per class fit to
    the page's pseudo-residuals).  Total margin for a point is the prior margin
    plus the contribution of every node on its root-to-leaf path.
    """

    def __init__(self, index, parent, depth, bounds):
        self.index   = index
        self.parent  = parent
        self.depth   = depth
        self.bounds  = bounds
        self.models_ = None

    def query(self, coordinates):
        x, y = coordinates.T
        min_x, min_y, max_x, max_y = self.bounds
        return (min_x <= x) & (x <= max_x) & (min_y <= y) & (y <= max_y)


class SpatialIndexBoostingClassifier:
    """
    Supervised-index gradient-boosting classifier.

    Builds a spatial index whose root fits one regressor per class on the
    pseudo-residuals of the class priors; every accepted split fits a further
    per-class regressor on the residuals remaining after summing all ancestor
    margins.  A split is accepted when the child's cumulative margins improve the
    mean multinomial log-likelihood over the same subset by at least
    ``split_tolerance``.  Prediction sums the per-class margin contributions of
    every node on the path from root to the deepest matching leaf and applies a
    softmax link.

    Parameters
    ----------
    model : sklearn regressor (default: LinearRegression)
        Cloned per class at each node; fits continuous pseudo-residuals.
    learning_rate : float (default: 1.0)
        Shrinkage applied to each node's margin contribution.
    split_tolerance : float (default: 1e-5)
        Minimum mean-log-likelihood improvement to accept a split.
    min_leaf_size : int (default: 10)
        Smallest allowed leaf.
    max_depth : int (default: 10)
        Maximum depth of the spatial index.
    pbar : bool (default: True)
        Show a progress bar during fitting.
    bounds : array-like or None
        [min_x, min_y, max_x, max_y] for the root cell; derived from
        coordinates if None.
    """

    def __init__(self, model=None, learning_rate=1.0, split_tolerance=1e-5,
                 min_leaf_size=10, max_depth=10, pbar=True, bounds=None):
        self.model           = model
        self.learning_rate   = learning_rate
        self.split_tolerance = split_tolerance
        self.min_leaf_size   = min_leaf_size
        self.max_depth       = max_depth
        self.pbar            = pbar
        self.bounds          = bounds

    def _children(self, coords, bounds, weights):
        """Return [(child_bounds, local_mask), ...] using the index geometry."""
        raise NotImplementedError("Subclasses must implement _children.")

    @staticmethod
    def _metric(Y, P):
        """Mean multinomial log-likelihood (higher is better)."""
        P = numpy.clip(P, 1e-12, 1.0)
        return float((Y * numpy.log(P)).sum() / len(P))

    def _fit_regressors(self, Xc, Yc, Fc, base_model, clone):
        residuals = Yc - _softmax(Fc)
        models    = {}
        contrib   = numpy.zeros_like(Fc)
        for k in range(Yc.shape[1]):
            m = clone(base_model)
            m.fit(Xc, residuals[:, k])
            models[k]     = m
            contrib[:, k] = m.predict(Xc)
        return models, contrib

    def fit(self, X, y, coordinates):
        from sklearn.base import clone
        from sklearn.linear_model import LinearRegression

        X           = numpy.asarray(X)
        y           = numpy.asarray(y)
        coordinates = numpy.asarray(coordinates)

        if self.bounds is None:
            self.bounds = [*numpy.min(coordinates, axis=0),
                           *numpy.max(coordinates, axis=0)]

        self.classes_ = numpy.unique(y)
        K   = len(self.classes_)
        col = {c: i for i, c in enumerate(self.classes_)}
        Y   = numpy.zeros((len(y), K))
        Y[numpy.arange(len(y)), [col[v] for v in y]] = 1.0
        self.n_classes_ = K

        base_model        = self.model if self.model is not None else LinearRegression()
        priors            = numpy.clip(Y.mean(axis=0), 1e-12, None)
        self.init_margin_ = numpy.log(priors)
        F = numpy.tile(self.init_margin_, (len(y), 1))

        self.nodes_  = {}
        self.leaves_ = {}

        if self.pbar:
            from tqdm import tqdm
            self._pbar = tqdm(unit=" leaves", desc="Boosting splits...")

        root     = _GBNode(0, None, 0, self.bounds)
        root_ids = numpy.arange(len(y))
        root.models_, contrib = self._fit_regressors(
            X[root_ids], Y[root_ids], F[root_ids], base_model, clone
        )
        F[root_ids] += self.learning_rate * contrib
        self.nodes_[0] = root

        queue = [(0, root_ids)]
        while queue:
            index, ids = queue.pop()
            queue.extend(
                self._evaluate_splits(index, ids, X, Y, F, coordinates,
                                      base_model, clone)
            )

        if self.pbar:
            self._pbar.close()
        return self

    def _evaluate_splits(self, index, ids, X, Y, F, coordinates, base_model, clone):
        node = self.nodes_[index]
        if node.depth >= self.max_depth or len(ids) <= self.min_leaf_size:
            self._register_leaf(node)
            return []

        P       = _softmax(F[ids])
        weights = numpy.abs(Y[ids] - P).sum(axis=1)
        coords  = coordinates[ids]

        retained = []
        any_rejected = False
        for i, (child_bounds, local_mask) in enumerate(
            self._children(coords, node.bounds, weights)
        ):
            child_ids = ids[local_mask]
            if child_ids.size <= self.min_leaf_size:
                any_rejected = True
                continue

            parent_score = self._metric(Y[child_ids], _softmax(F[child_ids]))
            models, contrib = self._fit_regressors(
                X[child_ids], Y[child_ids], F[child_ids], base_model, clone
            )
            child_F     = F[child_ids] + self.learning_rate * contrib
            child_score = self._metric(Y[child_ids], _softmax(child_F))

            if child_score - parent_score >= self.split_tolerance:
                F[child_ids] = child_F
                child = _GBNode(index * 10 + (i + 1), index,
                                node.depth + 1, child_bounds)
                child.models_ = models
                self.nodes_[child.index] = child
                retained.append((child.index, child_ids))
            else:
                any_rejected = True

        if any_rejected or not retained:
            self._register_leaf(node)
        return retained

    def _register_leaf(self, node):
        if node.index not in self.leaves_:
            self.leaves_[node.index] = node
            if self.pbar:
                self._pbar.update(1)

    def _query(self, coordinates):
        """
        Returns the leaf index for each point, or -1 outside the root.

        Each point descends from the root by the rule in `_children`, so a
        point on a split line or plane gets the same leaf as a training row
        at that location.
        """
        coordinates = numpy.asarray(coordinates)
        children = defaultdict(dict)
        for node in self.nodes_.values():
            if node.parent is not None:
                children[node.parent][node.index - node.parent * 10 - 1] = node

        root = self.nodes_[0]
        out = numpy.full(len(coordinates), -1, dtype=object)
        stack = [(root, numpy.flatnonzero(root.query(coordinates)))]
        while stack:
            node, rows = stack.pop()
            if node.index in self.leaves_:
                out[rows] = node.index
            if not children[node.index] or rows.size == 0:
                continue
            positions = self._child_positions(
                node.bounds, coordinates[rows], children[node.index]
            )
            for position, child in children[node.index].items():
                stack.append((child, rows[positions == position]))
        return out

    def decision_function(self, X, coordinates):
        X           = numpy.asarray(X)
        coordinates = numpy.asarray(coordinates)
        leaf_labels = self._query(coordinates)
        F = numpy.tile(self.init_margin_, (len(X), 1))

        for leaf_idx in numpy.unique(leaf_labels):
            if leaf_idx == -1:
                continue
            mask     = leaf_labels == leaf_idx
            node_idx = leaf_idx
            while node_idx is not None:
                node = self.nodes_.get(node_idx)
                if node is None:
                    break
                if node.models_ is not None:
                    for k, m in node.models_.items():
                        F[mask, k] += self.learning_rate * m.predict(X[mask])
                node_idx = node.parent
        return F, leaf_labels

    def predict_proba(self, X, coordinates):
        F, _ = self.decision_function(X, coordinates)
        return _softmax(F)

    def predict(self, X, coordinates):
        F, leaf_labels = self.decision_function(X, coordinates)
        out = self.classes_[_softmax(F).argmax(axis=1)]
        unrouted = leaf_labels == -1
        if unrouted.any():
            out = out.astype(object)
            out[unrouted] = None
        return out

    def score(self, X, y, coordinates):
        from sklearn.metrics import accuracy_score
        return accuracy_score(numpy.asarray(y), self.predict(X, coordinates))

    @property
    def n_leaves(self):
        return len(self.leaves_)


class QuadtreeBoostingClassifier(SpatialIndexBoostingClassifier):
    """Supervised quadtree gradient-boosting classifier (4-way midpoint splits)."""

    def _children(self, coords, bounds, weights):
        min_x, min_y, max_x, max_y = bounds
        mid_x = min_x + (max_x - min_x) / 2
        mid_y = min_y + (max_y - min_y) / 2
        labels = (coords[:, 0] <= mid_x).astype(int) + \
                 (coords[:, 1] <= mid_y).astype(int) * 2
        child_bounds = [
            [mid_x, mid_y, max_x, max_y],
            [min_x, mid_y, mid_x, max_y],
            [mid_x, min_y, max_x, mid_y],
            [min_x, min_y, mid_x, mid_y],
        ]
        return [(child_bounds[i], labels == i) for i in range(4)]

    def _child_positions(self, bounds, coordinates, children):
        """Returns the child position of each point by the midpoint rule in `_children`."""
        min_x, min_y, max_x, max_y = bounds
        mid_x = min_x + (max_x - min_x) / 2
        mid_y = min_y + (max_y - min_y) / 2
        return (coordinates[:, 0] <= mid_x).astype(int) + \
               (coordinates[:, 1] <= mid_y).astype(int) * 2


class KDTreeBoostingClassifier(SpatialIndexBoostingClassifier):
    """
    Supervised KD-tree gradient-boosting classifier (2-way splits along the
    longest axis).

    Parameters
    ----------
    split_method : {'weighted', 'median'} (default: 'weighted')
        'median'   → split at the coordinate median of the longest axis.
        'weighted' → split at the residual-weighted median (weights are the
                     per-point pseudo-residual L1 norm), concentrating splits
                     where the current model misclassifies most.
    """

    def __init__(self, *, split_method='weighted', **kwargs):
        super().__init__(**kwargs)
        self.split_method = split_method

    def _children(self, coords, bounds, weights):
        from kdtree_regression import weighted_median

        min_x, min_y, max_x, max_y = bounds
        vertical = abs(max_x - min_x) > abs(max_y - min_y)
        vals = coords[:, 0] if vertical else coords[:, 1]

        if self.split_method == 'weighted' and weights is not None and weights.sum() > 0:
            mid = weighted_median(vals, weights)
        else:
            mid = numpy.median(vals)

        labels = (vals > mid).astype(int)
        if vertical:
            child_bounds = [[min_x, min_y, mid, max_y],
                            [mid, min_y, max_x, max_y]]
        else:
            child_bounds = [[min_x, min_y, max_x, mid],
                            [min_x, mid, max_x, max_y]]
        return [(child_bounds[i], labels == i) for i in range(2)]

    def _child_positions(self, bounds, coordinates, children):
        """
        Returns the child position of each point by the plane rule in `_children`.

        The plane is the upper edge of child 0 or the lower edge of child 1
        on the split axis. A point on the plane goes to child 0.
        """
        min_x, min_y, max_x, max_y = bounds
        axis = 0 if abs(max_x - min_x) > abs(max_y - min_y) else 1
        position, child = next(iter(children.items()))
        mid = child.bounds[2 + axis] if position == 0 else child.bounds[axis]
        return (coordinates[:, axis] > mid).astype(int)
