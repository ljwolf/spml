"""
External-tree page-membership transformer.

``ExternalTreeTransformer`` wraps an externally provided tree and exposes a
uniform view over it via :func:`functools.singledispatch`:

* fitted sklearn decision trees,
* scipy ``cluster.hierarchy`` linkage matrices (with ``coordinates=``),
* arbitrary ``apply`` callables (with ``leaf_ids=``).

The transformer encodes coordinates as one-hot page membership at a requested
depth (or the leaf partition).  The growing external-tree ensembles
(``SpatialTreeRegressor`` / ``SpatialTreeClassifier``) read the underlying
hierarchy from ``transformer._adapter`` to grow through the tree's splits.

Depth convention
----------------
depth=0      : root (one page covering everything)
depth=d      : up to branching_factor**d pages
depth=None   : the actual leaf partition
depth=[1,2,3]: OHE at each level concatenated horizontally
"""

import functools
import collections.abc

import numpy as np
import scipy.sparse
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import BaseDecisionTree


class _TreeAdapter:
    """Uniform view over an external tree for page-membership queries."""

    def __init__(self, apply, leaf_ids, parent=None, depths=None,
                 ancestor_fn=None, nodes_at_depth_fn=None):
        self._apply_fn          = apply
        self.leaf_ids           = sorted(leaf_ids)
        self.parent             = parent
        self.depths             = depths
        self._ancestor_fn       = ancestor_fn
        self._nodes_at_depth_fn = nodes_at_depth_fn
        self.max_depth          = int(depths.max()) if depths is not None else None
        self._nodes_by_depth    = {}

    def apply(self, coordinates):
        return np.asarray(self._apply_fn(np.asarray(coordinates)))

    def _walk_up(self, node, d):
        while self.depths[node] > d:
            node = self.parent[node]
        return node

    def ancestor(self, node_ids, depth):
        if self.parent is not None:
            depth = min(depth, self.max_depth)
            return np.array([self._walk_up(n, depth) for n in node_ids])
        if self._ancestor_fn is None:
            raise NotImplementedError(
                "depth != None requires ancestor information for this tree"
            )
        return np.asarray(self._ancestor_fn(node_ids, depth))

    def all_nodes_at_depth(self, depth):
        if self.parent is not None:
            depth = min(depth, self.max_depth)
            if depth not in self._nodes_by_depth:
                self._nodes_by_depth[depth] = sorted({
                    self._walk_up(leaf, depth) for leaf in self.leaf_ids
                })
            return self._nodes_by_depth[depth]
        if self._nodes_at_depth_fn is None:
            raise NotImplementedError(
                "depth != None requires nodes_at_depth information for this tree"
            )
        return self._nodes_at_depth_fn(depth)

    def all_leaf_ids(self):
        return self.leaf_ids


@functools.singledispatch
def _build_adapter(tree, *, coordinates=None, leaf_ids=None,
                   ancestor_fn=None, nodes_at_depth_fn=None):
    raise TypeError(
        f"ExternalTreeTransformer cannot adapt a tree of type {type(tree)!r}. "
        "Pass a fitted sklearn decision tree, a scipy linkage matrix (with "
        "coordinates=), or a callable apply_fn (with leaf_ids=)."
    )


@_build_adapter.register(BaseDecisionTree)
def _adapt_sklearn(tree, *, coordinates=None, **__):
    t = tree.tree_
    n_nodes = t.node_count
    depths = np.zeros(n_nodes, dtype=int)
    parent = np.full(n_nodes, -1, dtype=int)
    stack = [(0, 0)]
    while stack:
        node, d = stack.pop()
        depths[node] = d
        left, right = t.children_left[node], t.children_right[node]
        if left != -1:
            parent[left] = parent[right] = node
            stack.append((left, d + 1))
            stack.append((right, d + 1))
    leaf_ids = np.where(t.children_left == -1)[0]
    return _TreeAdapter(
        apply=lambda c: tree.apply(c),
        leaf_ids=leaf_ids.tolist(),
        parent=parent,
        depths=depths,
    )


@_build_adapter.register(np.ndarray)
def _adapt_linkage(Z, *, coordinates=None, **__):
    if coordinates is None:
        raise ValueError("a linkage matrix requires coordinates=")
    from scipy.spatial import cKDTree

    coordinates = np.asarray(coordinates)
    n = len(Z) + 1
    n_nodes = 2 * n - 1
    depths = np.zeros(n_nodes, dtype=int)
    parent = np.full(n_nodes, -1, dtype=int)
    root = n_nodes - 1
    stack = [(root, 0)]
    while stack:
        node, d = stack.pop()
        depths[node] = d
        if node >= n:
            i = node - n
            left, right = int(Z[i, 0]), int(Z[i, 1])
            parent[left] = parent[right] = node
            stack.append((left, d + 1))
            stack.append((right, d + 1))
    nn = cKDTree(coordinates)
    return _TreeAdapter(
        apply=lambda c: nn.query(c)[1],
        leaf_ids=list(range(n)),
        parent=parent,
        depths=depths,
    )


@_build_adapter.register(collections.abc.Callable)
def _adapt_callable(apply_fn, *, coordinates=None, leaf_ids=None,
                    ancestor_fn=None, nodes_at_depth_fn=None):
    if leaf_ids is None:
        raise ValueError("a callable apply_fn requires leaf_ids=")
    return _TreeAdapter(
        apply=lambda c: apply_fn(c),
        leaf_ids=list(leaf_ids),
        ancestor_fn=ancestor_fn,
        nodes_at_depth_fn=nodes_at_depth_fn,
    )


class ExternalTreeTransformer(TransformerMixin, BaseEstimator):
    """
    Page-membership transformer backed by any external tree.

    A single class that adapts several tree formats via functools.singledispatch
    on the ``tree`` argument:

    * fitted sklearn DecisionTreeRegressor / DecisionTreeClassifier::

          ExternalTreeTransformer(tree)

    * scipy.cluster.hierarchy linkage matrix Z (needs training coordinates)::

          ExternalTreeTransformer(Z, coordinates=coords)

    * any other format, via a user apply callable::

          ExternalTreeTransformer(apply_fn, leaf_ids=...,
                                  ancestor_fn=..., nodes_at_depth_fn=...)

    The tree must operate on coordinates (X[:, 0], X[:, 1]).

    Parameters
    ----------
    tree : fitted sklearn tree, linkage ndarray, or apply callable
    coordinates : array-like of shape (n, 2)
        Required for a linkage matrix (the clustered training coordinates).
    depth : int, list of int, or None (default: None)
        Default depth used by transform(); None means the leaf partition.
    leaf_ids : array-like or None
        Required for a callable tree: all possible leaf node IDs.
    ancestor_fn : callable or None
        ancestor_fn(node_ids, depth) -> node IDs at depth <= d (callable trees).
    nodes_at_depth_fn : callable or None
        nodes_at_depth_fn(depth) -> sorted node IDs at that depth (callable trees).
    """

    def __init__(self, tree, coordinates=None, depth=None, *,
                 leaf_ids=None, ancestor_fn=None, nodes_at_depth_fn=None):
        self.tree              = tree
        self.coordinates       = coordinates
        self.depth             = depth
        self.leaf_ids          = leaf_ids
        self.ancestor_fn       = ancestor_fn
        self.nodes_at_depth_fn = nodes_at_depth_fn
        self._adapter = _build_adapter(
            tree,
            coordinates=coordinates,
            leaf_ids=leaf_ids,
            ancestor_fn=ancestor_fn,
            nodes_at_depth_fn=nodes_at_depth_fn,
        )

    def transform(self, coordinates, depth=None):
        """Return a sparse one-hot matrix of page membership."""
        coordinates = np.asarray(coordinates)
        if depth is None:
            depth = self.depth

        leaf_ids = self._adapter.apply(coordinates)

        if depth is None:
            return self._ohe(leaf_ids, self._adapter.all_leaf_ids())

        if isinstance(depth, (list, tuple)):
            parts = [
                self._ohe(self._adapter.ancestor(leaf_ids, d),
                          self._adapter.all_nodes_at_depth(d))
                for d in depth
            ]
            return scipy.sparse.hstack(parts, format="csr")

        return self._ohe(self._adapter.ancestor(leaf_ids, depth),
                         self._adapter.all_nodes_at_depth(depth))

    def get_feature_names_out(self, depth=None):
        """Return feature names matching the columns of transform()."""
        if depth is None:
            depth = self.depth

        if depth is None:
            ids = self._adapter.all_leaf_ids()
        elif isinstance(depth, (list, tuple)):
            return np.concatenate([
                np.array([f"page_{i}" for i in self._adapter.all_nodes_at_depth(d)])
                for d in depth
            ])
        else:
            ids = self._adapter.all_nodes_at_depth(depth)

        return np.array([f"page_{i}" for i in ids])

    @staticmethod
    def _ohe(node_ids, categories):
        ohe = OneHotEncoder(sparse_output=True, handle_unknown="ignore")
        ohe.fit(np.asarray(categories).reshape(-1, 1))
        return ohe.transform(np.asarray(node_ids).reshape(-1, 1))
