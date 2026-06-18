"""
Spatial tree ensembles grown over a contiguity graph.

A connectivity graph is turned into a minimum spanning tree and grown top-down
into regions. Growth blends three ideas:

* **General objective.** Each candidate cut is scored by the within-region fit
  of the *actual* estimator. When the estimator is a plain intercept
  ``LinearRegression`` the scoring uses the fast decomposed OLS
  residual-sum-of-squares path (provably identical cuts, no per-edge refits);
  any other estimator (or a classifier) is scored by refitting it on each side
  of every candidate edge.
* **Global growth.** At every step the single most-improving edge across *all*
  current regions is cut (SKATER-style greedy), not a per-region depth-first
  descent.
* **Adaptive stopping.** A cut is accepted only when a child's local model beats
  its parent's predictions on the child's own observations by at least
  ``split_tolerance``; growth also respects ``min_leaf_size``, ``max_depth`` and
  ``max_leaves``. There is no fixed number of regions -- the data decide.

Prediction routes each point to the region of its nearest training observation
and applies that region's model.
"""

from collections import namedtuple

import numpy as np
from scipy import sparse
from scipy.optimize import OptimizeWarning
from scipy.sparse import csgraph as cg
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone

deletion = namedtuple("deletion", ("in_node", "out_node", "score"))


class _ConstantClassifier:
    """Predicts a single observed label; used for degenerate single-class regions."""

    def __init__(self, label):
        self.classes_ = np.array([label])
        self._label = label

    def predict(self, X):
        return np.full(len(X), self._label)

    def predict_proba(self, X):
        return np.ones((len(X), 1))


# ---------------------------------------------------------------------------
# Minimum-spanning-forest cut engine (implementation detail; not public API)
# ---------------------------------------------------------------------------


def _to_sparse(graph):
    """Coerce a libpysal ``Graph``/``W`` or a scipy sparse matrix to CSR."""
    if sparse.issparse(graph):
        A = graph
    elif hasattr(graph, "sparse"):  # libpysal Graph / W
        A = graph.sparse
    else:
        raise TypeError(
            "graph must be a libpysal Graph/W or a scipy sparse matrix; got "
            f"{type(graph)!r}."
        )
    A = sparse.csr_matrix(A)
    A.eliminate_zeros()
    return A


def _find_cut(
    MSF,
    data,
    quorum=-np.inf,
    labels=None,
    target_label=None,
    y=None,
    model=None,
    score_function=None,
):
    """
    Find the single best edge to cut from a minimum spanning forest.

    Three scoring paths, chosen by the arguments:

    * ``model`` given -> **general**: refit a clone of ``model`` on each side of
      every candidate edge and score it with ``score_function`` (higher better).
      ``y`` is required. Used for classifiers and non-OLS regressors.
    * ``y`` given (no ``model``) -> **decomposed OLS**: within-region OLS
      residual sum of squares of ``y ~ data`` (with intercept), from additive
      sufficient statistics. Identical cuts to refitting an intercept-OLS, far
      cheaper.
    * neither -> **decomposed SSD**: squared deviation of ``data`` about its
      mean (classic SKATER).

    Returns ``(deletion(in_node, out_node, score), base_total)`` where ``score``
    is the total objective after the cut and ``base_total`` is the total before
    it, so ``base_total - score`` is the cut's objective reduction.
    """
    data = np.asarray(data, dtype=float)
    if data.ndim == 1:
        data = data[:, None]

    if (labels is None) != (target_label is None):
        raise ValueError("Both ``labels`` and ``target_label`` must be supplied.")
    zero_in = (labels is not None) and (target_label is not None)

    N, P = data.shape
    sym = (MSF + MSF.T).tocsr()
    stored = set(zip(*MSF.nonzero(), strict=False))

    general = model is not None
    if general and y is None:
        raise ValueError("`y` (the response) is required when `model` is given.")
    if general and score_function is None:
        raise ValueError("`score_function` is required when `model` is given.")
    regression = (y is not None) and not general

    if general:
        # No decomposition: region cost is the model's own loss. Negate the
        # higher-is-better score (a per-observation mean) and rescale by region
        # size to get a *sum* loss. Summed losses are additive across a
        # partition, so costC - (costA + costB) is a meaningful reduction that is
        # comparable across regions of different sizes -- matching the summed-RSS
        # decomposed paths. A bare mean loss is not additive and would bias the
        # global greedy ranking toward (or against) small regions.
        y = np.asarray(y).ravel()

        def _region_cost(idx):
            try:
                m = clone(model)
                m.fit(data[idx], y[idx])
                return -float(score_function(y[idx], m.predict(data[idx]))) * len(idx)
            except (ValueError, np.linalg.LinAlgError):
                return np.inf
    elif regression:
        y = np.asarray(y, dtype=float).ravel()
        Z = np.empty((N, P + 1))
        Z[:, 0] = 1.0
        Z[:, 1:] = data
        d = P + 1
        if quorum == -np.inf:
            quorum = d

        def _rss(ZZ, Zy, yy):
            try:
                beta = np.linalg.solve(ZZ, Zy)
            except np.linalg.LinAlgError:
                beta = np.linalg.lstsq(ZZ, Zy, rcond=None)[0]
            return float(yy - Zy @ beta)

    # --- Pass 1: DFS each component; record topology + per-component totals. ---
    seen = np.zeros(N, dtype=bool)
    components = []
    for seed in range(N):
        if seen[seed]:
            continue
        order, pred = cg.depth_first_order(
            sym, seed, directed=False, return_predecessors=True
        )
        seen[order] = True
        components.append((order, pred))

    comp_cost = []
    comp_stats = []
    base_total = 0.0
    for order, pred in components:
        nC = order.size
        if general:
            cC = 0.0 if nC <= 1 else _region_cost(order)
            comp_stats.append(None)
        elif regression:
            Zc = Z[order]
            yc = y[order]
            ZZ_C = Zc.T @ Zc
            Zy_C = Zc.T @ yc
            yy_C = float(yc @ yc)
            cC = _rss(ZZ_C, Zy_C, yy_C)
            comp_stats.append((ZZ_C, Zy_C, yy_C))
        else:
            block = data[order]
            sC = block.sum(axis=0)
            sqC = (block * block).sum(axis=0)
            cC = (sqC - sC * sC / nC).sum()
            comp_stats.append((sC, sqC))
        comp_cost.append(cC)
        base_total += cC

    # --- Pass 2: score every tree edge with the decomposed/general objective. ---
    best_deletion = deletion(np.nan, np.nan, np.inf)
    for ci, (order, pred) in enumerate(components):
        if order.size <= 1:
            continue
        nC = order.size
        costC = comp_cost[ci]
        subn = np.zeros(N, dtype=np.int64)
        subn[order] = 1
        if general:
            # Only subtree membership is needed. In DFS preorder a subtree is
            # contiguous, so order[pos_u : pos_u + nA] is exactly child u's
            # subtree, and posarr maps node -> position.
            for u in order[::-1]:
                p = pred[u]
                if p == -9999 or p == u:
                    continue
                subn[p] += subn[u]
            posarr = np.empty(N, dtype=np.int64)
            posarr[order] = np.arange(nC)
        elif regression:
            subZZ = np.zeros((N, d, d))
            subZy = np.zeros((N, d))
            subyy = np.zeros(N)
            Zo = Z[order]
            yo = y[order]
            subZZ[order] = Zo[:, :, None] * Zo[:, None, :]
            subZy[order] = Zo * yo[:, None]
            subyy[order] = yo * yo
            for u in order[::-1]:
                p = pred[u]
                if p == -9999 or p == u:
                    continue
                subn[p] += subn[u]
                subZZ[p] += subZZ[u]
                subZy[p] += subZy[u]
                subyy[p] += subyy[u]
            ZZ_C, Zy_C, yy_C = comp_stats[ci]
        else:
            subs = np.zeros((N, P))
            subsq = np.zeros((N, P))
            subs[order] = data[order]
            subsq[order] = data[order] * data[order]
            for u in order[::-1]:
                p = pred[u]
                if p == -9999 or p == u:
                    continue
                subn[p] += subn[u]
                subs[p] += subs[u]
                subsq[p] += subsq[u]
            sC, sqC = comp_stats[ci]

        for u in order:
            p = pred[u]
            if p == -9999 or p == u:
                continue
            if zero_in and labels[u] != target_label:
                continue
            nA = subn[u]
            nB = nC - nA
            if nA < quorum or nB < quorum:
                continue
            if general:
                iu = posarr[u]
                A_idx = order[iu : iu + nA]
                B_idx = np.concatenate((order[:iu], order[iu + nA :]))
                costA = _region_cost(A_idx)
                costB = _region_cost(B_idx)
            elif regression:
                costA = _rss(subZZ[u], subZy[u], subyy[u])
                costB = _rss(ZZ_C - subZZ[u], Zy_C - subZy[u], yy_C - subyy[u])
            else:
                sA = subs[u]
                sqA = subsq[u]
                sB = sC - sA
                sqB = sqC - sqA
                costA = (sqA - sA * sA / nA).sum()
                costB = (sqB - sB * sB / nB).sum()
            score = base_total - costC + costA + costB
            if score < best_deletion.score:
                edge = (p, u) if (p, u) in stored else (u, p)
                best_deletion = deletion(edge[0], edge[1], score)
    return best_deletion, base_total


def _make_cut(in_node, out_node, score, MSF):
    """Remove edge (in_node, out_node) from ``MSF`` in place; return components."""
    if not np.isfinite(score):
        raise OptimizeWarning(
            f"Score of the ({in_node},{out_node}) cut is inf; quorum unmet."
        )
    MSF[in_node, out_node] = 0
    MSF.eliminate_zeros()
    return (MSF, *cg.connected_components(MSF, directed=False))


def _split_sides(MSF, labels, lbl, in_node, out_node):
    """The two subtrees region ``lbl`` falls into when edge (in,out) is removed."""
    sym = (MSF + MSF.T).tocsr()
    indptr, indices = sym.indptr, sym.indices
    visited = {in_node}
    stack = [in_node]
    while stack:
        x = stack.pop()
        for nb in indices[indptr[x] : indptr[x + 1]]:
            nb = int(nb)
            if (x == in_node and nb == out_node) or (x == out_node and nb == in_node):
                continue
            if labels[nb] != lbl or nb in visited:
                continue
            visited.add(nb)
            stack.append(nb)
    members = np.flatnonzero(labels == lbl)
    a_mask = np.isin(members, list(visited))
    return members[a_mask], members[~a_mask]


# ---------------------------------------------------------------------------
# Public estimators
# ---------------------------------------------------------------------------


class SpatialTreeEnsemble(BaseEstimator):
    """
    Base spatial tree ensemble (see module docstring).

    Parameters
    ----------
    graph : libpysal Graph/W or scipy sparse matrix
        Symmetric contiguity graph over the rows of ``X``.
    model : sklearn estimator or None (default None)
        Cloned to score cuts and to fit each region. Defaults to the
        task-appropriate model.
    split_tolerance : float (default 1e-5)
        Minimum improvement of a child's model over its parent's (on the child's
        observations) for a cut to be accepted.
    min_leaf_size : int (default 10)
        Smallest allowed region; no region below ``2 * min_leaf_size`` is split.
    max_depth : int or None (default None)
        Stop growing past this depth. None grows until no cut is accepted.
    max_leaves : int or float (default inf)
        Cap on the number of regions.
    """

    def __init__(
        self,
        graph,
        model=None,
        split_tolerance=1e-5,
        min_leaf_size=10,
        max_depth=None,
        max_leaves=float("inf"),
    ):
        self.graph = graph
        self.model = model
        self.split_tolerance = split_tolerance
        self.min_leaf_size = min_leaf_size
        self.max_depth = max_depth
        self.max_leaves = max_leaves

    # ---------------------------------------------------------------- hooks
    def _make_default_model(self):
        raise NotImplementedError

    def _score(self, y_true, y_pred):
        raise NotImplementedError

    def _fit_page(self, model, X, y):
        model.fit(X, y)
        return model

    def _cut_kwargs(self, model, X, y):
        """Objective configuration for ``_find_cut`` (general path by default)."""
        return {"y": np.asarray(y).ravel(), "model": model, "score_function": self._score}

    # ------------------------------------------------------------------ fit
    def fit(self, X, y, coordinates):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).ravel()
        self._coords = np.asarray(coordinates)
        n = len(y)

        model = clone(self.model) if self.model is not None else self._make_default_model()
        cut_kwargs = self._cut_kwargs(model, X, y)

        MSF = cg.minimum_spanning_tree(_to_sparse(self.graph))
        MSF.eliminate_zeros()

        labels = np.zeros(n, dtype=int)
        next_lbl = 1
        region_members = {0: np.arange(n)}
        region_depth = {0: 0}
        region_model = {0: self._fit_page(clone(model), X, y)}
        cand = {}  # lbl -> (reduction, cut, A_idx, B_idx, model_A, model_B)

        def consider(lbl):
            members = region_members[lbl]
            depth = region_depth[lbl]
            if len(members) < 2 * self.min_leaf_size:
                return
            if self.max_depth is not None and depth >= self.max_depth:
                return
            cut, base_total = _find_cut(
                MSF,
                X,
                quorum=self.min_leaf_size,
                labels=labels,
                target_label=lbl,
                **cut_kwargs,
            )
            if not np.isfinite(cut.score) or np.isnan(cut.in_node):
                return
            in_node, out_node = int(cut.in_node), int(cut.out_node)
            A_idx, B_idx = _split_sides(MSF, labels, lbl, in_node, out_node)
            if len(A_idx) < self.min_leaf_size or len(B_idx) < self.min_leaf_size:
                return
            model_a = self._fit_page(clone(model), X[A_idx], y[A_idx])
            model_b = self._fit_page(clone(model), X[B_idx], y[B_idx])
            parent_model = region_model[lbl]
            gain_a = self._score(y[A_idx], model_a.predict(X[A_idx])) - self._score(
                y[A_idx], parent_model.predict(X[A_idx])
            )
            gain_b = self._score(y[B_idx], model_b.predict(X[B_idx])) - self._score(
                y[B_idx], parent_model.predict(X[B_idx])
            )
            if max(gain_a, gain_b) < self.split_tolerance:
                return
            cand[lbl] = (base_total - cut.score, cut, A_idx, B_idx, model_a, model_b)

        consider(0)
        while cand:
            if len(region_members) + 1 > self.max_leaves:
                break
            best = max(cand, key=lambda lbl: cand[lbl][0])
            _, cut, A_idx, B_idx, model_a, model_b = cand.pop(best)

            _make_cut(int(cut.in_node), int(cut.out_node), cut.score, MSF)
            new_lbl = next_lbl
            next_lbl += 1
            labels[B_idx] = new_lbl
            depth = region_depth[best] + 1
            region_members[best] = A_idx
            region_model[best] = model_a
            region_depth[best] = depth
            region_members[new_lbl] = B_idx
            region_model[new_lbl] = model_b
            region_depth[new_lbl] = depth
            consider(best)
            consider(new_lbl)

        self.labels_ = labels
        self.page_models_ = dict(region_model)
        self.page_depth_ = dict(region_depth)
        self.depth_ = np.empty(n, dtype=int)
        for lbl, members in region_members.items():
            self.depth_[members] = region_depth[lbl]
        return self

    # -------------------------------------------------------------- routing
    def _route(self, coordinates):
        from scipy.spatial import cKDTree

        coordinates = np.asarray(coordinates)
        nearest = cKDTree(self._coords).query(coordinates)[1]
        return self.labels_[nearest]

    @property
    def n_leaves(self):
        return len(self.page_models_)


class SpatialTreeRegressor(RegressorMixin, SpatialTreeEnsemble):
    """Spatial tree ensemble for regression (one local regressor per region)."""

    def _make_default_model(self):
        from sklearn.linear_model import LinearRegression

        return LinearRegression()

    def _score(self, y_true, y_pred):
        from sklearn.metrics import mean_squared_error

        return -mean_squared_error(y_true, y_pred)

    def _cut_kwargs(self, model, X, y):
        from sklearn.linear_model import LinearRegression

        # a plain intercept-OLS is reproduced exactly by the fast decomposed path
        if (
            type(model) is LinearRegression
            and getattr(model, "fit_intercept", True)
            and not getattr(model, "positive", False)
        ):
            return {"y": np.asarray(y, dtype=float).ravel()}
        return {
            "y": np.asarray(y, dtype=float).ravel(),
            "model": model,
            "score_function": self._score,
        }

    def predict(self, X, coordinates):
        X = np.asarray(X, dtype=float)
        route = self._route(coordinates)
        out = np.full(len(X), np.nan)
        for lbl, model in self.page_models_.items():
            mask = route == lbl
            if mask.any():
                out[mask] = model.predict(X[mask])
        return out

    def score(self, X, y, coordinates):
        from sklearn.metrics import r2_score

        return r2_score(np.asarray(y), self.predict(X, coordinates))


class SpatialTreeClassifier(ClassifierMixin, SpatialTreeEnsemble):
    """
    Spatial tree ensemble for classification (one local classifier per region).

    Classification has no decomposable cut objective, so cuts are always scored
    by refitting the classifier on each candidate edge. Single-class regions
    predict their constant label.
    """

    def _make_default_model(self):
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression()

    def _score(self, y_true, y_pred):
        from sklearn.metrics import accuracy_score

        return accuracy_score(y_true, y_pred)

    def _fit_page(self, model, X, y):
        if len(np.unique(y)) < 2:
            return _ConstantClassifier(y[0])
        model.fit(X, y)
        return model

    def fit(self, X, y, coordinates):
        self.classes_ = np.unique(np.asarray(y))
        return super().fit(X, y, coordinates)

    def predict_proba(self, X, coordinates):
        X = np.asarray(X, dtype=float)
        route = self._route(coordinates)
        proba = np.zeros((len(X), len(self.classes_)))
        col = {c: i for i, c in enumerate(self.classes_)}

        def _fill(model, mask):
            rows = np.flatnonzero(mask)
            local = model.predict_proba(X[mask])
            for j, c in enumerate(model.classes_):
                proba[rows, col[c]] = local[:, j]

        for lbl, model in self.page_models_.items():
            mask = route == lbl
            if mask.any():
                _fill(model, mask)
        return proba

    def predict(self, X, coordinates):
        proba = self.predict_proba(X, coordinates)
        out = self.classes_[proba.argmax(axis=1)]
        unrouted = ~proba.any(axis=1)
        if unrouted.any():
            out = out.astype(object)
            out[unrouted] = None
        return out

    def score(self, X, y, coordinates):
        from sklearn.metrics import accuracy_score

        return accuracy_score(np.asarray(y), self.predict(X, coordinates))
