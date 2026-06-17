"""optimized spanning forest regressor, departing from spopt.
Carries decomposed-SSD edge-scan optimizations to fit and find_cut.
"""
# ruff: noqa: C408, B006, E731, N803, N806

import time
import warnings
from collections import namedtuple

import numpy as np
from scipy import sparse
from scipy.optimize import OptimizeWarning
from scipy.sparse import csgraph as cg
from sklearn.metrics import pairwise as skm


deletion = namedtuple("deletion", ("in_node", "out_node", "score"))


class SpanningForest:
    def __init__(
        self,
        dissimilarity=skm.manhattan_distances,
        affinity=None,
        reduction=np.sum,
        center=np.mean,
        verbose=False,
    ):
        """
        Initialize the SKATER algorithm.

        Parameters
        ----------

        dissimilarity : callable (default sklearn.metrics.pairwise.manhattan_distances)
            A callable distance metric.
        affinity : callable (default None)
            A callable affinity metric between 0 and 1, which is inverted to provide a
            dissimilarity metric. Either ``affinity`` or ``dissimilarity`` should be
            provided. If both arguments are provided ``dissimilarity`` is chosen.
        reduction : callable (default numpy.sum())
            The reduction applied over all clusters to provide the map score.
        center : callable (default numpy.mean())
            The method for computing the center of each region in attribute space.
        verbose : bool, int (default False)
            Flag for how much output to provide to the user,
            in terms of print statements and progress bars. Set to ``1`` for
            minimal output and ``2`` for full output.

        Notes
        -----

        Optimization occurs with respect to a *dissimilarity* metric, so the
        reduction should yield some kind of score where larger values are
        *less desirable* than smaller values. Typically, this means we use addition.

        """

        if affinity is not None and dissimilarity is not None:
            warnings.warn(
                "Both the `affinity` and `dissimilarity` arguments "
                "were passed in. Defaulting `dissimilarity`.",
                UserWarning,
                stacklevel=3,
            )
            affinity = None

        if affinity is not None:
            # invert the 0,1 affinity to an unbounded positive dissimilarity
            metric = lambda x, y: -np.log(affinity(x, y))
        else:
            metric = lambda x, y: dissimilarity(x, y)

        self.metric = metric
        self.reduction = reduction
        self.center = center
        self.verbose = verbose

    def __repr__(self):
        return (
            f"Minimum_Spanning_Tree_Pruning(metric={self.metric}, "
            f"reduction={self.reduction}, center={self.center})"
        )

    def fit(
        self,
        n_clusters,
        W,
        data=None,
        quorum=-np.inf,
        trace=False,
        islands="increase",
        y=None,
    ):
        """

        Parameters
        ----------

        n_clusters : int
            The number of clusters to form.
        W : libpysal.weights.W
            A PySAL weights object created from given data expressing the neighbor
            relationships between observations. It must be symmetric and binary, for
            example: Queen/Rook, DistanceBand, or a symmetrized KNN.
        data : numpy.ndarray (default None)
            An array of shape :math:`(N,P)` with :math:`N`
            observations and :math:`P` features.
        quorum : int, float (default -numpy.inf)
            The floor on the size of regions.
        trace : bool (default False)
            Flag denoting whether to store intermediate
            labelings as the tree gets pruned.
        islands : str (default 'increase')
            Description of what to do with islands. If ``'ignore'``, the algorithm will
            discover ``n_clusters`` regions, treating islands as their own regions. If
            "increase", the algorithm will discover ``n_clusters`` regions,
            treating islands as separate from ``n_clusters``.
        y : numpy.ndarray (default None)
            An optional :math:`(N,)` response vector. If provided, regions are
            scored by the within-region ordinary-least-squares residual sum of
            squares of ``y ~ data`` (with an intercept) instead of the squared
            deviation of ``data`` about its mean. ``data`` then plays the role of
            the regression design matrix :math:`X`. The MST is still built on
            ``data`` so the tree respects covariate contiguity.

        Notes
        -----

        Optimization occurs with respect to a *dissimilarity* metric, so the
        problem *minimizes* the map dissimilarity. Therefore, lower scores are better.

        """
        if trace:
            self._trace = []
        if data is None:
            data = np.ones((W.n, 1))
        data = np.asarray(data, dtype=float)
        if data.ndim == 1:
            data = data[:, None]
        if y is not None and quorum == -np.inf:
            # an OLS fit with an intercept needs at least P + 1 observations,
            # so that is the smallest sensible region when none is requested.
            quorum = data.shape[1] + 1
        W.transform = "b"
        W = W.sparse
        start = time.time()

        super_verbose = self.verbose > 1
        start_W = time.time()
        # Edgewise covariate distance over W's sparse structure. We never
        # materialize the dense N x N attribute kernel: only the |E| existing
        # edges get a weight. Euclidean distance is used so the MST weighting is
        # consistent with the decomposable SSD objective optimized in find_cut.
        Wsp = W.tocoo()
        diff = data[Wsp.row] - data[Wsp.col]
        wd = np.sqrt(np.einsum("ij,ij->i", diff, diff))
        # MST treats exact zeros as absent edges; nudge so zero-distance
        # neighbors remain connectable (matches the all-ones/no-data case too).
        wd = np.where(wd > 0, wd, np.finfo(float).tiny)
        dissim = sparse.csr_matrix((wd, (Wsp.row, Wsp.col)), shape=Wsp.shape)
        dissim.eliminate_zeros()
        end_W = time.time() - start_W

        if super_verbose:
            print(f"Computing Affinity Kernel took {end_W:.2f}s")

        tree_time = time.time()
        MSF = cg.minimum_spanning_tree(dissim)
        tree_time = time.time() - tree_time
        if super_verbose:
            print(f"Computing initial MST took {tree_time:.2f}s")

        init_component_time = time.time()
        current_n_subtrees, current_labels = cg.connected_components(
            MSF, directed=False
        )
        init_component_time = time.time() - init_component_time

        if super_verbose:
            print(f"Computing connected components took {init_component_time:.2f}s.")

        if current_n_subtrees > 1:
            island_warnings = [
                (
                    f"Increasing `n_clusters` from {n_clusters} to "
                    f"{n_clusters + current_n_subtrees} in order to "
                    "account for islands."
                ),
                (
                    "Counting islands towards the remaining "
                    f"{n_clusters - current_n_subtrees} clusters."
                ),
            ]
            ignoring_islands = int(islands.lower() == "ignore")
            chosen_warning = island_warnings[ignoring_islands]
            warnings.warn(
                f"By default, the graph is disconnected! {chosen_warning}",
                OptimizeWarning,
                stacklevel=3,
            )
            if not ignoring_islands:
                n_clusters += current_n_subtrees
            _, island_populations = np.unique(current_labels, return_counts=True)
            if (island_populations < quorum).any():
                raise ValueError(
                    "Islands must be larger than the quorum. If not, drop the small "
                    "islands and solve for clusters in the remaining field."
                )
        if trace:
            self._trace.append((current_labels, deletion(np.nan, np.nan, np.inf)))
            if super_verbose:
                print(self._trace[-1])
        while current_n_subtrees < n_clusters:  # while we don't have enough regions
            best_deletion = self.find_cut(
                MSF,
                data,
                quorum=quorum,
                labels=None,
                target_label=None,
                y=y,
            )

            # if our search succeeds
            if np.isfinite(best_deletion.score):
                # accept the best move as *the* move
                if super_verbose:
                    print(f"making cut {best_deletion}...")
                MSF, current_n_subtrees, current_labels = self.make_cut(
                    *best_deletion, MSF=MSF
                )
            # otherwise, it means the MSF admits no further cuts (no backtracking here)
            else:
                current_n_subtrees, current_labels = cg.connected_components(
                    MSF, directed=False
                )
                warnings.warn(
                    (
                        "MSF contains no valid moves after finding "
                        f"{current_n_subtrees} subtrees. Decrease the "
                        f"size of your quorum to find the remaining "
                        f"{n_clusters - current_n_subtrees} subtrees."
                    ),
                    OptimizeWarning,
                    stacklevel=3,
                )
                self.current_labels_ = current_labels
                self.minimum_spanning_forest_ = MSF
                self._elapsed_time = time.time() - start
                return self
            if trace:
                self._trace.append((current_labels, best_deletion))

        self.current_labels_ = current_labels
        self.minimum_spanning_forest_ = MSF
        self._elapsed_time = time.time() - start
        return self

    def score(self, data, labels=None, quorum=-np.inf):
        """
        This yields a score for the data, given the labels provided.
        If no labels are provided, and the object has been fit, then
        the labels discovered from the previous fit are used.
        If ``quorum`` is not passed, it is assumed to be irrelevant.
        If ``quorum`` is passed and the labels do not meet ``quorum``,
        the score is ``inf``.

        Parameters
        ----------

        data : numpy.array
            An :math:`(N,P)` array of data on which to compute
            the score of the regions expressed in ``labels``.
        labels : numpy.array (default None)
            An :math:`(N,)` vector of labels expressing the
            classification of each observation into a region.
        quorum : int, float (default -numpy.inf)
            The floor on the size of regions, which can be -inf if there is
            no lower bound (default). Any region below quorum makes the score inf.

        Notes
        -----

        Optimization occurs with respect to a *dissimilarity* metric, so the
        problem *minimizes* the map dissimilarity. So, lower scores are better.

        """
        if labels is None:
            try:
                labels = self.current_labels_
            except AttributeError:
                raise ValueError(
                    "Labels not provided and ``MSF_Prune object`` "
                    "has not been fit to data yet."
                ) from None

        assert data.shape[0] == len(labels), (
            f"Length of label array ({labels.shape[0]}) "
            f"does not match length of data ({data.shape[0]})!"
        )

        _, subtree_quorums = np.unique(labels, return_counts=True)
        n_subtrees = len(subtree_quorums)
        if (subtree_quorums < quorum).any():
            return np.inf
        part_scores = [
            self.reduction(
                self.metric(
                    data[labels == _l],
                    self.center(data[labels == _l], axis=0).reshape(1, -1),
                )
            )
            for _l in range(n_subtrees)
        ]
        return self.reduction(part_scores).item()

    def find_cut(
        self,
        MSF,
        data=None,
        quorum=-np.inf,
        labels=None,
        target_label=None,
        make=False,
        y=None,
    ):
        """
        Find the best cut from the MSF.

        Parameters
        ----------

        MSF : scipy.sparse.csgraph.minimum_spanning_tree
            An :math:`(N,N)` scipy sparse matrix with zero elements removed.
            representing the adjacency matrix for the minimum spanning forest.
            It is constructed from ``sparse.csgraph.sparse_from_dense`` or using
            ``MSF.eliminate_zeros()``. You **MUST** remove zero entries for this
            to work, otherwise they are considered no-cost paths.
        data : numpy.array (default None)
            An :math:`(N,P)` attribute matrix. If not provided, it is
            replaced with an :math:`(N,1)` vector of ones.
        quorum : int, float (default -numpy.inf)
            The minimum number of elements in the region
        labels : numpy.array (default None)
            An :math:`(N,)` vector of labels expressing the classification of each
            observation into a region. This represents the "cluster labels"
            for disconnected components of the graph.
        target_label : int (default None)
            The target label from the labels array to subset the MSF. If passed along
            with ``labels``, then a cut will be found that is restricted
            to that subset of the ``MSF``.
        make : bool (default False)
            Whether or not to modify the input ``MSF`` in order
            to make the best cut that was found.
        y : numpy.array (default None)
            An optional :math:`(N,)` response. When provided, each region is
            scored by the within-region OLS residual sum of squares of
            ``y ~ data`` (with an intercept); ``data`` is the design matrix.
            When ``None``, regions are scored by the squared deviation of
            ``data`` about its mean (the classic SKATER SSD objective).

        Returns
        -------

        namedtuple
            A ``namedtuple`` with ``in_node``, ``out_node``, and ``score``.

        """
        if data is None:
            data = np.ones((MSF.shape[0], 1))
        data = np.asarray(data, dtype=float)
        if data.ndim == 1:
            data = data[:, None]

        if (labels is None) != (target_label is None):
            raise ValueError(
                "Both ``labels`` and ``target_label`` must be supplied! Only "
                f"{['labels', 'target_label'][int(target_label is None)]} provided."
            )
        zero_in = (labels is not None) and (target_label is not None)

        N, P = data.shape
        # Undirected view of the forest for traversal and neighbor lookups. The
        # MSF is stored in a single orientation; symmetrize so DFS sees both ends.
        sym = (MSF + MSF.T).tocsr()
        # Orientation each edge is actually stored in, so make_cut zeros a real
        # entry rather than inserting a spurious one.
        stored = set(zip(*MSF.nonzero(), strict=False))

        regression = y is not None
        if regression:
            # Design matrix with intercept; OLS sufficient statistics are
            # (n, Z'Z, Z'y, y'y), all additive over observations, so the same
            # subtree rollup that serves SSD also serves regression RSS.
            y = np.asarray(y, dtype=float).ravel()
            Z = np.empty((N, P + 1))
            Z[:, 0] = 1.0
            Z[:, 1:] = data
            d = P + 1
            if quorum == -np.inf:
                # OLS with an intercept is unidentified below P + 1 points
                quorum = d

            def _rss(ZZ, Zy, yy):
                # within-region OLS residual sum of squares from sufficient stats
                try:
                    beta = np.linalg.solve(ZZ, Zy)
                except np.linalg.LinAlgError:
                    beta = np.linalg.lstsq(ZZ, Zy, rcond=None)[0]
                return float(yy - Zy @ beta)

        # --- Pass 1: DFS each component (via scipy); record topology + totals. ---
        # Removing a tree edge (parent, child) splits a component into the child's
        # subtree A and the remainder B. Both the SSD and the regression RSS are
        # built from additive sufficient statistics, so one rooted DFS per
        # component yields every candidate edge's score in O(P) (SSD) or O(P^3)
        # (regression solve) instead of recomputing components + score per edge.
        seen = np.zeros(N, dtype=bool)
        components = []  # (order, pred)
        for seed in range(N):
            if seen[seed]:
                continue
            order, pred = cg.depth_first_order(
                sym, seed, directed=False, return_predecessors=True
            )
            seen[order] = True
            components.append((order, pred))

        # component totals + base score (sum of every component's cost)
        comp_cost = []
        comp_stats = []
        base_total = 0.0
        for order, pred in components:
            nC = order.size
            if regression:
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

        # --- Pass 2: evaluate every tree edge with the decomposed score. ---
        best_deletion = deletion(np.nan, np.nan, np.inf)
        for ci, (order, pred) in enumerate(components):
            if order.size <= 1:
                continue  # isolated node: no internal edge to cut
            nC = order.size
            costC = comp_cost[ci]
            subn = np.zeros(N, dtype=np.int64)
            subn[order] = 1
            if regression:
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
                if regression:
                    costA = _rss(subZZ[u], subZy[u], subyy[u])
                    costB = _rss(ZZ_C - subZZ[u], Zy_C - subZy[u], yy_C - subyy[u])
                else:
                    sA = subs[u]
                    sqA = subsq[u]
                    sB = sC - sA
                    sqB = sqC - sqA
                    costA = (sqA - sA * sA / nA).sum()
                    costB = (sqB - sB * sB / nB).sum()
                # only this component's cost changes; others stay in base_total
                score = base_total - costC + costA + costB
                if score < best_deletion.score:
                    edge = (p, u) if (p, u) in stored else (u, p)
                    best_deletion = deletion(edge[0], edge[1], score)
        if make:
            return self.make_cut(*best_deletion, MSF=MSF)
        return best_deletion

    def make_cut(self, in_node, out_node, score, MSF=None):
        """
        Make a cut on the MSF inplace.

        Parameters
        ----------

        in_node : int
            The ID of the source node for the edge to be cut.
        out_node : int
            The ID of the destination node for the edge to be cut.
        score : float
            The value of the score being cut. If the score is
            infinite, the cut is not made.
        MSF : scipy.sparse.csgraph.minimum_spanning_tree (default None)
            The spanning forest to use when making the cut. If not provided,
            uses the default tree in ``self.minimum_spanning_forest_``.

        """
        if MSF is None:
            MSF = self.minimum_spanning_forest_
        if np.isfinite(score):
            MSF[in_node, out_node] = 0
            MSF.eliminate_zeros()
            return (MSF, *cg.connected_components(MSF, directed=False))
        raise OptimizeWarning(
            "Score of the ({},{}) cut is inf, the quorum is likely not met!"
        )
