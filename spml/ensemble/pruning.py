"""
Pruning for fitted spatial index ensembles.

Each function takes a fitted ensemble and returns a pruned copy. The input
ensemble is not changed.
"""

import copy
from collections import defaultdict

import numpy
import pandas

from ._spatial_index_boosting import SpatialIndexBoosting
from ._spatial_index_ensemble import KDTreeEnsemble, SpatialIndexEnsemble


def permutation_prune(estimator, n_permutations=199, alpha=0.05, random_state=None):
    """
    Returns a copy of a fitted spatial index ensemble without its non-significant splits.

    The pruning works from the deepest accepted child up to the root. A child
    whose own accepted children were all pruned is tested against its parent.
    The test statistic is the child's in-sample gain: the reduction in summed
    loss when the child's model replaces the parent's model on the child's
    rows. The reference distribution permutes the locations of the parent's
    rows, so each row keeps its features and outcome but moves to another
    row's location. For each permutation, the index's split rule is applied
    again to the permuted parent and the children are refit. A permuted child
    that violates the depth or size floor has no gain. If the p-value is more
    than `alpha`, the child is pruned, and the parent's model covers the
    child's area. A child with a surviving descendant is kept and is not
    tested.

    The test assumes that the parent's rows are exchangeable when the
    relationship between the features and the outcome doesn't vary inside the
    parent. Residual spatial autocorrelation violates this assumption and
    gives p-values that are too small.

    Parameters
    ----------
    estimator : SpatialIndexEnsemble
        A fitted quadtree or KD-tree ensemble. Boosted ensembles aren't
        supported.
    n_permutations : int (default: 199)
        The number of permutations in each reference distribution.
    alpha : float (default: 0.05)
        The significance level. A child with a p-value above `alpha` is
        pruned.
    random_state : int, numpy.random.Generator or None (default: None)
        Seeds the permutations.

    Returns
    -------
    SpatialIndexEnsemble
        A pruned deep copy of `estimator`. Its ``pruning_`` attribute is a
        DataFrame with one row for each accepted child of the input ensemble:
        ``parent``, ``child``, ``depth``, ``size``, ``gain``, ``p_value``,
        ``tested``, and ``pruned``. ``p_value`` is NaN for an untested child.
    """
    if isinstance(estimator, SpatialIndexBoosting) or not isinstance(
        estimator, SpatialIndexEnsemble
    ):
        raise TypeError(
            "permutation_prune supports quadtree and KD-tree ensembles, "
            f"not {type(estimator).__name__}."
        )
    if not hasattr(estimator, "leaves_"):
        raise ValueError("The estimator must be fitted before it is pruned.")
    if n_permutations < 1:
        raise ValueError("n_permutations must be at least 1.")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")

    rng = numpy.random.default_rng(random_state)
    cells = {c.index: c for level in estimator.levels_.values() for c in level}

    accepted = set()
    for index in estimator.leaves_:
        while index is not None and index not in accepted:
            accepted.add(index)
            index = cells[index].parent
    children = defaultdict(list)
    for index in accepted:
        if cells[index].parent is not None:
            children[cells[index].parent].append(index)

    surviving = set(accepted)
    null_gains = {}
    records = []
    for index in sorted(
        (i for i in accepted if cells[i].parent is not None),
        key=lambda i: cells[i].depth,
        reverse=True,
    ):
        child = cells[index]
        parent = cells[child.parent]
        gain = _gain(parent, child)
        record = dict(
            parent=parent.index, child=index, depth=child.depth,
            size=child.size, gain=gain, p_value=numpy.nan,
            tested=False, pruned=False,
        )
        if not any(c in surviving for c in children[index]):
            if parent.index not in null_gains:
                null_gains[parent.index] = _null_gains(
                    estimator, parent, n_permutations, rng
                )
            null = null_gains[parent.index][:, index - parent.index * 10 - 1]
            record["p_value"] = (1 + numpy.sum(null >= gain)) / (n_permutations + 1)
            record["tested"] = True
            if record["p_value"] > alpha:
                record["pruned"] = True
                surviving.discard(index)
        records.append(record)

    leaves = {
        index
        for index in surviving
        if index in estimator.leaves_
        or not children[index]
        or any(c not in surviving for c in children[index])
    }

    pruned = copy.deepcopy(estimator)
    pruned_cells = {c.index: c for level in pruned.levels_.values() for c in level}
    pruned.leaves_ = {
        index: pruned_cells[index]
        for index in sorted(leaves, key=lambda i: pruned_cells[i].depth)
    }
    root = pruned.levels_[0][0]
    pruned.labels_ = numpy.full(root.size, 0, dtype=object)
    pruned.prediction_ = root.prediction_.copy()
    for leaf in pruned.leaves_.values():
        pruned.labels_[leaf.ids] = leaf.index
        pruned.prediction_[leaf.ids] = leaf.prediction_
    pruned.geoms_["pruned"] = ~pruned.geoms_["index"].isin(leaves)
    pruned.pruning_ = pandas.DataFrame.from_records(
        records,
        columns=["parent", "child", "depth", "size", "gain", "p_value", "tested", "pruned"],
    )
    return pruned


def _gain(parent, child):
    """The reduction in summed in-sample loss when `child` replaces `parent` on the child's rows."""
    parent_score = parent._score_function(
        child.y, parent.prediction_[child.parent_mask]
    )
    return child.size * (child.score_ - parent_score)


def _resplit(estimator, cell):
    """Applies the ensemble's split rule to `cell` without registering the children."""
    if isinstance(estimator, KDTreeEnsemble) and estimator.split_method == "optimal":
        return cell.split(min_leaf_size=estimator.min_leaf_size)
    return cell.split()


def _null_gains(estimator, parent, n_permutations, rng):
    """
    Returns the gains of each child position after the parent's locations are permuted.

    The result has one row for each permutation and one column for each child
    position. A child that violates the depth or size floor has a gain of
    negative infinity. For a KD-tree, this applies to both children, because
    the KD-tree rejects a split if either child violates the floor.
    """
    n_children = 4 if not isinstance(estimator, KDTreeEnsemble) else 2
    gains = numpy.full((n_permutations, n_children), -numpy.inf)
    for b in range(n_permutations):
        shuffled = copy.copy(parent)
        shuffled.coordinates = parent.coordinates[rng.permutation(parent.size)]
        splits = _resplit(estimator, shuffled)
        below = [estimator._below_floor(split) for split in splits]
        if isinstance(estimator, KDTreeEnsemble) and any(below):
            continue
        for position, (split, is_below) in enumerate(zip(splits, below)):
            if not is_below:
                gains[b, position] = _gain(parent, split)
    return gains
