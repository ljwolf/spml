"""
Bagged spatial-index ensembles.

Bootstrap-resample the training rows, fit an independent spatial-index ensemble
on each resample, and combine the members' predictions.  Two axes control the
combination:

``weighting`` -- where the per-member weight comes from
    - ``"uniform"`` : every member weighs the same (the plain bagging baseline).
    - ``"tree"``    : one scalar weight per member, from that member's overall
      out-of-bag skill.
    - ``"leaf"``    : a per-point weight, from the out-of-bag skill of the leaf
      each point routes into within each member.

``reduction`` -- how the weighted member predictions collapse to one number
    - ``"softmax"`` : weighted mean (the weights are a softmax over skill).
    - ``"median"``  : weighted median (robust to a badly-extrapolating member).

Skill is always evaluated **out-of-bag**: the rows a member's bootstrap sample
omitted are routed through that fitted member, so the member (and each of its
leaves) is scored on data it never trained on.  All of this is computed here in
the bag -- the underlying ensemble is not modified, and its in-sample ``score_``
is left untouched.  Scoring out-of-bag, after growth, is neither a per-split tax
nor an ensemble-level moving target: each finished member is scored once.

The score -> weight map is a softmax whose scores are first normalized across the
competing members (subtract the column max, divide by the column spread) so the
``temperature`` is scale-free and transfers across regression / classification /
reconstruction losses.  ``temperature`` -> infinity recovers uniform weights;
``temperature`` -> 0 approaches a per-competitor argmax.
"""

import copy
import warnings

import numpy


def _softmax_weights(skill, valid, temperature):
    """
    Softmax weights over the member axis (axis 0), computed per column.

    Parameters
    ----------
    skill : ndarray of shape (n_members, n_points)
        Higher is better.  Entries where ``valid`` is False are ignored.
    valid : ndarray of bool, shape (n_members, n_points)
        Which (member, point) entries have a usable prediction and skill.
    temperature : float
        Softmax temperature (> 0).  Large -> uniform; small -> argmax.

    Returns
    -------
    weights : ndarray of shape (n_members, n_points)
        Non-negative, summing to 1 down axis 0 for every column with at least
        one valid member, and all-zero for columns with none.
    """
    skill = numpy.asarray(skill, dtype=float)
    masked = numpy.where(valid, skill, numpy.nan)

    with warnings.catch_warnings():
        # all-NaN columns (a point outside every member) are handled explicitly
        warnings.simplefilter("ignore", RuntimeWarning)
        col_max = numpy.nanmax(masked, axis=0, keepdims=True)
        col_spread = numpy.nanstd(masked, axis=0, keepdims=True)
    col_max = numpy.where(numpy.isnan(col_max), 0.0, col_max)
    # a single valid competitor (or all-equal skill) has zero spread -> use 1 so
    # the normalization is a no-op and the softmax falls back to uniform.
    col_spread = numpy.where((col_spread > 0) & numpy.isfinite(col_spread),
                             col_spread, 1.0)

    z = (masked - col_max) / (col_spread * temperature)
    z = numpy.clip(z, -700.0, 700.0)  # guard exp overflow for tiny temperatures
    e = numpy.exp(z)
    e = numpy.where(valid, e, 0.0)

    denom = e.sum(axis=0, keepdims=True)
    return numpy.divide(e, denom, out=numpy.zeros_like(e), where=denom > 0)


def _weighted_mean(predictions, weights):
    """Weighted mean down axis 0; NaN predictions carry zero weight."""
    filled = numpy.where(numpy.isnan(predictions), 0.0, predictions)
    out = (weights * filled).sum(axis=0)
    any_weight = weights.sum(axis=0) > 0
    return numpy.where(any_weight, out, numpy.nan)


def _weighted_median(predictions, weights):
    """
    Weighted median down axis 0.

    For each column, sort the members by predicted value and return the value at
    which the cumulative weight first reaches half of the column's total weight
    (the lower weighted median).  NaN predictions sort last and carry zero
    weight, so they never move the crossing.  Columns with no valid member yield
    NaN.
    """
    n_points = predictions.shape[1]
    order = numpy.argsort(
        numpy.where(numpy.isnan(predictions), numpy.inf, predictions), axis=0
    )
    sorted_pred = numpy.take_along_axis(predictions, order, axis=0)
    sorted_w = numpy.take_along_axis(weights, order, axis=0)

    cum = numpy.cumsum(sorted_w, axis=0)
    total = cum[-1:, :]
    reached = cum >= (0.5 * total)
    idx = numpy.argmax(reached, axis=0)
    out = sorted_pred[idx, numpy.arange(n_points)]
    return numpy.where(total.ravel() > 0, out, numpy.nan)


class BaggedSpatialIndexEnsemble:
    """
    Bag of spatial-index ensembles with skill-aware aggregation.

    Parameters
    ----------
    base_estimator : SpatialIndexEnsemble
        An *unfitted* spatial-index ensemble (e.g. ``KDTreeEnsembleRegressor`` or
        ``QuadtreeEnsembleRegressor``).  It is deep-copied for each bootstrap
        member, so its per-node ``model`` and all growth settings are inherited.
        The original instance is never fitted or mutated.
    n_estimators : int (default: 25)
        Number of bootstrap members.
    weighting : {"uniform", "tree", "leaf"} (default: "uniform")
        Source of the per-member weights (see module docstring).
    reduction : {"softmax", "median"} (default: "softmax")
        How weighted predictions collapse to one value.
    temperature : float (default: 1.0)
        Softmax temperature for ``weighting`` in {"tree", "leaf"}.  Ignored for
        "uniform".  Can be overridden per call in :meth:`predict`.
    random_state : int, numpy.random.Generator or None
        Seeds the bootstrap resampling.
    pbar : bool (default: False)
        Forwarded to each member (forced off during fitting regardless).

    Attributes
    ----------
    members_ : list of fitted ensembles
    tree_skill_ : ndarray of shape (n_estimators,)
        Out-of-bag skill of each member (NaN if it had no routable OOB rows).
    leaf_skill_ : list of dict
        Per member, ``{leaf_index: out-of-bag skill}``.  Leaves with no OOB rows
        fall back to their in-sample ``score_`` (a warning is emitted once).
    bounds_ : list
        The shared ``[min_x, min_y, max_x, max_y]`` root box used by every member.
    """

    _VALID_WEIGHTING = {"uniform", "tree", "leaf"}
    _VALID_REDUCTION = {"softmax", "median"}

    def __init__(
        self,
        base_estimator,
        n_estimators=25,
        weighting="uniform",
        reduction="softmax",
        temperature=1.0,
        random_state=None,
        pbar=False,
    ):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.weighting = weighting
        self.reduction = reduction
        self.temperature = temperature
        self.random_state = random_state
        self.pbar = pbar

    # -- fitting ------------------------------------------------------------

    def fit(self, X, y, coordinates):
        """
        Fit ``n_estimators`` bootstrap members and record their OOB skill.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        coordinates : array-like of shape (n_samples, 2)

        Returns
        -------
        self
        """
        if self.weighting not in self._VALID_WEIGHTING:
            raise ValueError(
                f"weighting must be one of {sorted(self._VALID_WEIGHTING)}, "
                f"got {self.weighting!r}"
            )
        if self.reduction not in self._VALID_REDUCTION:
            raise ValueError(
                f"reduction must be one of {sorted(self._VALID_REDUCTION)}, "
                f"got {self.reduction!r}"
            )

        X = numpy.asarray(X.values if hasattr(X, "values") else X)
        y = numpy.asarray(y.values if hasattr(y, "values") else y).squeeze()
        coordinates = numpy.asarray(coordinates)

        rng = numpy.random.default_rng(self.random_state)
        n = len(y)
        # One root box shared by every member.  Bootstrap resamples can miss the
        # extreme corner points; without a shared box each member's root -- and,
        # for the quadtree, every geographic-midpoint split -- would differ, so
        # the members would not be comparable/aggregable.
        self.bounds_ = [*numpy.min(coordinates, axis=0),
                        *numpy.max(coordinates, axis=0)]

        self.members_ = []
        self.tree_skill_ = numpy.full(self.n_estimators, numpy.nan)
        self.leaf_skill_ = []
        n_fallback = 0
        n_leaves_total = 0

        for i in range(self.n_estimators):
            idx = rng.integers(0, n, n)
            in_bag = numpy.zeros(n, dtype=bool)
            in_bag[idx] = True
            oob = ~in_bag

            member = copy.deepcopy(self.base_estimator)
            member.bounds = list(self.bounds_)
            member.pbar = False
            member.fit(X[idx], y[idx], coordinates=coordinates[idx])
            self.members_.append(member)

            tree_skill, leaf_skill, fell_back = self._oob_skill(
                member, X[oob], y[oob], coordinates[oob]
            )
            self.tree_skill_[i] = tree_skill
            self.leaf_skill_.append(leaf_skill)
            n_fallback += fell_back
            n_leaves_total += len(member.leaves_)

        if n_fallback:
            warnings.warn(
                f"{n_fallback} of {n_leaves_total} leaves had no out-of-bag "
                "rows; used their in-sample score_ as the skill weight. "
                "Consider fewer/larger leaves or more estimators if this is a "
                "large fraction.",
                stacklevel=2,
            )
        return self

    def _oob_skill(self, member, X_oob, y_oob, coords_oob):
        """
        Out-of-bag tree- and leaf-level skill for one fitted member.

        Returns ``(tree_skill, leaf_skill, n_fallback_leaves)``.  Everything is
        computed from the member's public surface (``query``, ``leaves_``,
        ``model_``, ``_score_function``); the member is not modified.
        """
        score = member._score_function
        leaf_skill = {}
        fell_back = 0

        if len(y_oob) == 0:
            # No OOB rows at all (essentially impossible for a real bootstrap).
            for name, leaf in member.leaves_.items():
                leaf_skill[name] = leaf.score_
                fell_back += 1
            return numpy.nan, leaf_skill, fell_back

        labels = member.query(coords_oob)
        pred = member.predict(X_oob, coords_oob)
        routed = ~numpy.isnan(pred)
        tree_skill = (
            score(y_oob[routed], pred[routed]) if routed.any() else numpy.nan
        )

        for name, leaf in member.leaves_.items():
            mask = labels == name
            if mask.any() and leaf.model_ is not None:
                leaf_skill[name] = score(
                    y_oob[mask], leaf.model_.predict(X_oob[mask])
                )
            else:
                # empty OOB leaf (or degenerate local fit): fall back to the
                # leaf's in-sample score_ so it still carries a weight.
                leaf_skill[name] = leaf.score_
                fell_back += 1

        return tree_skill, leaf_skill, fell_back

    # -- prediction ---------------------------------------------------------

    def _member_matrices(self, X, coordinates):
        """
        Stack member predictions and (for leaf weighting) routed-leaf skills.

        Returns ``(predictions, skill)``, both of shape (n_members, n_points).
        ``skill`` is None for uniform weighting, tree-broadcast for tree
        weighting, and per-point for leaf weighting.
        """
        preds = []
        skills = [] if self.weighting == "leaf" else None
        for i, member in enumerate(self.members_):
            preds.append(member.predict(X, coordinates))
            if self.weighting == "leaf":
                labels = member.query(coordinates)
                table = self.leaf_skill_[i]
                skills.append(
                    numpy.array([table.get(l, numpy.nan) for l in labels])
                )
        predictions = numpy.vstack(preds)

        if self.weighting == "uniform":
            skill = None
        elif self.weighting == "tree":
            skill = numpy.repeat(
                self.tree_skill_[:, None], predictions.shape[1], axis=1
            )
        else:  # leaf
            skill = numpy.vstack(skills)
        return predictions, skill

    def predict(self, X, coordinates, temperature=None):
        """
        Predict by aggregating the bootstrap members.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        coordinates : array-like of shape (n_samples, 2)
        temperature : float or None
            Override the constructor temperature for this call (skill and members
            do not depend on it, so a sweep needs no refit).  Ignored for
            ``weighting="uniform"``.

        Returns
        -------
        ndarray of shape (n_samples,)
        """
        X = numpy.asarray(X.values if hasattr(X, "values") else X)
        coordinates = numpy.asarray(coordinates)
        temp = self.temperature if temperature is None else temperature
        if self.weighting != "uniform" and temp <= 0:
            raise ValueError("temperature must be > 0 (use a small value for "
                             "near-argmax behaviour)")

        predictions, skill = self._member_matrices(X, coordinates)
        valid = ~numpy.isnan(predictions)
        if self.weighting == "uniform":
            skill = numpy.zeros_like(predictions)
        else:
            valid &= ~numpy.isnan(skill)
        weights = _softmax_weights(skill, valid, temp)

        if self.reduction == "softmax":
            return _weighted_mean(predictions, weights)
        return _weighted_median(predictions, weights)

    def score(self, X, y, coordinates):
        """Score predictions with the base estimator's score function."""
        y = numpy.asarray(y.values if hasattr(y, "values") else y).squeeze()
        return self.base_estimator._score_function(
            y, self.predict(X, coordinates)
        )

    @property
    def n_leaves(self):
        """Total leaves across all bootstrap members."""
        return sum(m.n_leaves for m in self.members_)
