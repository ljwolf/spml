import warnings, numpy, pandas
from scipy import optimize
from sklearn import linear_model, metrics
from sklearn.base import clone, is_classifier

class _Cell:
    
    def __init__(
        self,
        index,
        parent,
        parent_mask,
        coordinates,
        bounds,
        X,
        y,
        ids,
        score_function,
        model,
        pruned=None,
    ):
        """
        Class to contain cells at each level of a Quadtree_Regression object.
        This class should not be instantiated directly by users.

        index : int
            index of the cell, recording parent relations from left to right.
        parent : int
            index of the cell's parent; matches `index` up to the last digit.
        parent_mask : numpy.ndarray
            boolean array that can be used to slice this cell's data from
            parent.X, parent.y, and parents.coordinates
        coordinates : numpy.ndarray
            locations of coordinates used in the cell. Must be two-dimensional,
            with the second dimension being "2"
        bounds : numpy.ndarray
            the cell's bounding box
        X : numpy.ndarray
            feature matrix containing observations that fall within the cell
        y : numpy.ndarray
            outcome array containing observations that fall within the cell
        ids : numpy.ndarray
            indices of points in the cell. These indices relate to the *full data*,
            not the indices of the parent's data.
        score_function : callable
            function used to calculate the score of the cell's model. negative mean
            squared error used by default.
        model : class instance
            instance of a scikit-learn model supporting both fit and
            predict methods. Should not generally cache global data,
            since a model is instantiated at each node in the tree.
        pruned : bool (default: None)
            whether the leaf has been pruned in the tree. If None, the pruning
            for the cell has not been evaluated. If True, the cell is
            considered "pruned", and should not be used for prediction
            or fitting.
        """
        self.index = index
        self.parent = parent
        self.parent_mask = parent_mask
        self.depth = len(str(int(index)).split(".")[0]) if index>0 else 0
        self.ids = ids
        self.coordinates = coordinates
        self.bounds = bounds
        self.size = X.shape[0]
        self.X = X
        self.y = y
        self.pruned = pruned
        try:
            self._score_function = score_function
            self._model = model
            self.model_ = self._model.fit(self.X, self.y)
            self.prediction_ = model.predict(self.X)
        except ValueError:
            self._score_function = score_function
            self._model = model
            self.model_ = None
            self.prediction_ = numpy.full(len(y), numpy.nan, dtype=float)

    @property
    def residuals_(self):
        """Construct on demand: self.prediction_ or self.y get overridden in boosting"""
        return self.y - self.prediction_

    @property
    def score_(self):
        """Construct on demand: self.prediction_ or self.y get overridden in boosting"""
        if (len(self.y) == 0) or (len(self.prediction_) == 0):
            return -numpy.inf
        if self.model_ is None:
            # degenerate fit (e.g. single-class child): no usable prediction
            return -numpy.inf
        return self._score_function(self.y, self.prediction_)

    def split(self, boost=False):
        """
        Split a cell into its four constituent cells
        """
        min_x, min_y, max_x, max_y = self.bounds
        # mid_x,mid_y = numpy.median(self.coordinates, axis=0) #not quite quadtree, since splitting based on median location
        mid_x, mid_y = min_x + (max_x - min_x) / 2, min_y + (max_y - min_y) / 2

        # Z-shaped! 0: bigx & bigy, 1: smallx & bigy, 2: bigx & smally, 3: smallx & smally
        labels = (self.coordinates[:, 0] <= mid_x) + (
            self.coordinates[:, 1] <= mid_y
        ) * 2

        bounds = [
            [mid_x, mid_y, max_x, max_y],
            [min_x, mid_y, mid_x, max_y],
            [mid_x, min_y, max_x, mid_y],
            [min_x, min_y, mid_x, mid_y],
        ]
        children = []
        if boost:
            y = self.residuals_
        else:
            y = self.y
        for i in range(4):
            mask = labels == i
            ids = self.ids[mask]
            cell = _Cell(
                index=self.index * 10 + (i + 1),
                parent=self.index,
                ids=ids,
                parent_mask=mask,
                coordinates=self.coordinates[mask,],
                bounds=bounds[i],
                X=self.X[mask],
                y=y[mask],
                score_function=self._score_function,
                model=clone(self._model),
            )
            if boost:
                # self.y is always reset to the actual data after boosting
                # which is necessary to propagate downwards. But, the preceeding
                # boost check before splitting ensures that the residuals are
                # used to train when boosting, and the y/preds are constructed
                # correctly afterwards.
                cell.y = self.y[mask]
                cell.prediction_ += self.prediction_[mask]
                cell.boosted = True
            children.append(cell)
        return children


    def _child_positions(self, coordinates, children):
        """
        Returns the position of the child that holds each point.

        Applies the midpoint rule from `split`, so a point on a midpoint line
        goes to the child below or to the left of the line. `children` maps
        each child position to its cell.
        """
        min_x, min_y, max_x, max_y = self.bounds
        mid_x, mid_y = min_x + (max_x - min_x) / 2, min_y + (max_y - min_y) / 2
        return (coordinates[:, 0] <= mid_x) + (coordinates[:, 1] <= mid_y) * 2

    def query(self, coordinates):
        """
        Query a cell to determine if coordinates in the input
        lie within the cell's geographical boundaries.
        """
        x,y = coordinates.T
        min_x, min_y, max_x, max_y = self.bounds
        in_xrange = (min_x <= x) & (x <= max_x)
        in_yrange = (min_y <= y) & (y <= max_y)
        return in_xrange*in_yrange


class _Node(_Cell):
    def _child_positions(self, coordinates, children):
        """
        Returns the position of the child that holds each point.

        Recovers the plane from the children's bounds and applies the rule
        from `_split_by_plane`, so a point on the plane goes to child 0.
        `children` maps each child position to its cell.
        """
        low, high = children[0].bounds, children[1].bounds
        if high[0] == low[2]:
            return (coordinates[:, 0] > low[2]).astype(int)
        return (coordinates[:, 1] > low[3]).astype(int)

    def _split_by_plane(self, plane, boost=False):
        min_x, min_y, max_x, max_y = self.bounds
        if plane[0][0] == plane[1][0]: # vertical, left-right split
            labels = self.coordinates[:,0] > plane[0][0]
        else: # horizontal, up/down split
            labels = self.coordinates[:,1] > plane[0][1]
        bounds = [
            [min_x, min_y, *plane[1]],
            [*plane[0], max_x, max_y]
        ]

        children = []
        if boost:
            y = self.residuals_
        else:
            y = self.y
        for i, bbox in enumerate(bounds):
            mask = labels == i
            ids = self.ids[mask]
            cell = type(self)(
                # binary index by depth
                index=self.index * 10 + (i + 1),
                parent=self.index,
                ids=ids,
                parent_mask=mask,
                coordinates=self.coordinates[mask,],
                bounds=bbox,
                X=self.X[mask],
                y=y[mask],
                score_function=self._score_function,
                model=clone(self._model),
            )
            if boost:
                # self.y is always reset to the actual data after boosting
                # which is necessary to propagate downwards. But, the preceeding
                # boost check before splitting ensures that the residuals are
                # used to train when boosting, and the y/preds are constructed
                # correctly afterwards.
                cell.y = self.y[mask]
                cell.prediction_ += self.prediction_[mask]
                cell.boosted = True
            children.append(cell)
        return children

    def split(self, boost=False):
        min_x, min_y, max_x, max_y = self.bounds
        if abs(max_x - min_x)>abs(max_y-min_y):
            mid_x = numpy.median(self.coordinates[:,0])
            plane = [(mid_x, min_y), (mid_x, max_y)]
        else:
            mid_y = numpy.median(self.coordinates[:,1])
            plane = [(min_x, mid_y), (max_x, mid_y)]
        return self._split_by_plane(plane, boost=boost)

class _RandomSplitNode(_Node):
    def split(self, boost=False, frac=1):
        min_x, min_y, max_x, max_y = self.bounds
        n_in_node = len(self.y)
        take = numpy.ceil(frac*n_in_node).astype(int)
        # allow replacement iff the take is bigger than the leaf
        ixs = numpy.random.choice(numpy.arange(n_in_node), take, replace=take>n_in_node)
        if abs(max_x - min_x)>abs(max_y-min_y):
            mid_x = numpy.median(self.coordinates[ixs,0])
            plane = [(mid_x, min_y), (mid_x, max_y)]
        else:
            mid_y = numpy.median(self.coordinates[ixs,1])
            plane = [(min_x, mid_y), (max_x, mid_y)]
        return self._split_by_plane(plane, boost=boost)

class _WeightedMedianSplitNode(_Node):
    def _split_weights(self):
        # regression: residual magnitude. classification: soft error
        # 1 - P(true class), the probabilistic analogue of |y - y_hat|.
        # |y - prediction_| would be the label-encoding distance, which is
        # meaningless for non-binary or non-numeric labels.
        if not is_classifier(self._model):
            return numpy.abs(self.residuals_)
        if self.model_ is None:
            return numpy.ones(self.size)
        proba = self.model_.predict_proba(self.X)
        col = {c: j for j, c in enumerate(self.model_.classes_)}
        idx = numpy.array([col[c] for c in self.y])
        return 1 - proba[numpy.arange(self.size), idx]

    def split(self, boost=False):
            min_x, min_y, max_x, max_y = self.bounds
            weights = self._split_weights()
            if abs(max_x - min_x)>abs(max_y-min_y):
                # in regression, assigns even residuals to each half
                mid_x = weighted_median(self.coordinates[:,0], weights=weights)
                plane = [(mid_x, min_y), (mid_x, max_y)]
            else:
                mid_y = weighted_median(self.coordinates[:,1], weights=weights)
                plane = [(min_x, mid_y), (max_x, mid_y)]
            return self._split_by_plane(plane, boost=boost)

class _OptimalSplitNode(_Node):
    def split(self, boost=False, min_leaf_size=10):
        min_x, min_y, max_x, max_y = self.bounds
        if abs(max_x - min_x)>abs(max_y-min_y):
            # in regression, assigns even residuals to each half
            def objective(t):
                plane = [(t, min_y), (t, max_y)]
                left, right = self._split_by_plane(plane, boost=boost)
                if (left.size < min_leaf_size) | (right.size < min_leaf_size):
                    return numpy.inf
                return -(left.size*left.score_ + right.size*right.score_)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mid_x = type(min_x)(optimize.minimize_scalar(
                    objective,
                    bounds=(min_x, max_x),
                    method="bounded",
                    ).x
                    )

            plane = [(mid_x, min_y), (mid_x, max_y)]
        else:
            def objective(t):
                plane = [(min_x, t), (max_x, t)]
                left, right = self._split_by_plane(plane, boost=boost)
                if (left.size < min_leaf_size) | (right.size < min_leaf_size):
                    return numpy.inf
                return -(left.size*left.score_ + right.size*right.score_)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mid_y = type(min_x)(optimize.minimize_scalar(
                    objective,
                    bounds=(min_y, max_y),
                    method="bounded",
                    ).x
                    )

            plane = [(min_x, mid_y), (max_x, mid_y)]
        return self._split_by_plane(plane, boost=boost)

_SCAN_GRID_SIZE = 50


def _neg_mean_squared_error(y_true, y_pred):
    """Default split score. Module-level (not a closure) so fitted ensembles pickle."""
    return -metrics.mean_squared_error(y_true, y_pred)


def _scan_is_exact(model, score_function):
    """True if the scan can use the exact OLS prefix-sum path; false otherwise."""
    return (
        type(model) is linear_model.LinearRegression
        and getattr(model, "fit_intercept", True)
        and not getattr(model, "positive", False)
        and score_function is _neg_mean_squared_error
    )


class _ScanSplitNode(_Node):
    def split(self, boost=False, min_leaf_size=10):
        """
        Split the longer axis at the admissible plane with the lowest summed
        child loss.

        A plane is admissible if both children hold more than `min_leaf_size`
        observations. Planes fall midway between distinct sorted coordinates,
        so tied coordinates always stay on the same side. For an intercept OLS
        model scored by mean squared error, every admissible plane is scored
        exactly from prefix sums of the normal equations. For any other model
        or score, `_SCAN_GRID_SIZE` admissible planes, evenly spaced in rank,
        are scored by refitting the children. If no plane is admissible, the
        cell splits at its coordinate median, and the size floor rejects the
        split.
        """
        min_x, min_y, max_x, max_y = self.bounds
        axis = 0 if abs(max_x - min_x) > abs(max_y - min_y) else 1
        order = numpy.argsort(self.coordinates[:, axis], kind="stable")
        values = self.coordinates[order, axis]

        # k = number of points left of the plane, between values[k-1] and values[k]
        m = min_leaf_size + 1
        k = numpy.arange(m, self.size - m + 1)
        k = k[values[k - 1] < values[k]]

        if k.size == 0:
            threshold = numpy.median(values)
        elif _scan_is_exact(self._model, self._score_function):
            y = self.residuals_ if boost else self.y
            k_best = k[numpy.argmin(self._ols_split_rss(order, y, k))]
            threshold = (values[k_best - 1] + values[k_best]) / 2
        else:
            grid = k[numpy.unique(
                numpy.linspace(0, k.size - 1, min(_SCAN_GRID_SIZE, k.size)).round()
            ).astype(int)]
            thresholds = (values[grid - 1] + values[grid]) / 2
            losses = [
                self._refit_split_loss(self._plane(axis, t), boost)
                for t in thresholds
            ]
            threshold = thresholds[numpy.argmin(losses)]
        return self._split_by_plane(self._plane(axis, threshold), boost=boost)

    def _plane(self, axis, threshold):
        min_x, min_y, max_x, max_y = self.bounds
        if axis == 0:
            return [(threshold, min_y), (threshold, max_y)]
        return [(min_x, threshold), (max_x, threshold)]

    def _ols_split_rss(self, order, y, k):
        """Summed OLS residual sum of squares of both children for each left size k."""
        # centering leaves each child's intercept-OLS residuals unchanged
        X = self.X[order] - self.X.mean(axis=0)
        y = y[order] - y.mean()
        Z = numpy.column_stack([numpy.ones(len(y)), X])
        ZZ = numpy.cumsum(Z[:, :, None] * Z[:, None, :], axis=0)
        Zy = numpy.cumsum(Z * y[:, None], axis=0)
        yy = numpy.cumsum(y * y)

        def rss(ZZ_k, Zy_k, yy_k):
            beta = numpy.einsum("kij,kj->ki", numpy.linalg.pinv(ZZ_k), Zy_k)
            return yy_k - numpy.einsum("ki,ki->k", Zy_k, beta)

        left = rss(ZZ[k - 1], Zy[k - 1], yy[k - 1])
        right = rss(ZZ[-1] - ZZ[k - 1], Zy[-1] - Zy[k - 1], yy[-1] - yy[k - 1])
        return left + right

    def _refit_split_loss(self, plane, boost):
        left, right = self._split_by_plane(plane, boost=boost)
        return -(left.size * left.score_ + right.size * right.score_)


def weighted_median(x, weights):
    x = numpy.asarray(x).ravel()
    weights = numpy.asarray(weights).ravel()
    df = pandas.DataFrame.from_dict(dict(data=x, weight=weights))
    df.sort_values("data", inplace=True)
    cutpoint = df.weight.sum() / 2
    wsums = df.weight.cumsum()
    return df.data[wsums >= cutpoint].iloc[0]
