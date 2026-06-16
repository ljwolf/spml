import warnings, numpy, pandas
from sklearn.base import clone

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
            self.prediction_ = numpy.ones_like(y) * numpy.nan

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
    def split(self, boost=False):
            min_x, min_y, max_x, max_y = self.bounds
            if abs(max_x - min_x)>abs(max_y-min_y):
                # in classification, assigns even prediction misses to each half
                # in regression, assigns even residuals to each half
                mid_x = weighted_median(self.coordinates[:,0], weights=numpy.abs(self.residuals_))
                plane = [(mid_x, min_y), (mid_x, max_y)]
            else:
                mid_y = weighted_median(self.coordinates[:,1], weights=numpy.abs(self.residuals_))
                plane = [(min_x, mid_y), (max_x, mid_y)]
            return self._split_by_plane(plane, boost=boost)

class _OptimalSplitNode(_Node):
    def split(self, boost=False, min_leaf_size=10):
        min_x, min_y, max_x, max_y = self.bounds
        if abs(max_x - min_x)>abs(max_y-min_y):
            # in classification, assigns even prediction misses to each half
            # in regression, assigns even residuals to each half
            def objective(t):
                plane = [(t, min_y), (t, max_y)]
                left, right = self._split_by_plane(plane, boost=boost)
                if (left.size < min_leaf_size) | (right.size < min_leaf_size):
                    return numpy.inf
                return -(left.score_+right.score_)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mid_x = type(min_x)(optimize.minimize(
                    objective,
                    bounds=[(min_x, max_x)],
                    x0=[weighted_median(self.coordinates[:,0], weights=numpy.abs(self.residuals_))]
                    ).x.item()
                    )

            plane = [(mid_x, min_y), (mid_x, max_y)]
        else:
            def objective(t):
                plane = [(min_x, t), (max_x, t)]
                left, right = self._split_by_plane(plane, boost=boost)
                if (left.size < min_leaf_size) | (right.size < min_leaf_size):
                    return numpy.inf
                return -(left.score_ + right.score_)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mid_y = type(min_x)(optimize.minimize(
                    objective,
                    bounds=[(min_y, max_y)],
                    x0=[weighted_median(self.coordinates[:,1], weights=numpy.abs(self.residuals_))]
                    ).x.item()
                    )

            plane = [(min_x, mid_y), (max_x, mid_y)]
        return self._split_by_plane(plane, boost=boost)

def weighted_median(x, weights):
    x = numpy.asarray(x).ravel()
    weights = numpy.asarray(weights).ravel()
    df = pandas.DataFrame.from_dict(dict(data=x, weight=weights))
    df.sort_values("data", inplace=True)
    cutpoint = df.weight.sum() / 2
    wsums = df.weight.cumsum()
    return df.data[wsums >= cutpoint].iloc[0]
