"""
Route-to-leaf spatial ensembles backed by an external tree.

Each page at the chosen depth of an ExternalTreeTransformer (see transform.py)
gets its own local model fit on y directly; prediction routes every point to
its page model.  Boosting variants live in boosting.py.
"""

import numpy as np


class _ConstantClassifier:
    """Predicts a single observed label; used for degenerate single-class pages."""

    def __init__(self, label):
        self.classes_ = np.array([label])

    def predict_proba(self, X):
        return np.ones((len(X), 1))


class SpatialTreeEnsemble:
    """
    Route-to-leaf spatial ensemble backed by any ExternalTreeTransformer.

    Each page at the chosen depth gets its own local model fit on y directly
    (no residual chaining, unlike SpatialTreeBoosting).  Prediction routes
    every point to its page model.

    Parameters
    ----------
    transformer : ExternalTreeTransformer
        A fitted transformer (see transform.py).
    model : sklearn estimator or None
        Cloned for each page.  Defaults to the task-appropriate model.
    depth : int, list of int, or None (default: None)
        Page resolution; None uses the leaf partition.
    min_leaf_size : int (default: 10)
        Pages smaller than this are skipped.
    """

    def __init__(self, transformer, model=None, depth=None, min_leaf_size=10):
        self.transformer   = transformer
        self.model         = model
        self.depth         = depth
        self.min_leaf_size = min_leaf_size

    def _make_default_model(self):
        raise NotImplementedError

    def _pages(self, coordinates):
        enc         = self.transformer.transform(coordinates, depth=self.depth)
        page_labels = np.asarray(enc.argmax(axis=1)).ravel()
        in_page     = np.asarray(enc.sum(axis=1) > 0).ravel()
        return page_labels, in_page

    def fit(self, X, y, coordinates):
        from sklearn.base import clone

        X           = np.asarray(X)
        y           = np.asarray(y)
        coordinates = np.asarray(coordinates)
        model       = self.model if self.model is not None else self._make_default_model()

        page_labels, in_page = self._pages(coordinates)
        self.page_models_ = {}
        for page in np.unique(page_labels[in_page]):
            mask = in_page & (page_labels == page)
            if mask.sum() < self.min_leaf_size:
                continue
            self.page_models_[page] = self._fit_page(clone(model), X[mask], y[mask])
        return self

    def _fit_page(self, model, X, y):
        model.fit(X, y)
        return model


class SpatialTreeRegressor(SpatialTreeEnsemble):
    """Route-to-leaf external-tree ensemble for regression."""

    def _make_default_model(self):
        from sklearn.linear_model import LinearRegression
        return LinearRegression()

    def predict(self, X, coordinates):
        X           = np.asarray(X)
        coordinates = np.asarray(coordinates)
        page_labels, in_page = self._pages(coordinates)
        output = np.full(len(X), np.nan)
        for page, model in self.page_models_.items():
            mask = in_page & (page_labels == page)
            if mask.any():
                output[mask] = model.predict(X[mask])
        return output

    def score(self, X, y, coordinates):
        from sklearn.metrics import mean_squared_error
        return -mean_squared_error(np.asarray(y), self.predict(X, coordinates))


class SpatialTreeClassifier(SpatialTreeEnsemble):
    """
    Route-to-leaf external-tree ensemble for classification.

    Adds predict_proba with class alignment across pages; single-class pages
    predict their constant label.
    """

    def _make_default_model(self):
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression()

    def fit(self, X, y, coordinates):
        self.classes_ = np.unique(np.asarray(y))
        return super().fit(X, y, coordinates)

    def _fit_page(self, model, X, y):
        if len(np.unique(y)) < 2:
            return _ConstantClassifier(y[0])
        model.fit(X, y)
        return model

    def predict_proba(self, X, coordinates):
        X           = np.asarray(X)
        coordinates = np.asarray(coordinates)
        page_labels, in_page = self._pages(coordinates)
        proba = np.zeros((len(X), len(self.classes_)))
        col   = {c: i for i, c in enumerate(self.classes_)}
        for page, model in self.page_models_.items():
            mask = in_page & (page_labels == page)
            if not mask.any():
                continue
            rows  = np.flatnonzero(mask)
            local = model.predict_proba(X[mask])
            for j, c in enumerate(model.classes_):
                proba[rows, col[c]] = local[:, j]
        return proba

    def predict(self, X, coordinates):
        proba = self.predict_proba(X, coordinates)
        out   = self.classes_[proba.argmax(axis=1)]
        unrouted = ~proba.any(axis=1)
        if unrouted.any():
            out = out.astype(object)
            out[unrouted] = None
        return out

    def score(self, X, y, coordinates):
        from sklearn.metrics import accuracy_score
        return accuracy_score(np.asarray(y), self.predict(X, coordinates))
