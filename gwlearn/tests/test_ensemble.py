import numpy as np
import pandas as pd
import pytest

from gwlearn.ensemble import (
    GWGradientBoostingClassifier,
    GWGradientBoostingRegressor,
    GWRandomForestClassifier,
    GWRandomForestRegressor,
    KDTreeEnsembleRegressor,
    QuadtreeBoostingClassifier,
    QuadtreeEnsembleRegressor,
    SpatialTreeRegressor,
)


def test_gwrf_init():
    """Test GWRandomForestClassifier initialization."""
    model = GWRandomForestClassifier(bandwidth=100)

    # Check default parameters
    assert model.bandwidth == 100
    assert model.fixed is False
    assert model.kernel == "bisquare"
    assert model._model_type == "random_forest"

    # Check custom parameters
    model = GWRandomForestClassifier(
        bandwidth=50, fixed=True, kernel="tricube", n_estimators=100, max_depth=5
    )
    assert model.bandwidth == 50
    assert model.fixed is True
    assert model.kernel == "tricube"
    assert model._model_kwargs["n_estimators"] == 100
    assert model._model_kwargs["max_depth"] == 5


def test_gwrf_fit_basic(sample_data):  # noqa: F811
    """Test that GWRandomForestClassifier fit method works and runs as expected."""
    X, y, geometry = sample_data

    model = GWRandomForestClassifier(
        bandwidth=150000,
        fixed=True,
        random_state=42,
        strict=False,  # To avoid warnings on invariance
        n_estimators=50,  # Use fewer trees for faster testing
        n_jobs=1,
    )

    fitted_model = model.fit(X, y, geometry)

    # Test that fitting works and returns self
    assert fitted_model is model

    # Test specific attributes of GWRandomForestClassifier
    assert hasattr(model, "feature_importances_")
    assert hasattr(model, "oob_y_pooled_")
    assert hasattr(model, "oob_pred_pooled_")

    # Check structure of feature importances
    assert isinstance(model.feature_importances_, pd.DataFrame)
    assert model.feature_importances_.shape[0] == len(X)
    assert model.feature_importances_.shape[1] == X.shape[1]
    assert all(col in model.feature_importances_.columns for col in X.columns)
    pd.testing.assert_series_equal(
        model.feature_importances_.mean(),
        pd.Series(
            [0.33556985859855293, 0.36600269841232724, 0.2984274429891198],
            index=["Crm_prs", "Litercy", "Wealth"],
        ),
        check_exact=False,
        atol=0.01,
    )


def test_gwgb_init():
    """Test GWGradientBoostingClassifier initialization."""
    model = GWGradientBoostingClassifier(bandwidth=100)

    # Check default parameters
    assert model.bandwidth == 100
    assert model.fixed is False
    assert model.kernel == "bisquare"
    assert model._model_type == "gradient_boosting"

    # Check custom parameters
    model = GWGradientBoostingClassifier(
        bandwidth=50,
        fixed=True,
        kernel="tricube",
        n_estimators=100,
        learning_rate=0.1,
        subsample=0.8,
    )
    assert model.bandwidth == 50
    assert model.fixed is True
    assert model.kernel == "tricube"
    assert model._model_kwargs["n_estimators"] == 100
    assert model._model_kwargs["learning_rate"] == 0.1
    assert model._model_kwargs["subsample"] == 0.8


def test_gwgb_fit_basic(sample_data):  # noqa: F811
    """Test that GWGradientBoostingClassifier fit method works as expected."""
    X, y, geometry = sample_data

    model = GWGradientBoostingClassifier(
        bandwidth=150000,
        fixed=True,
        random_state=42,
        strict=False,  # To avoid warnings on invariance
        n_estimators=50,  # Use fewer trees for faster testing
        n_jobs=1,
    )

    fitted_model = model.fit(X, y, geometry)

    # Test that fitting works and returns self
    assert fitted_model is model

    # Test specific attributes of GWGradientBoostingClassifier
    assert hasattr(model, "feature_importances_")

    # Check structure of feature importances
    assert isinstance(model.feature_importances_, pd.DataFrame)
    assert model.feature_importances_.shape[0] == len(X)
    assert model.feature_importances_.shape[1] == X.shape[1]
    assert all(col in model.feature_importances_.columns for col in X.columns)


def test_gwrf_regressor_init():
    """Test GWRandomForestRegressor initialization."""
    model = GWRandomForestRegressor(bandwidth=100)

    # Check default parameters
    assert model.bandwidth == 100
    assert model.fixed is False
    assert model.kernel == "bisquare"
    assert model._model_type == "random_forest"

    # Check custom parameters
    model = GWRandomForestRegressor(
        bandwidth=50, fixed=True, kernel="tricube", n_estimators=100, max_depth=5
    )
    assert model.bandwidth == 50
    assert model.fixed is True
    assert model.kernel == "tricube"
    assert model._model_kwargs["n_estimators"] == 100
    assert model._model_kwargs["max_depth"] == 5


def test_gwrf_regressor_fit_basic(sample_regression_data):  # noqa: F811
    """Test that GWRandomForestRegressor fit method works and runs as expected."""
    X, y, geometry = sample_regression_data

    model = GWRandomForestRegressor(
        bandwidth=150000,
        fixed=True,
        random_state=42,
        strict=False,  # To avoid warnings on invariance
        n_estimators=50,  # Use fewer trees for faster testing
        n_jobs=1,
    )

    fitted_model = model.fit(X, y, geometry)

    # Test that fitting works and returns self
    assert fitted_model is model

    # Test specific attributes of GWRandomForestRegressor
    assert hasattr(model, "feature_importances_")
    assert hasattr(model, "oob_y_pooled_")
    assert hasattr(model, "oob_pred_pooled_")

    # Check structure of feature importances
    assert isinstance(model.feature_importances_, pd.DataFrame)
    assert model.feature_importances_.shape[0] == len(X)
    assert model.feature_importances_.shape[1] == X.shape[1]
    assert all(col in model.feature_importances_.columns for col in X.columns)

    # Check regressor-specific attributes
    assert hasattr(model, "pred_")
    assert hasattr(model, "resid_")
    assert hasattr(model, "RSS_")
    assert hasattr(model, "TSS_")
    assert hasattr(model, "y_bar_")
    assert hasattr(model, "local_r2_")
    # IC-related attributes are not defined for non-linear models
    assert not hasattr(model, "hat_values_")
    assert not hasattr(model, "effective_df_")
    assert not hasattr(model, "log_likelihood_")
    assert not hasattr(model, "aic_")
    assert not hasattr(model, "aicc_")
    assert not hasattr(model, "bic_")


def test_kdtree_validation_fraction(sample_regression_data):  # noqa: F811
    """validation_fraction=0 reproduces the in-sample default; >0 is no less conservative."""
    X, y, geometry = sample_regression_data
    coords = np.c_[geometry.x, geometry.y]

    default = KDTreeEnsembleRegressor(min_leaf_size=5, pbar=False).fit(X, y, coords)
    explicit_zero = KDTreeEnsembleRegressor(
        min_leaf_size=5, pbar=False, validation_fraction=0.0
    ).fit(X, y, coords)
    held_out = KDTreeEnsembleRegressor(
        min_leaf_size=5, pbar=False, validation_fraction=0.25, random_state=0
    ).fit(X, y, coords)

    # explicit 0.0 must be identical to the in-sample default
    np.testing.assert_array_equal(default.labels_, explicit_zero.labels_)
    # holding rows out can only ever reject more splits
    assert held_out.n_leaves <= default.n_leaves


def test_spatialtree_validation_fraction(sample_regression_data):  # noqa: F811
    """validation_fraction=0 reproduces the in-sample default; >0 is no less conservative."""
    from libpysal.graph import Graph

    X, y, geometry = sample_regression_data
    coords = np.c_[geometry.x, geometry.y]
    graph = Graph.build_triangulation(geometry, kernel="identity")

    default = SpatialTreeRegressor(graph=graph, min_leaf_size=5).fit(X, y, coords)
    explicit_zero = SpatialTreeRegressor(
        graph=graph, min_leaf_size=5, validation_fraction=0.0
    ).fit(X, y, coords)
    held_out = SpatialTreeRegressor(
        graph=graph, min_leaf_size=5, validation_fraction=0.25, random_state=0
    ).fit(X, y, coords)

    np.testing.assert_array_equal(default.labels_, explicit_zero.labels_)
    assert held_out.n_leaves <= default.n_leaves


def test_gwgb_regressor_init():
    """Test GWGradientBoostingRegressor initialization."""
    model = GWGradientBoostingRegressor(bandwidth=100)

    # Check default parameters
    assert model.bandwidth == 100
    assert model.fixed is False
    assert model.kernel == "bisquare"
    assert model._model_type == "gradient_boosting"

    # Check custom parameters
    model = GWGradientBoostingRegressor(
        bandwidth=50,
        fixed=True,
        kernel="tricube",
        n_estimators=100,
        learning_rate=0.1,
        subsample=0.8,
    )
    assert model.bandwidth == 50
    assert model.fixed is True
    assert model.kernel == "tricube"
    assert model._model_kwargs["n_estimators"] == 100
    assert model._model_kwargs["learning_rate"] == 0.1
    assert model._model_kwargs["subsample"] == 0.8


def test_gwgb_regressor_fit_basic(sample_regression_data):  # noqa: F811
    """Test that GWGradientBoostingRegressor fit method works as expected."""
    X, y, geometry = sample_regression_data

    model = GWGradientBoostingRegressor(
        bandwidth=150000,
        fixed=True,
        random_state=42,
        strict=False,  # To avoid warnings on invariance
        n_estimators=50,  # Use fewer trees for faster testing
        n_jobs=1,
    )

    fitted_model = model.fit(X, y, geometry)

    # Test that fitting works and returns self
    assert fitted_model is model

    # Test specific attributes of GWGradientBoostingRegressor
    assert hasattr(model, "feature_importances_")

    # Check structure of feature importances
    assert isinstance(model.feature_importances_, pd.DataFrame)
    assert model.feature_importances_.shape[0] == len(X)
    assert model.feature_importances_.shape[1] == X.shape[1]
    assert all(col in model.feature_importances_.columns for col in X.columns)

    # Check regressor-specific attributes
    assert hasattr(model, "pred_")
    assert hasattr(model, "resid_")
    assert hasattr(model, "RSS_")
    assert hasattr(model, "TSS_")
    assert hasattr(model, "y_bar_")
    assert hasattr(model, "local_r2_")
    # IC-related attributes are not defined for non-linear models
    assert not hasattr(model, "hat_values_")
    assert not hasattr(model, "effective_df_")
    assert not hasattr(model, "log_likelihood_")
    assert not hasattr(model, "aic_")
    assert not hasattr(model, "aicc_")
    assert not hasattr(model, "bic_")


def _deep_quadtree_data():
    """Points where a tight cluster forces the quadtree about 30 levels deep."""
    rng = np.random.default_rng(0)
    coords = np.vstack(
        [rng.uniform(0, 1e-9, size=(40, 2)), rng.uniform(0.5, 1, size=(40, 2))]
    )
    X = rng.normal(size=(80, 1))
    return rng, X, coords


def test_quadtree_deep_leaf_labels_are_exact():
    """Leaf labels stay distinct when path codes exceed float64 precision."""
    rng, X, coords = _deep_quadtree_data()
    y = rng.normal(size=80)
    model = QuadtreeEnsembleRegressor(
        split_tolerance=-np.inf, min_leaf_size=2, pbar=False
    ).fit(X, y, coords)

    assert max(leaf.depth for leaf in model.leaves_.values()) > 20

    expected = np.empty(len(y), dtype=object)
    for leaf in model.leaves_.values():
        expected[leaf.ids] = leaf.index
    assert model.labels_.tolist() == expected.tolist()

    assert len(set(model.labels_.tolist())) > 16  # float64 collapsed these to 16

    # routing training points reproduces their fitted leaves and predictions
    assert model.query(coords).tolist() == model.labels_.tolist()
    np.testing.assert_allclose(model.predict(X, coords), model.prediction_)


def test_quadtree_boosting_classifier_deep_leaf_labels():
    """Routing doesn't overflow when path codes exceed int64."""
    rng, X, coords = _deep_quadtree_data()
    y = rng.integers(0, 2, size=80)
    model = QuadtreeBoostingClassifier(
        split_tolerance=-np.inf, min_leaf_size=2, max_depth=60, pbar=False
    ).fit(X, y, coords)

    assert max(leaf.depth for leaf in model.leaves_.values()) > 20

    labels = model._query(coords)
    assert set(labels.tolist()) <= set(model.leaves_)
    assert model.predict(X, coords).shape == (80,)


def test_holds_out():
    from gwlearn.ensemble._spatial_tree_ensemble import _holds_out, _leaf_split

    assert _holds_out(3, 0.0)
    assert _holds_out(10, 0.2)
    assert not _holds_out(10, 0.01)  # rounds to zero held-out rows
    assert not _holds_out(1, 0.9)  # rounds to every row held out
    with pytest.raises(ValueError, match="can't hold out"):
        _leaf_split(np.arange(10), 0.01, np.random.default_rng(0))


@pytest.mark.parametrize("ensemble_cls", [KDTreeEnsembleRegressor, QuadtreeEnsembleRegressor])
def test_index_rejects_split_without_holdout(sample_regression_data, ensemble_cls):  # noqa: F811
    """A child that can't hold rows out is below the quota, so it isn't accepted."""
    X, y, geometry = sample_regression_data
    coords = np.c_[geometry.x, geometry.y]

    in_sample = ensemble_cls(min_leaf_size=5, pbar=False).fit(X, y, coords)
    # every child of the 85-row root holds fewer than 50 rows, so 1% rounds to 0
    no_holdout = ensemble_cls(
        min_leaf_size=5, pbar=False, validation_fraction=0.01, random_state=0
    ).fit(X, y, coords)

    assert in_sample.n_leaves > 1
    assert no_holdout.n_leaves == 1


def test_spatialtree_rejects_cut_without_holdout(sample_regression_data):  # noqa: F811
    """A side that can't hold rows out is below the quota, so no cut is made."""
    from libpysal.graph import Graph

    X, y, geometry = sample_regression_data
    coords = np.c_[geometry.x, geometry.y]
    graph = Graph.build_triangulation(geometry, kernel="identity")

    in_sample = SpatialTreeRegressor(graph=graph, min_leaf_size=5).fit(X, y, coords)
    no_holdout = SpatialTreeRegressor(
        graph=graph, min_leaf_size=5, validation_fraction=0.01, random_state=0
    ).fit(X, y, coords)

    assert in_sample.n_leaves > 1
    assert no_holdout.n_leaves == 1
