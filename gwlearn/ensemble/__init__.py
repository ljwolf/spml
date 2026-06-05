from ._geoweighted import (
    GWGradientBoostingClassifier,
    GWGradientBoostingRegressor,
    GWRandomForestClassifier,
    GWRandomForestRegressor
)
from ._spatial_index_ensemble import(
    QuadtreeEnsembleClassifier,
    QuadtreeEnsembleRegressor,
    KDTreeEnsembleClassifier,
    KDTreeEnsembleRegressor
)
from ._spatial_tree_ensemble import SpatialTreeClassifier, SpatialTreeRegressor
from ._spatial_index_boosting import (
    QuadtreeBoostingClassifier,
    KDTreeBoostingClassifier,
    QuadtreeBoostingRegressor,
    KDTreeBoostingRegressor
)