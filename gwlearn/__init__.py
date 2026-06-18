import contextlib
from importlib.metadata import PackageNotFoundError, version

from . import base, ensemble, linear_model, search, transform
from .transform import ExternalTreeTransformer

with contextlib.suppress(PackageNotFoundError):
    __version__ = version("gwlearn")
