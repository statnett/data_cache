from data_cache.cache_tools import numpy_cache, pandas_cache, read_metadata  # noqa: F401
import importlib.metadata

__version__ = importlib.metadata.version(__name__)
