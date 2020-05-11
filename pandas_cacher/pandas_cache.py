import functools
import hashlib
import inspect
import json
import os
import pathlib
from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, Tuple, Type, Union

import h5py
import numpy as np
import pandas as pd

pandas_function = Callable[..., Union[Tuple[pd.DataFrame], pd.DataFrame]]
numpy_function = Callable[..., Union[Tuple[np.ndarray], np.ndarray]]
cached_data_type = Union[Tuple[Any], Any]
cache_able_function = Callable[..., cached_data_type]
store_function = Callable[[str, Callable[..., Any], Tuple[Any], Dict[str, Any]], Any]


def get_path() -> pathlib.Path:
    cache_path = os.environ.get("CACHE_PATH", "")
    cache_path = pathlib.Path.cwd() if cache_path == "" else pathlib.Path(cache_path)
    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path


class StoreClass:
    def __init__(self, file_path: str, mode: str):
        raise NotImplementedError

    def __enter__(self):
        raise NotImplementedError

    def __exit__(self, exc_type, exc_val, exc_tb):
        raise NotImplementedError

    def keys(self) -> Iterable:
        raise NotImplementedError

    def create_dataset(self, key: str, data: ...) -> None:
        raise NotImplementedError

    def __getitem__(self, key: str) -> ...:
        raise NotImplementedError


class PandasStore(pd.HDFStore):

    def create_dataset(self, key: str, data: pd.DataFrame) -> None:
        data.to_hdf(self, key)

    def __getitem__(self, key: str) -> pd.DataFrame:
        dfs = [pd.read_hdf(self, key=k) for k in self.keys() if key in k]
        return tuple(dfs) if len(dfs) > 1 else dfs[0]


def add_metadata():
    pass

def store_factory(data_storer: Type[StoreClass]) -> Type[store_function]:
    """Factory function for creating storing functions for the cache decorator.

    Args:
        data_storer: class with a context manager, and file_path + mode parameters.

    Returns: function for storing tables

    """

    def store_func(
        func_key: str,
        arg_key: str,
        func: cache_able_function,
        f_args: Tuple[Any],
        f_kwargs: Dict[str, Any],
        metadata: dict = None,
        group_metadata: dict = None,
    ) -> cached_data_type:
        """Retrieves stored data if key exists in stored data if the key is new, retrieves data from
        decorated function & stores the result with the given key.

        Args:
            arg_key: unique key used to retrieve/store data
            func: original cached function
            f_args: args to pass to the function
            f_kwargs: kwargs to pass to the function

        Returns:
            Data retrieved from the store if existing else from function

        """
        file_path = get_path() / "data.h5"
        path = f"/{func_key}/{arg_key}"
        with data_storer(file_path, mode="a") as store:
            if store.__contains__(path):
                data = store[path]
                if isinstance(data, h5py.Group):
                    return tuple([store[f"{path}/{data_idx}"][:] for data_idx in data.keys()])
                return data[:]
        data = func(*f_args, **f_kwargs)
        with data_storer(file_path, mode="a") as store:
            if isinstance(data, tuple):
                for i, data_ in enumerate(data):
                    store.create_dataset(f"{path}/data{i}", data=data_)
            else:
                store.create_dataset(path, data=data)
            return data
    return store_func


def cache_decorator_factory(table_getter: Type[store_function]) -> Type[cache_able_function]:
    # pylint: disable=keyword-arg-before-vararg
    def cache_decorator(
        orig_func: cache_able_function = None, *args: str
    ) -> Type[cache_able_function]:
        if isinstance(orig_func, str):
            args = list(args) + [orig_func]
            orig_func = None

        def decorated(func: cache_able_function) -> Type[cache_able_function]:
            @functools.wraps(func)
            def wrapped(*f_args: Tuple[Any], **f_kwargs: Dict[str, Any]) -> cached_data_type:
                """Hashes function arguments to a unique key, and uses the key to store/retrieve
                data from the configured store.

                Args:
                    *f_args: Arguments passed along to the function
                    **f_kwargs: Keyword-Arguments passed along to the function

                Returns: Stored data if existing, else result from the function

                """
                if os.environ.get("DISABLE_CACHE", "FALSE") == "TRUE":
                    return func(*f_args, **f_kwargs)
                argspec = inspect.getfullargspec(func)
                defaults = (
                    dict(zip(argspec.args[::-1], argspec.defaults[::-1]))
                    if argspec.defaults
                    else {}
                )
                kw_defaults = argspec.kwonlydefaults if argspec.kwonlydefaults else {}
                full_args = {
                    **kw_defaults,
                    **defaults,
                    **f_kwargs,
                    **dict(zip(argspec.args, f_args)),
                    **{"arglist": f_args[len(argspec.args) :]},
                }
                full_args = full_args if not args else {arg: full_args[arg] for arg in args}
                full_args.pop("self", "")
                full_args = {k: str(v) for k, v in full_args.items()}
                group = "a" + hashlib.md5(inspect.getsource(func).encode("utf-8")).hexdigest()
                key = "a" + hashlib.md5(json.dumps(full_args).encode("utf-8")).hexdigest()
                return table_getter(group, key, func, f_args, f_kwargs)

            return wrapped

        if orig_func:
            return decorated(orig_func)
        return decorated

    return cache_decorator


pandas_cache = cache_decorator_factory(store_factory(PandasStore))
numpy_cache = cache_decorator_factory(store_factory(h5py.File))
