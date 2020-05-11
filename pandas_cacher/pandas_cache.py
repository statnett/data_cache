import functools
import hashlib
import inspect
import json
import os
import pathlib
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
        return pd.read_hdf(self, key=key)


def store_factory(data_storer: Type[StoreClass]) -> Type[store_function]:
    """Factory function for creating storing functions for the cache decorator.

    Args:
        data_storer: class with a context manager, and file_path + mode parameters.

    Returns: function for storing tables

    """

    def store_func(
        key: str, func: cache_able_function, f_args: Tuple[Any], f_kwargs: Dict[str, Any],
    ) -> cached_data_type:
        """Retrieves stored data if key exists in stored data if the key is new, retrieves data from
        decorated function & stores the result with the given key.

        Args:
            key: unique key used to retrieve/store data
            func: original cached function
            f_args: args to pass to the function
            f_kwargs: kwargs to pass to the function

        Returns:
            Data retrieved from the store if existing else from function

        """
        file_path = get_path() / "data.h5"
        mode = "r+" if file_path.exists() else "w"
        with data_storer(file_path, mode=mode) as store:
            arrays = [
                store[store_key][:]
                for store_key in store.keys()
                if store_key.split("-")[0].strip("/") == key
            ]
            if arrays:
                return tuple(arrays) if len(arrays) > 1 else arrays[0]
        data = func(*f_args, **f_kwargs)
        with data_storer(file_path, mode=mode) as store:
            if isinstance(data, tuple):
                for i, data_ in enumerate(data):
                    store.create_dataset(f"{key}-data{i}", data=data_)
            else:
                store.create_dataset(key, data=data)
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
                key = (
                    "df"
                    + hashlib.md5(
                        (func.__name__ + json.dumps(full_args)).encode("utf-8")
                    ).hexdigest()
                )
                return table_getter(key, func, f_args, f_kwargs)

            return wrapped

        if orig_func:
            return decorated(orig_func)
        return decorated

    return cache_decorator


pandas_cache = cache_decorator_factory(store_factory(PandasStore))
numpy_cache = cache_decorator_factory(store_factory(h5py.File))
