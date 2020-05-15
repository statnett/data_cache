import functools
import hashlib
import inspect
import json
import os
import pathlib
from datetime import datetime
from typing import Any, Callable, Dict, List, Tuple, Type, Union

import h5py
import numpy as np
import pandas as pd
import tables

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


class StoreClass:  # pragma: no cover
    def __init__(self, file_path: str, mode: str):
        raise NotImplementedError

    def __enter__(self):
        raise NotImplementedError

    def __exit__(self, exc_type, exc_val, exc_tb):
        raise NotImplementedError

    def __contains__(self, path: str) -> bool:
        raise NotImplementedError

    def create_dataset(self, key: str, data: ...) -> None:
        raise NotImplementedError

    def __getitem__(self, key: str) -> ...:
        raise NotImplementedError


class PandasGroup:
    def __init__(self, store: "PandasStore", path: str):
        self.store = store
        self.path = path

    @property
    def attrs(self):
        return self.store.get_storer(self.path).attrs

    def __getitem__(self, key: str) -> Union[pd.DataFrame, Tuple[pd.DataFrame]]:
        dfs = [pd.read_hdf(self.store, key=k) for k in self.store.keys() if self.path in k]
        return tuple(dfs) if len(dfs) > 1 else dfs[0]


class PandasStore(pd.HDFStore):
    def create_dataset(self, key: str, data: pd.DataFrame) -> None:
        data.to_hdf(self, key)

    def __getitem__(self, key: str) -> PandasGroup:
        return PandasGroup(self, key)


def add_metadata(
    group: Union[h5py.Group, PandasGroup],
    func: cache_able_function,
    metadata: Dict[str, str] = None,
):
    metadata = metadata if metadata is not None else {}
    group.attrs["metadata_function_name"] = func.__name__
    group.attrs["metadata_module_path"] = inspect.getfile(func)
    group.attrs["metadata_date_stored"] = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    for k, v in metadata.items():
        group.attrs[f"metadata_{k}"] = v


def read_metadata(path: str) -> Dict[str, Dict[str, str]]:
    with tables.open_file(path, mode="r") as file:
        return {
            group._v_pathname: {
                attr.replace("metadata_", ""): group._v_attrs[attr]
                for attr in group._v_attrs._f_list()
                if attr.startswith("metadata")
            }
            for group in file.root._f_walk_groups()
            if not len(group._v_groups)
        }


def extract_args(
    func: cache_able_function, args: List[str], f_args: Tuple[Any], f_kwargs: Dict[str, Any]
) -> Dict[str, Any]:
    """Extracts arguments names from the function and pairs them with the corresponding values in a
    dictionary.

    Args:
        func:
        args: Selected arguments to extract from the function
        f_args: Arguments passed along to the function
        f_kwargs: Keyword-Arguments passed along to the function

    Returns: Dict like: {"argname": arg}

    """
    argspec = inspect.getfullargspec(func)
    defaults = dict(zip(argspec.args[::-1], argspec.defaults[::-1])) if argspec.defaults else {}
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
    return full_args


def store_factory(data_storer: Type[StoreClass],) -> Type[store_function]:
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
        metadata: Dict[str, str] = None,
    ) -> cached_data_type:
        """Retrieves stored data if key exists in stored data if the key is new, retrieves data from
        decorated function & stores the result with the given key.

        Args:
            func_key: unique key generated from function source
            arg_key: unique key generated from function arguments
            func: original cached function
            f_args: args to pass to the function
            f_kwargs: kwargs to pass to the function
            metadata: dictionary of metadata data to store alongside the data

        Returns:
            Data retrieved from the store if existing else from function

        """
        file_path = get_path() / "data.h5"
        path = f"/{func_key}/{arg_key}"
        suffix = "/array" if issubclass(data_storer, h5py.File) else ""
        with data_storer(file_path, mode="a") as store:
            if store.__contains__(path):
                if isinstance(store[path], h5py.Group) and "array" not in store[path].keys():
                    return tuple(
                        [store[f"{path}/{data_idx}{suffix}"][:] for data_idx in store[path].keys()]
                    )
                return store[f"{path}{suffix}"][:]
        data = func(*f_args, **f_kwargs)
        with data_storer(file_path, mode="a") as store:
            if isinstance(data, tuple):
                for i, data_ in enumerate(data):
                    store.create_dataset(f"{path}/data{i}{suffix}", data=data_)
                    add_metadata(store[f"{path}/data{i}"], func, metadata)
            else:
                store.create_dataset(f"{path}{suffix}", data=data)
                add_metadata(store[path], func, metadata)
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
                extracted_args = extract_args(func, args, f_args, f_kwargs)
                extracted_args = {k: str(v) for k, v in extracted_args.items()}
                group = "a" + hashlib.md5(inspect.getsource(func).encode("utf-8")).hexdigest()
                key = "a" + hashlib.md5(json.dumps(extracted_args).encode("utf-8")).hexdigest()
                return table_getter(group, key, func, f_args, f_kwargs, extracted_args)

            return wrapped

        if orig_func:
            return decorated(orig_func)
        return decorated

    return cache_decorator


pandas_cache = cache_decorator_factory(store_factory(PandasStore))
numpy_cache = cache_decorator_factory(store_factory(h5py.File))
