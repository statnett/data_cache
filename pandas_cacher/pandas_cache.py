import functools
import hashlib
import inspect
import json
import os
import pathlib
from collections import defaultdict
from typing import Any, Callable, Tuple, Union

import pandas as pd

pandas_function = Callable[..., Union[Tuple[pd.DataFrame], pd.DataFrame]]


def get_path() -> pathlib.Path:
    cache_path = os.environ.get("PANDAS_CACHE_PATH", "")
    cache_path = pathlib.Path.cwd() if cache_path == "" else pathlib.Path(cache_path)
    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path


def get_df_hdf(
    key: str, func: pandas_function, f_args: Any, f_kwargs: Any
) -> Union[Tuple[pd.DataFrame], pd.DataFrame]:
    """Retrieves the DataFrames from the HDFStore if the key exists,
    else run the function then store & return the resulting DataFrames.

    Args:
        key: Unique str hash of function call
        func: Wrapped function, should return a DataFrame or tuple of them.
        f_args: Arguments passed along to the function
        f_kwargs: Keyword-Arguments passed along to the function

    Returns: DataFrames that func would originally return.

    """
    file_path = get_path() / "data.h5"
    mode = "r+" if file_path.exists() else "w"
    with pd.HDFStore(file_path, mode=mode) as store:
        keys = defaultdict(list)
        for s_key in store.keys():
            keys[s_key.split("/")[1]].append(s_key)
        if key in keys.keys():
            dfs = [pd.read_hdf(store, key=key_) for key_ in keys[key]]
            return tuple(dfs) if len(dfs) > 1 else dfs[0]
        df = func(*f_args, **f_kwargs)
        if isinstance(df, tuple):
            for i, df_ in enumerate(df):
                df_.to_hdf(store, key=f"{key}/df{i}")
        else:
            df.to_hdf(store, key=key)
        return df


def pandas_cache(orig_func: pandas_function = None, *args: str) -> pandas_function:
    """Decorator for caching function calls that return pandas DataFrames.

    Args:
        *args: arguments of the function to use as filename
        **kwargs: keyword-arguments of the function to use as filename


    Returns: decorated function

    """
    if isinstance(orig_func, str):
        args = list(args) + [orig_func]
        orig_func = None

    def decorated(func: pandas_function) -> pandas_function:
        """Wrapper of function that returns pandas DataFrames.

        Args:
            func: function to be wrapped, should return a DataFrame or tuple of them.

        Returns: wrapped function

        """

        @functools.wraps(func)
        def wrapped(*f_args: ..., **f_kwargs: ...) -> Union[Tuple[pd.DataFrame], pd.DataFrame]:
            """ Hashes function arguments to a unique key, and uses the key
            to store/retrieve DataFrames from the HDFStore.

            Args:
                *f_args: Arguments passed along to the function
                **f_kwargs: Keyword-Arguments passed along to the function

            Returns: DataFrame(s)

            """
            if os.environ.get("DISABLE_PANDAS_CACHE", "FALSE") == "TRUE":
                return func(*f_args, **f_kwargs)
            argspec = inspect.getfullargspec(func)
            defaults = (
                dict(zip(argspec.args[::-1], argspec.defaults[::-1])) if argspec.defaults else {}
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
                + hashlib.md5((func.__name__ + json.dumps(full_args)).encode("utf-8")).hexdigest()
            )
            return get_df_hdf(key, func, f_args, f_kwargs)

        return wrapped

    if orig_func:
        return decorated(orig_func)
    return decorated
