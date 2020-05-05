import datetime
import os
import tempfile
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pathos.multiprocessing as mp
from pandas.testing import assert_frame_equal

from pandas_cacher import numpy_cache, pandas_cache


def test_pd_cache():
    df_getter = Mock()
    df_getter.return_value = pd.DataFrame([[1, 2, 3], [2, 3, 4]])

    @pandas_cache("a", "b", "c")
    def pandas_getter(a, b, *args, c=False, **kwargs):
        return df_getter()

    @pandas_cache
    def pandas_getter_clean(a, b):
        return df_getter()

    class ClassFunc:
        @pandas_cache
        def pandas_getter(self, a, b, *args):
            return df_getter()

        @pandas_cache
        def pandas_getter_2(self, a, b, *args):
            return df_getter()

    c = ClassFunc()

    with tempfile.TemporaryDirectory() as d:
        with patch.dict("os.environ", {"CACHE_PATH": str(d)}, clear=True):

            df1 = pandas_getter_clean(1, 32)
            df2 = pandas_getter_clean(1, 32)
            assert_frame_equal(df1, df2)
            df_getter.assert_called_once()
            pandas_getter(1, 32, 3, c=True)
            pandas_getter(1, 32, 4, c=True)
            assert 2 == df_getter.call_count

            df_getter.reset_mock()

            date = datetime.datetime(2019, 1, 1)
            c.pandas_getter(date, 2)
            c.pandas_getter(date, 2)
            c.pandas_getter(date, 2, [1, 2, 3])
            c.pandas_getter_2(date, 2, [1, 2, 3])
            assert 3 == df_getter.call_count
            c.pandas_getter(1, 2, 3)
            c.pandas_getter(1, 2, 4, 5)
            assert 5 == df_getter.call_count

            df_getter.reset_mock()

            os.environ["DISABLE_CACHE"] = "TRUE"
            pandas_getter_clean(1, 2)
            pandas_getter_clean(1, 2)
            assert 2 == df_getter.call_count


def test_np_cache():
    array_getter = Mock()
    array_getter.return_value = np.array([[1, 2, 3], [2, 3, 4]])

    @numpy_cache("a", "b", "c")
    def numpy_getter(a, b, *args, c=False, **kwargs):
        return array_getter()

    @numpy_cache
    def numpy_getter_clean(a, b):
        return array_getter()

    class ClassFunc:
        @numpy_cache
        def numpy_getter(self, a, b, *args):
            return array_getter()

        @numpy_cache
        def numpy_getter_2(self, a, b, *args):
            return array_getter()

    c = ClassFunc()

    with tempfile.TemporaryDirectory() as d:
        with patch.dict("os.environ", {"CACHE_PATH": str(d)}, clear=True):

            a1 = numpy_getter_clean(1, 32)
            a2 = numpy_getter_clean(1, 32)
            np.testing.assert_equal(a1, a2)
            array_getter.assert_called_once()
            numpy_getter(1, 32, 3, c=True)
            numpy_getter(1, 32, 4, c=True)
            assert 2 == array_getter.call_count

            array_getter.reset_mock()

            date = datetime.datetime(2019, 1, 1)
            c.numpy_getter(date, 2)
            c.numpy_getter(date, 2)
            c.numpy_getter(date, 2, [1, 2, 3])
            c.numpy_getter_2(date, 2, [1, 2, 3])
            assert 3 == array_getter.call_count
            c.numpy_getter(1, 2, 3)
            c.numpy_getter(1, 2, 4, 5)
            assert 5 == array_getter.call_count

            array_getter.reset_mock()

            os.environ["DISABLE_CACHE"] = "TRUE"
            numpy_getter_clean(1, 2)
            numpy_getter_clean(1, 2)
            assert 2 == array_getter.call_count


def test_multiple_pd_cache():
    df_getter = Mock()
    df_getter.return_value = (
        pd.DataFrame([[1, 2, 3], [2, 3, 4]]),
        pd.DataFrame([[1, 2, 3], [2, 3, 4]]) + 1,
        pd.DataFrame([[1, 2, 3], [2, 3, 4]]) + 2,
    )

    @pandas_cache("a", "b", "c")
    def pandas_getter(a, b, *args, c=False, **kwargs):
        return df_getter()

    class ClassFunc:
        @pandas_cache
        def pandas_getter(self, a, b, *args):
            return df_getter()

        @pandas_cache
        def pandas_getter_2(self, a, b, *args):
            return df_getter()

    c = ClassFunc()

    with tempfile.TemporaryDirectory() as d:
        with patch.dict("os.environ", {"CACHE_PATH": str(d)}, clear=True):

            df1, df2, df3 = pandas_getter(1, 2)
            df11, df12, df13 = pandas_getter(1, 2)

            assert_frame_equal(df1, df11)
            assert_frame_equal(df2, df12)
            assert_frame_equal(df3, df13)

            pandas_getter(1, 2, d=[1, 2, 3])
            df_getter.assert_called_once()
            pandas_getter(1, 2, 3, c=True)
            pandas_getter(1, 2, 4, c=True)
            assert 2 == df_getter.call_count

            df_getter.reset_mock()

            date = datetime.datetime(2019, 1, 1)
            c.pandas_getter(date, 2)
            c.pandas_getter(date, 2)
            c.pandas_getter(date, 2, [1, 2, 3])
            c.pandas_getter_2(date, 2, [1, 2, 3])
            assert 3 == df_getter.call_count
            c.pandas_getter(1, 2, 3)
            c.pandas_getter(1, 2, 4, 5)
            assert 5 == df_getter.call_count


def test_multiple_np_cache():
    array_getter = Mock()
    array_getter.return_value = (
        np.array([[1, 2, 3], [2, 3, 4]]),
        np.array([[1, 2, 3], [2, 3, 4]]) + 1,
        np.array([[1, 2, 3], [2, 3, 4]]) + 2,
    )

    @numpy_cache("a", "b", "c")
    def numpy_getter(a, b, *args, c=False, **kwargs):
        return array_getter()

    class ClassFunc:
        @numpy_cache
        def numpy_getter(self, a, b, *args):
            return array_getter()

        @numpy_cache
        def numpy_getter_2(self, a, b, *args):
            return array_getter()

    c = ClassFunc()

    with tempfile.TemporaryDirectory() as d:
        with patch.dict("os.environ", {"CACHE_PATH": str(d)}, clear=True):

            df1, df2, df3 = numpy_getter(1, 2)
            df11, df12, df13 = numpy_getter(1, 2)

            np.testing.assert_equal(df1, df11)
            np.testing.assert_equal(df2, df12)
            np.testing.assert_equal(df3, df13)

            numpy_getter(1, 2, d=[1, 2, 3])
            array_getter.assert_called_once()
            numpy_getter(1, 2, 3, c=True)
            numpy_getter(1, 2, 4, c=True)
            assert 2 == array_getter.call_count

            array_getter.reset_mock()

            date = datetime.datetime(2019, 1, 1)
            c.numpy_getter(date, 2)
            c.numpy_getter(date, 2)
            c.numpy_getter(date, 2, [1, 2, 3])
            c.numpy_getter_2(date, 2, [1, 2, 3])
            assert 3 == array_getter.call_count
            c.numpy_getter(1, 2, 3)
            c.numpy_getter(1, 2, 4, 5)
            assert 5 == array_getter.call_count


def test_pathos():
    def df_getter(*args, **kwargs):
        return pd.DataFrame([[1, 2, 3], [4, 5, 6]])

    @pandas_cache("a", "b", "c")
    def pandas_multi_getter(a, b, *args, c=False, **kwargs):
        pool = mp.Pool(processes=8)
        return (pool.apply(df_getter), pool.apply(df_getter), pool.apply(df_getter))

    @pandas_cache("a", "b", "c")
    def pandas_getter(a, b, *args, c=False, **kwargs):
        return df_getter()

    with tempfile.TemporaryDirectory() as d:
        with patch.dict("os.environ", {"CACHE_PATH": str(d)}, clear=True):

            pandas_multi_getter(1, 2)
            pandas_getter(1, 2)
