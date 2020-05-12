# Table cache

Works by hashing the combinations of arguments of a function call with
the function name to create a unique id of a table retrieval.  If
the function call is new the original function will be called, and the
resulting tables(s) will be stored in a HDFStore indexed by the
hashed key.  Next time the function is called with the same args the
tables(s) will be retrieved from the store instead of executing the
function.

The hashing of the arguments is done by first applying str() on the
argument, and then taking th md5 hash of the combination of these args
together with the function name.  This means that if a argument for
some reason does not have a str representation the key generation will
fail.  To omit this issue one can specify which arguments the cache
should consider such that 'un-stringable' arguments are skipped.  This
functionality is also used for skipping arguments the should by design
not be considered for the key-generation like for example
database-clients.


#### Setting cache file location

The module automatically creates a `cache/data.h5` relative to
`__main__`, to change this set the environment variable
`CACHE_PATH` to be the desired directory of the `data.h5` file.

#### Disabling the cache with env-variable

To disable the cache set the environment variable
`DISABLE_CACHE` to `TRUE`.

### Usage

#### Decorating functions

```python
from pandas_cacher import pandas_cache
from time import sleep
from datetime import datetime
import pandas as pd

@pandas_cache
def simple_func():
    sleep(5)
    return pd.DataFrame([[1,2,3], [2,3,4]])


t0 = datetime.now()
print(simple_func())
print(datetime.now() - t0)

t0 = datetime.now()
print(simple_func())
print(datetime.now() - t0)
```
```commandline
   0  1  2
0  1  2  3
1  2  3  4
0:00:05.343027
   0  1  2
0  1  2  3
1  2  3  4
0:00:00.015987
```

#### Decorating class methods

The decorator ignores arguments named 'self' such that it will work across different instances of the same object.

```python
from pandas_cacher import pandas_cache
from time import sleep
from datetime import datetime
import pandas as pd


class PandasClass:
    def __init__(self):
        print(self)

    @pandas_cache
    def simple_func(self):
        sleep(5)
        return pd.DataFrame([[1,2,3], [2,3,4]])

c = PandasClass()
t0 = datetime.now()
print(c.simple_func())
print(datetime.now() - t0)

c = PandasClass()
t0 = datetime.now()
print(c.simple_func())
print(datetime.now() - t0)
```
```commandline
<__main__.PandasClass object at 0x003451F0>
   0  1  2
0  1  2  3
1  2  3  4
0:00:05.375342
<__main__.PandasClass object at 0x124814B0>
   0  1  2
0  1  2  3
1  2  3  4
0:00:00.014959
```

#### Selecting arguments

```python
from pandas_cacher import pandas_cache
from time import sleep
from datetime import datetime
import pandas as pd

@pandas_cache("a", "c")
def simple_func(a, b, c=True):
    sleep(5)
    return pd.DataFrame([[1,2,3], [2,3,4]])


t0 = datetime.now()
print(simple_func(a=1, b=2))
print(datetime.now() - t0)

# b is not considered
t0 = datetime.now()
print(simple_func(a=1, b=3))
print(datetime.now() - t0)
```
```commandline
   0  1  2
0  1  2  3
1  2  3  4
0:00:05.619620
   0  1  2
0  1  2  3
1  2  3  4
0:00:00.017980
```

#### Multi-DataFrame returns

```python
from pandas_cacher import pandas_cache
from time import sleep
from datetime import datetime
import pandas as pd


@pandas_cache("a", "c")
def simple_func(a, *args, **kwargs):
    sleep(5)
    return pd.DataFrame([[1,2,3], [2,3,4]]), pd.DataFrame([[1,2,3], [2,3,4]]) * 10


t0 = datetime.now()
print(simple_func(1, b=2, c=True))
print(datetime.now() - t0)

t0 = datetime.now()
print(simple_func(a=1, b=3, c=True))
print(datetime.now() - t0)
```
```commandline
(   0  1  2
0  1  2  3
1  2  3  4,     0   1   2
0  10  20  30
1  20  30  40)
0:00:05.368545
(   0  1  2
0  1  2  3
1  2  3  4,     0   1   2
0  10  20  30
1  20  30  40)
0:00:00.019578
```

#### Disabling cache for tests

Caching can be disabled using the environment variable DISABLE_CACHE to TRUE

```python
from mock import patch
def test_cached_function():
    with patch.dict("os.environ", {"DISABLE_PANDAS_CACHE": "TRUE"}, clear=True):
        assert cached_function() == target
```

#### Numpy caching

```python
from pandas_cacher import numpy_cache
from time import sleep
from datetime import datetime
import numpy as np


@numpy_cache("a", "c")
def simple_func(a, *args, **kwargs):
    sleep(5)
    return np.array([[1, 2, 3], [2, 3, 4]]), np.array([[1, 2, 3], [2, 3, 4]]) * 10


t0 = datetime.now()
print(simple_func(1, b=2, c=True))
print(datetime.now() - t0)

t0 = datetime.now()
print(simple_func(a=1, b=3, c=True))
print(datetime.now() - t0)
```

```commandline
(array([[1, 2, 3],
       [2, 3, 4]]), array([[10, 20, 30],
       [20, 30, 40]]))
0:00:05.009084
(array([[1, 2, 3],
       [2, 3, 4]]), array([[10, 20, 30],
       [20, 30, 40]]))
0:00:00.002000
```

#### Metadata

Metadata is automatically stored with the data on the group node containing the
DataFrame/Array.

```python
from pandas_cacher import numpy_cache, pandas_cache, read_metadata
import pandas as pd
import numpy as np
from datetime import datetime


@pandas_cache
def function1(a, *args, b=1, **kwargs):
    return pd.DataFrame()

@numpy_cache
def function2(a, *args, b=1, **kwargs):
    return np.array([])

function1(1, True, datetime.date(2019, 11, 11))
function2(2, False, b=2, c=1.1)
read_metadata("path_to_data.h5")
```
results:
```json
[{
    "/a86f0a323bf20998b5deda81e9f90bb49/a5d320e5dcdc5d3f35a4ca366980b2dc1": {
        "a": "1",
        "arglist": "(True, datetime.date(2019, 11, 11))",
        "b": "1",
        "date_stored": "01/05/2020, 10:00:00",
        "function_name": "function1",
        "module_path": "path_to_module"
    },
    "/a56ad8af46bc5fd8b9320b00b12e6c115/a62734531fc99855292c9db04d5eba60a": {
        "a": "2",
        "arglist": "(False,)",
        "b": "2",
        "c": "1.1",
        "date_stored": "01/05/2020, 10:00:00",
        "function_name": "function2",
        "module_path":  "path_to_module"
    }
}]
```
