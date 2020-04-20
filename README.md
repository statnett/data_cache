# Pandas cache

Works by hashing the combinations of arguments of a function call with
the function name to create a unique id of a DataFrame retrieval.  If
the function call is new the original function will be called, and the
resulting DataFrame(s) will be stored in a HDFStore indexed by the
hashed key.  Next time the function is called with the same args the
DataFrame(s) will be retrieved from the store instead of executing the
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
`PANDAS_CACHE_PATH` to be the desired directory of the `data.h5` file.

#### Disabling the cache with env-variable

To disable the pandas cache set the environment variable
`DISABLE_PANDAS_CACHE` to `TRUE`.

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
    return pd.DataFrame([[1,2,3], [2,3,4]]), \
           pd.DataFrame([[1,2,3], [2,3,4]]) * 10


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
