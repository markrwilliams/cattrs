import sys

version_info = sys.version_info[0:3]
is_py2 = version_info[0] == 2
is_py3 = version_info[0] == 3

if is_py2:
    from functools32 import lru_cache
    from singledispatch import singledispatch
    unicode = unicode  # noqa
    bytes = str
    long = long        # noqa

    def raise_from(value, from_value):
        raise value

else:
    from functools import lru_cache, singledispatch  # noqa
    unicode = str
    bytes = bytes
    long = int

    exec("""def raise_from(value, from_value):
    try:
        raise value from from_value
    finally:
        value = None
    """)
