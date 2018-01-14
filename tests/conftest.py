import functools

import platform

import pytest

from hypothesis import HealthCheck, settings

from cattr import Converter


@pytest.fixture(params=[{}, {"contextualize_structure_errors": True}])
def converter(request):
    # type: Any -> Converter
    return Converter(**request.param)


@pytest.fixture()
def contextualizing_converter():
    """A ``Converter`` that always raises ``StructuringError``s."""
    return Converter(contextualize_structure_errors=True)


@pytest.fixture()
def hook_wrapper(request):
    """ Return a decorator for structuring hooks that hides the third
    ``ctx`` parameter when the ``converter`` has been configured with
    ``contextualize_structure_errors``.
    """
    converter = request.getfixturevalue('converter')
    if converter._contextualize_structure_errors:
        def hook_wrapper(f):
            @functools.wraps(f)
            def wrapper(a, b, ctx=()):
                return f(a, b)
            return wrapper
    else:
        def hook_wrapper(f):
            return f

    return hook_wrapper


if platform.python_implementation() == 'PyPy':
    settings.default.suppress_health_check.append(HealthCheck.too_slow)
