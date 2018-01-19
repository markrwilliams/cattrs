"""Loading of attrs classes."""
import attr
from attr import asdict, astuple, Factory, fields, NOTHING
from hypothesis import assume, given
from hypothesis.strategies import (booleans, composite, data,
                                   frozensets, integers, just, lists,
                                   randoms, sampled_from, sets,
                                   tuples)

from pytest import raises

from typing import (FrozenSet, Mapping, MutableSet, Union, Sequence,
                    Set, Tuple)

from . import (simple_classes, list_types, int_attrs, str_attrs,
               float_attrs, dict_attrs, make_nested_classes)
from cattr import StructuringError
from cattr.converters import (format_seq, format_dict, format_set,
                              format_frozenset, format_attribute)
from cattr._compat import unicode, long


def make_default(defaults, draw, strategy):
    if defaults is True or (defaults is None and draw(booleans())):
        return draw(strategy)
    return NOTHING


def make_type(draw, types, parameters):
    type_ = draw(types)
    maybe_parameter = draw(parameters)
    if maybe_parameter is None:
        return type_
    return type_[maybe_parameter]


@composite
def set_attrs(draw, defaults=None):
    type_ = make_type(draw,
                      types=sampled_from([Set, MutableSet]),
                      parameters=sampled_from([int, None]))
    val_strat = sets(integers())
    default = make_default(defaults, draw, val_strat)
    return (attr.ib(type=type_, default=default), val_strat)


@composite
def frozenset_attrs(draw, defaults=None):
    val_strat = frozensets(integers())
    type_ = make_type(draw,
                      types=just(FrozenSet),
                      parameters=sampled_from([int, None]))
    default = make_default(defaults, draw, val_strat)
    return (attr.ib(type=type_, default=default), val_strat)


@composite
def list_attrs(draw, defaults=None):
    type_ = make_type(draw,
                      types=list_types,
                      parameters=sampled_from([int, None]))
    val_strat = lists(integers())
    default = make_default(defaults, draw, val_strat)
    return (attr.ib(type=type_, default=default), val_strat)


@composite
def tuple_attrs(draw, defaults=None):
    type_ = make_type(draw,
                      types=just(Tuple),
                      parameters=sampled_from([int, None]))
    val_strat = tuples(integers())
    default = make_default(defaults, draw, val_strat)
    return (attr.ib(type=type_, default=default), val_strat)


def attrs_strategy(defaults):
    return (int_attrs(defaults)
            | str_attrs(defaults)
            | float_attrs(defaults)
            | dict_attrs(defaults)
            | frozenset_attrs(defaults)
            | set_attrs(defaults)
            | list_attrs(defaults)
            | tuple_attrs(defaults))


@given(simple_classes())
def test_structure_simple_from_dict(converter, cl_and_vals):
    # type: (Converter, Any) -> None
    """Test structuring non-nested attrs classes dumped with asdict."""
    cl, vals = cl_and_vals
    obj = cl(*vals)

    dumped = asdict(obj)
    loaded = converter.structure(dumped, cl)

    assert obj == loaded


@given(simple_classes(defaults=True), data())
def test_structure_simple_from_dict_default(converter, cl_and_vals, data):
    """Test structuring non-nested attrs classes with default value."""
    cl, vals = cl_and_vals
    obj = cl(*vals)
    attrs_with_defaults = [a for a in fields(cl)
                           if a.default is not NOTHING]
    to_remove = data.draw(lists(elements=sampled_from(attrs_with_defaults),
                                unique=True))

    for a in to_remove:
        if isinstance(a.default, Factory):
            setattr(obj, a.name, a.default.factory())
        else:
            setattr(obj, a.name, a.default)

    dumped = asdict(obj)

    for a in to_remove:
        del dumped[a.name]

    assert obj == converter.structure(dumped, cl)


class Poison(object):
    def __str__(self):
        raise Exception

    def __unicode__(self):
        raise Exception


def break_unstructured(random, unstructured, cls):
    """Make an unstructured class' dictionary un-structruable."""
    choices = []

    NoneType = type(None)

    def _recur(current, ctx, type_):
        if isinstance(type_, type(Union)):
            args = type_.__args__
            next_ctx = ctx
            next_type = (args[0] if args[1] is NoneType else args[1])
            _recur(current, next_ctx, next_type)
        elif issubclass(
                type_,
                (bool, bytes, float, int, long, NoneType, str, unicode)
        ):
            return
        elif issubclass(type_, FrozenSet) and type_.__args__:
            [next_type] = type_.__args__
            for el in current:
                next_ctx = ctx + (format_frozenset(next_type),)
                _recur(el, next_ctx, next_type)
        elif issubclass(type_, Set) and type_.__args__:
            [next_type] = type_.__args__
            for el in current:
                next_ctx = ctx + (format_set(next_type),)
                _recur(el, next_ctx, next_type)
        elif issubclass(type_, Sequence) and type_.__args__:
            [next_type] = type_.__args__
            for i, el in enumerate(current):
                next_ctx = ctx + (format_seq(i, next_type),)
                _recur(el, next_ctx, next_type)
        elif issubclass(type_, Mapping) and type_.__args__:
            [_, next_type] = type_.__args__
            items_ctxes = []
            for key, value in current.items():
                next_ctx = ctx + (format_dict(key, next_type),)
                items_ctxes.append((key, next_ctx))
                _recur(value, next_ctx, next_type)
            choices.append((current, items_ctxes))
        elif attr.has(type_):
            items_ctxes = []
            for a in attr.fields(type_):
                next_ctx = ctx + (format_attribute(a.name, a.type),)
                items_ctxes.append((a.name, next_ctx))
                _recur(current[a.name], next_ctx, a.type)
            choices.append((current, items_ctxes))

    _recur(unstructured, (), cls)

    condemned, items_ctxes = random.choice(choices)
    assume(condemned)

    item, ctx = random.choice(items_ctxes)
    condemned[item] = Poison()

    return ctx


@given(make_nested_classes(attrs_strategy), randoms(), data())
def test_structure_from_dict_with_errors(
        contextualizing_converter, cl_and_vals, random, data,
):
    # type: (Converter, Any, Any, Any) -> None
    """``StructureError``s ``ctx`` attribute refers to the point in
    the source where an error occurred.  On Python 3,
    ``StructureError``s are chained to the originating error.
    """
    cl, vals = cl_and_vals

    unstructured = attr.asdict(cl(*vals))
    assume(unstructured)
    ctx = break_unstructured(random, unstructured, cl)
    with raises(StructuringError) as exc_info:
        contextualizing_converter.structure(unstructured, cl)
    assert exc_info.value.ctx == ctx


@given(simple_classes())
def test_roundtrip(converter, cl_and_vals):
    # type: (Converter, Any) -> None
    """We dump the class, then we load it."""
    cl, vals = cl_and_vals
    obj = cl(*vals)

    dumped = converter.unstructure(obj)
    loaded = converter.structure(dumped, cl)

    assert obj == loaded


@given(simple_classes())
def test_structure_tuple(converter, cl_and_vals):
    # type: (Converter, Any) -> None
    """Test loading from a tuple, by registering the loader."""
    cl, vals = cl_and_vals
    converter.register_structure_hook(cl, converter.structure_attrs_fromtuple)
    obj = cl(*vals)

    dumped = astuple(obj)
    loaded = converter.structure(dumped, cl)

    assert obj == loaded


@given(simple_classes(defaults=False), simple_classes(defaults=False))
def test_structure_union(converter, cl_and_vals_a, cl_and_vals_b):
    """Structuring of automatically-disambiguable unions works."""
    # type: (Converter, Any, Any) -> None
    cl_a, vals_a = cl_and_vals_a
    cl_b, vals_b = cl_and_vals_b
    a_field_names = {a.name for a in fields(cl_a)}
    b_field_names = {a.name for a in fields(cl_b)}
    assume(a_field_names)
    assume(b_field_names)

    common_names = a_field_names & b_field_names
    if len(a_field_names) > len(common_names):
        obj = cl_a(*vals_a)
        dumped = asdict(obj)
        res = converter.structure(dumped, Union[cl_a, cl_b])
        assert isinstance(res, cl_a)
        assert obj == res


@given(simple_classes(), simple_classes())
def test_structure_union_explicit(
        converter, hook_wrapper, cl_and_vals_a, cl_and_vals_b
):
    """Structuring of manually-disambiguable unions works."""
    # type: (Converter, Any, Any) -> None
    cl_a, vals_a = cl_and_vals_a
    cl_b, vals_b = cl_and_vals_b

    @hook_wrapper
    def dis(obj, _):
        return converter.structure(obj, cl_a)

    converter.register_structure_hook(Union[cl_a, cl_b], dis)

    inst = cl_a(*vals_a)

    assert inst == converter.structure(converter.unstructure(inst),
                                       Union[cl_a, cl_b])
