"""Loading of attrs classes."""
import attr
from attr import asdict, astuple, Factory, fields, NOTHING
from hypothesis import assume, given
from hypothesis.strategies import (data, lists, randoms, sampled_from)

from pytest import raises

from typing import (Any, FrozenSet, Mapping, Union, Sequence, Set)

from . import (simple_classes, int_attrs, str_attrs, float_attrs,
               dict_attrs, make_nested_classes, tuple_attrs,
               set_attrs, list_attrs, frozenset_attrs)
from cattr import StructuringError
from cattr.converters import (format_seq, format_dict, format_set,
                              format_frozenset, format_attribute)
from cattr._compat import unicode, long


def primitive_attrs(defaults):
    return (int_attrs(defaults)
            | str_attrs(defaults)
            | float_attrs(defaults))


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


class TestContextualizingConverterStructuringErrors(object):
    """
    Tests ``StructuringErrors`` with a contextualizing ``Converter``.
    """

    def break_unstructured(self, random, unstructured, cls):
        """Make an unstructured class' dictionary un-structruable."""
        choices = []

        NoneType = type(None)

        def first_param(type_):
            return Any if not type_.__args__ else type_.__args__[0]

        def recur(current, ctx, type_):
            if isinstance(type_, type(Union)):
                args = type_.__args__
                next_ctx = ctx
                next_type = (args[0] if args[1] is NoneType else args[1])
                recur(current, next_ctx, next_type)
            elif isinstance(type_, type(Any)) or issubclass(
                    type_,
                    (bool, bytes, float, int, long, NoneType, str, unicode)
            ):
                return
            elif issubclass(type_, FrozenSet) and type_.__args__:
                next_type = first_param(type_)
                for el in current:
                    next_ctx = ctx + (format_frozenset(next_type),)
                    recur(el, next_ctx, next_type)
            elif issubclass(type_, Set) and type_.__args__:
                next_type = first_param(type_)
                for el in current:
                    next_ctx = ctx + (format_set(next_type),)
                    recur(el, next_ctx, next_type)
            elif issubclass(type_, Sequence) and type_.__args__:
                next_type = first_param(type_)
                for i, el in enumerate(current):
                    next_ctx = ctx + (format_seq(i, next_type),)
                    recur(el, next_ctx, next_type)
            elif issubclass(type_, Mapping) and type_.__args__:
                [_, next_type] = type_.__args__ if type_.__args__ else [None,
                                                                        Any]
                items_ctxes = []
                for key, value in current.items():
                    next_ctx = ctx + (format_dict(key, next_type),)
                    items_ctxes.append((key, next_ctx))
                    recur(value, next_ctx, next_type)
                choices.append((current, items_ctxes))
            elif attr.has(type_):
                items_ctxes = []
                for a in attr.fields(type_):
                    next_ctx = ctx + (format_attribute(a.name, a.type),)
                    items_ctxes.append((a.name, next_ctx))
                    recur(current[a.name], next_ctx, a.type)
                choices.append((current, items_ctxes))

        recur(unstructured, (), cls)

        condemned, items_ctxes = random.choice(choices)
        assume(condemned)

        item, ctx = random.choice(items_ctxes)
        condemned[item] = Poison()

        return ctx

    def assert_unstructured_error_and_context(
            self, contextualizing_converter, cl_and_vals, random, data,
    ):
        """``StructureError``s ``ctx`` attribute refers to the point in
        the source where an error occurred.  On Python 3,
        ``StructureError``s are chained to the originating error.
        """
        cl, vals = cl_and_vals

        unstructured = attr.asdict(cl(*vals))
        assume(unstructured)
        ctx = self.break_unstructured(random, unstructured, cl)
        with raises(StructuringError) as exc_info:
            contextualizing_converter.structure(unstructured, cl)
        assert exc_info.value.ctx == ctx

    @given(make_nested_classes(primitive_attrs), randoms(), data())
    def test_from_dict_with_primitive_attrs(
            self, contextualizing_converter, cl_and_vals, random, data,
    ):
        self.assert_unstructured_error_and_context(
            contextualizing_converter, cl_and_vals, random, data,
        )

    @given(
        make_nested_classes(lambda defaults:
                            dict_attrs(defaults) | int_attrs(defaults)),
        randoms(),
        data(),
    )
    def test_from_dict_with_dict_attrs(
            self, contextualizing_converter, cl_and_vals, random, data,
    ):
        self.assert_unstructured_error_and_context(
            contextualizing_converter, cl_and_vals, random, data,
        )

    @given(make_nested_classes(set_attrs), randoms(), data())
    def test_from_dict_with_set_attrs(
            self, contextualizing_converter, cl_and_vals, random, data,
    ):
        self.assert_unstructured_error_and_context(
            contextualizing_converter, cl_and_vals, random, data,
        )

    @given(make_nested_classes(frozenset_attrs), randoms(), data())
    def test_from_dict_with_frozenset_attrs(
            self, contextualizing_converter, cl_and_vals, random, data,
    ):
        self.assert_unstructured_error_and_context(
            contextualizing_converter, cl_and_vals, random, data,
        )

    @given(make_nested_classes(list_attrs), randoms(), data())
    def test_from_dict_with_list_attrs(
            self, contextualizing_converter, cl_and_vals, random, data,
    ):
        self.assert_unstructured_error_and_context(
            contextualizing_converter, cl_and_vals, random, data,
        )

    @given(make_nested_classes(tuple_attrs), randoms(), data())
    def test_from_dict_with_tuple_attrs(
            self, contextualizing_converter, cl_and_vals, random, data,
    ):
        self.assert_unstructured_error_and_context(
            contextualizing_converter, cl_and_vals, random, data,
        )


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
