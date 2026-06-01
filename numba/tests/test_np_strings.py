"""Tests for NumPy string operation support."""

from dataclasses import dataclass
import unittest

import numpy as np

from numba import njit
from numba.tests.support import TestCase


_STRINGS = getattr(np, 'strings', None)
_STRING_DTYPE = getattr(getattr(np, 'dtypes', None), 'StringDType', None)


def strings_equal(left, right):
    return np.strings.equal(left, right)


def strings_not_equal(left, right):
    return np.strings.not_equal(left, right)


def strings_greater_equal(left, right):
    return np.strings.greater_equal(left, right)


def strings_greater(left, right):
    return np.strings.greater(left, right)


def strings_less(left, right):
    return np.strings.less(left, right)


def strings_less_equal(left, right):
    return np.strings.less_equal(left, right)


def strings_count(value, pattern):
    return np.strings.count(value, pattern)


def strings_find(value, pattern):
    return np.strings.find(value, pattern)


def strings_rfind(value, pattern):
    return np.strings.rfind(value, pattern)


def strings_index(value, pattern):
    return np.strings.index(value, pattern)


def strings_rindex(value, pattern):
    return np.strings.rindex(value, pattern)


def strings_endswith(value, suffix):
    return np.strings.endswith(value, suffix)


def strings_startswith(value, prefix):
    return np.strings.startswith(value, prefix)


def strings_str_len(value):
    return np.strings.str_len(value)


def strings_isalpha(value):
    return np.strings.isalpha(value)


def strings_isalnum(value):
    return np.strings.isalnum(value)


def strings_isdecimal(value):
    return np.strings.isdecimal(value)


def strings_isdigit(value):
    return np.strings.isdigit(value)


def strings_islower(value):
    return np.strings.islower(value)


def strings_isnumeric(value):
    return np.strings.isnumeric(value)


def strings_isspace(value):
    return np.strings.isspace(value)


def strings_istitle(value):
    return np.strings.istitle(value)


def strings_isupper(value):
    return np.strings.isupper(value)


def char_equal(left, right):
    return np.char.equal(left, right)


def char_not_equal(left, right):
    return np.char.not_equal(left, right)


def char_greater_equal(left, right):
    return np.char.greater_equal(left, right)


def char_greater(left, right):
    return np.char.greater(left, right)


def char_less(left, right):
    return np.char.less(left, right)


def char_less_equal(left, right):
    return np.char.less_equal(left, right)


def char_count(value, pattern):
    return np.char.count(value, pattern)


def char_find(value, pattern):
    return np.char.find(value, pattern)


def char_rfind(value, pattern):
    return np.char.rfind(value, pattern)


def char_index(value, pattern):
    return np.char.index(value, pattern)


def char_rindex(value, pattern):
    return np.char.rindex(value, pattern)


def char_endswith(value, suffix):
    return np.char.endswith(value, suffix)


def char_startswith(value, prefix):
    return np.char.startswith(value, prefix)


def char_str_len(value):
    return np.char.str_len(value)


def char_isalpha(value):
    return np.char.isalpha(value)


def char_isalnum(value):
    return np.char.isalnum(value)


def char_isdecimal(value):
    return np.char.isdecimal(value)


def char_isdigit(value):
    return np.char.isdigit(value)


def char_islower(value):
    return np.char.islower(value)


def char_isnumeric(value):
    return np.char.isnumeric(value)


def char_isspace(value):
    return np.char.isspace(value)


def char_istitle(value):
    return np.char.istitle(value)


def char_isupper(value):
    return np.char.isupper(value)


def char_compare_chararrays(left, right, cmp, rstrip):
    return np.char.compare_chararrays(left, right, cmp, rstrip)


@dataclass(frozen=True)
class _Operation:
    family: str
    name: str
    nargs: int


@dataclass(frozen=True)
class _Case:
    dtype: str
    layout: str
    builder: object


_COMPARISON_OPS = (
    'equal', 'not_equal', 'greater', 'greater_equal', 'less', 'less_equal',
)
_OCCURRENCE_OPS = (
    'count', 'find', 'rfind', 'index', 'rindex', 'startswith', 'endswith',
)
_PROPERTY_OPS = (
    'str_len', 'isalpha', 'isalnum', 'isdecimal', 'isdigit', 'islower',
    'isnumeric', 'isspace', 'istitle', 'isupper',
)
_OPERATIONS = tuple(
    _Operation('comparison', name, 2) for name in _COMPARISON_OPS
) + tuple(
    _Operation('occurrence', name, 2) for name in _OCCURRENCE_OPS
) + tuple(
    _Operation('property', name, 1) for name in _PROPERTY_OPS
)

_CHAR_PYFUNCS = {
    name: globals()[f'char_{name}']
    for name in _COMPARISON_OPS + _OCCURRENCE_OPS + _PROPERTY_OPS
}
_CHAR_JITFUNCS = {
    name: njit(nogil=True, cache=False)(_CHAR_PYFUNCS[name])
    for name in _CHAR_PYFUNCS
}
_STRINGS_PYFUNCS = {
    name: globals()[f'strings_{name}']
    for name in _COMPARISON_OPS + _OCCURRENCE_OPS + _PROPERTY_OPS
}
_STRINGS_JITFUNCS = {
    name: njit(nogil=True, cache=False)(_STRINGS_PYFUNCS[name])
    for name in _STRINGS_PYFUNCS
} if _STRINGS is not None else {}


def _array(values, dtype_name):
    if dtype_name == 'S':
        return np.array([value.encode('ascii') for value in values],
                        dtype='S8')
    if dtype_name == 'U':
        return np.array(values, dtype='U8')
    if dtype_name == 'T':
        return np.array(values, dtype=_STRING_DTYPE())
    if dtype_name == 'T-na-string':
        return np.array(values, dtype=_STRING_DTYPE(na_object='MISSING'))
    if dtype_name == 'T-na-nan':
        return np.array(values, dtype=_STRING_DTYPE(na_object=np.nan))
    raise AssertionError(dtype_name)


def _values(dtype_name):
    values = ['abcabc', 'xabc', 'abcx', '', 'alpha', 'ABC123']
    if dtype_name == 'T-na-string':
        values[3] = 'MISSING'
    elif dtype_name == 'T-na-nan':
        values[3] = np.nan
    return _array(values, dtype_name)


def _patterns(dtype_name):
    values = ['abc', 'x', 'x', '', 'a', 'ABC']
    if dtype_name == 'T-na-string':
        values[3] = 'MISSING'
    elif dtype_name == 'T-na-nan':
        values[3] = np.nan
    return _array(values, dtype_name)


def _scalar_value(dtype_name):
    if dtype_name == 'S':
        return b'abcabc'
    if dtype_name == 'U':
        return 'abcabc'
    return np.array('abcabc', dtype=_STRING_DTYPE())


def _scalar_pattern(dtype_name):
    if dtype_name == 'S':
        return b'abc'
    if dtype_name == 'U':
        return 'abc'
    return np.array('abc', dtype=_STRING_DTYPE())


def _readonly(value):
    value.flags.writeable = False
    return value


def _zero_dim_from(array, index=0):
    return np.array(array[index], dtype=array.dtype)


def _case_builder(dtype_name, layout, nargs):
    def pair(left, right):
        return (left,) if nargs == 1 else (left, right)

    def build():
        values = _values(dtype_name)
        patterns = _patterns(dtype_name)
        if layout == 'python-scalar':
            return pair(_scalar_value(dtype_name), _scalar_pattern(dtype_name))
        if layout == 'zero-dimensional':
            return pair(_zero_dim_from(values), _zero_dim_from(patterns))
        if layout == 'contiguous-1d':
            return pair(values[:4].copy(), patterns[:4].copy())
        if layout == 'readonly-1d':
            return pair(_readonly(values[:4].copy()),
                        _readonly(patterns[:4].copy()))
        if layout == 'positive-stride-1d':
            return pair(values[::2], patterns[::2])
        if layout == 'negative-stride-1d':
            return pair(values[::-2], patterns[::-2])
        if layout == 'zero-stride-1d':
            return pair(np.broadcast_to(values[:1], (4,)),
                        np.broadcast_to(patterns[:1], (4,)))
        if layout == 'empty-stride-1d':
            return pair(values[:0:2], patterns[:0:2])
        if layout == 'contiguous-2d':
            return pair(values.reshape(2, 3), patterns.reshape(2, 3))
        if layout == 'broadcast-2d':
            return pair(values[:2].reshape(2, 1), patterns[:3].reshape(1, 3))
        if layout == 'shape-mismatch-1d':
            return pair(values[:3].copy(), patterns[:2].copy())
        raise AssertionError(layout)

    return build


def _audit_cases():
    dtype_names = ['S', 'U']
    if _STRING_DTYPE is not None:
        dtype_names.extend(['T', 'T-na-string', 'T-na-nan'])

    layouts = (
        'python-scalar',
        'zero-dimensional',
        'contiguous-1d',
        'readonly-1d',
        'positive-stride-1d',
        'negative-stride-1d',
        'zero-stride-1d',
        'empty-stride-1d',
        'contiguous-2d',
        'broadcast-2d',
        'shape-mismatch-1d',
    )

    for op in _OPERATIONS:
        for dtype_name in dtype_names:
            for layout in layouts:
                if layout == 'python-scalar' and dtype_name.startswith('T'):
                    continue
                yield op, _Case(dtype_name, layout,
                                _case_builder(dtype_name, layout, op.nargs))


def _copy_arg(arg):
    if isinstance(arg, np.ndarray):
        return arg.copy()
    return arg


def _copy_args(args):
    return tuple(_copy_arg(arg) for arg in args)


def _assert_arrays_unchanged(test, args, before):
    for arg, expected in zip(args, before):
        if not isinstance(arg, np.ndarray):
            continue
        if np.array_equal(arg, expected):
            continue
        try:
            unchanged = np.array_equal(arg, expected, equal_nan=True)
        except TypeError:
            unchanged = False
        if not unchanged:
            test.assertPreciseEqual(arg, expected)


def _assert_same_outcome(test, implementation, baseline, builder):
    impl_args = _copy_args(builder())
    base_args = _copy_args(builder())
    before = _copy_args(impl_args)

    try:
        expected = baseline(*base_args)
    except Exception as expected_exc:
        with test.assertRaises(type(expected_exc)):
            implementation(*impl_args)
    else:
        actual = implementation(*impl_args)
        actual_array = np.asarray(actual)
        expected_array = np.asarray(expected)
        test.assertEqual(actual_array.dtype, expected_array.dtype)
        test.assertEqual(actual_array.shape, expected_array.shape)
        test.assertPreciseEqual(actual_array, expected_array)

    _assert_arrays_unchanged(test, impl_args, before)


class TestNPStringParityAudit(TestCase):

    def test_np_char_and_strings_shape_parity(self):
        row_count = 0
        for op, case in _audit_cases():
            apis = [('char', _CHAR_JITFUNCS[op.name],
                     getattr(np.char, op.name))]
            if _STRINGS is not None:
                apis.append(('strings', _STRINGS_JITFUNCS[op.name],
                             getattr(_STRINGS, op.name)))

            for api, implementation, baseline in apis:
                if api == 'char' and case.dtype.startswith('T'):
                    continue
                with self.subTest(api=api, family=op.family, method=op.name,
                                  dtype=case.dtype, layout=case.layout):
                    _assert_same_outcome(self, implementation, baseline,
                                         case.builder)
                row_count += 1

        if _STRING_DTYPE is None:
            self.assertEqual(row_count, 506)
        else:
            self.assertEqual(row_count, 1702)

    def test_compare_chararrays_parity(self):
        implementation = njit(nogil=True, cache=False)(char_compare_chararrays)
        cases = [
            ('ab', np.array(['abc', 'ab', 'a'], dtype='U3')),
            (np.array('ab', dtype='U2'),
             np.array(['abc', 'ab', 'a'], dtype='U3')),
            (b'ab', np.array([b'abc', b'ab', b'a'], dtype='S3')),
            (np.array(b'ab', dtype='S2'),
             np.array([b'abc', b'ab', b'a'], dtype='S3')),
            (np.array(['abc ', 'abc', 'abd'], dtype='U4'), 'abc'),
            (
                np.array(['abc ', 'skip', 'abc\x00x', 'skip', 'abd'],
                         dtype='U6')[::2],
                np.array(['abc', 'skip', 'abc', 'skip', 'abc'],
                         dtype='U6')[::2],
            ),
            (
                np.array([b'abc ', b'skip', b'abc\x00x', b'skip', b'abd'],
                         dtype='S6')[::-2],
                np.array([b'abc', b'skip', b'abc', b'skip', b'abc'],
                         dtype='S6')[::-2],
            ),
            (
                np.broadcast_to(np.array(['abc\x00x'], dtype='U6'), (3,)),
                np.broadcast_to(np.array(['abc'], dtype='U6'), (3,)),
            ),
        ]
        for left, right in cases:
            for cmp in ('==', '!=', '>=', '>', '<', '<='):
                for rstrip in (True, False):
                    with self.subTest(cmp=cmp, rstrip=rstrip,
                                      left_type=type(left).__name__,
                                      right_type=type(right).__name__):
                        _assert_same_outcome(
                            self, implementation, np.char.compare_chararrays,
                            lambda left=left, right=right, cmp=cmp,
                                   rstrip=rstrip: (left, right, cmp, rstrip),
                        )


class TestNPStringFocusedSemantics(TestCase):

    def check(self, pyfunc, *args):
        cfunc = njit(nogil=True, cache=False)(pyfunc)
        expected = pyfunc(*args)
        got = cfunc(*args)
        self.assertPreciseEqual(expected, got)

    def test_np_char_rstrips_fixed_width_comparisons(self):
        left = np.array(['a ', 'a'], dtype='U2')
        right = np.array(['a', 'a'], dtype='U1')

        expected = np.array([True, True])
        self.assertPreciseEqual(np.char.equal(left, right), expected)
        self.check(char_equal, left, right)

    @unittest.skipUnless(_STRINGS is not None,
                         "np.strings is available on NumPy 2.x")
    def test_np_strings_preserves_fixed_width_comparison_semantics(self):
        left = np.array(['a ', 'a'], dtype='U2')
        right = np.array(['a', 'a'], dtype='U1')

        self.assertPreciseEqual(_STRINGS.equal(left, right),
                                np.array([False, True]))
        self.assertPreciseEqual(np.char.equal(left, right),
                                np.array([True, True]))
        self.check(strings_equal, left, right)

    def test_fixed_width_nd_scalar_broadcasting(self):
        cases = [
            (np.array(['abc ', 'abc\x00', 'abcx', 'abd'],
                      dtype='U4').reshape(2, 2), 'abc'),
            ('abc', np.array(['abc ', 'abc\x00', 'abcx', 'abd'],
                             dtype='U4').reshape(2, 2)),
            (np.array([b'abc ', b'abc\x00', b'abcx', b'abd'],
                      dtype='S4').reshape(2, 2), b'abc'),
            (b'abc', np.array([b'abc ', b'abc\x00', b'abcx', b'abd'],
                              dtype='S4').reshape(2, 2)),
        ]
        apis = [('char', _CHAR_JITFUNCS, np.char)]
        if _STRINGS is not None:
            apis.append(('strings', _STRINGS_JITFUNCS, _STRINGS))
        for api, implementations, baselines in apis:
            for name in _COMPARISON_OPS:
                for left, right in cases:
                    with self.subTest(api=api, method=name,
                                      left_type=type(left).__name__,
                                      right_type=type(right).__name__):
                        _assert_same_outcome(
                            self, implementations[name],
                            getattr(baselines, name),
                            lambda left=left, right=right: (left, right),
                        )

        occurrence_cases = [
            (np.array(['abcabc', 'xabc', 'zzzz', 'abcx'],
                      dtype='U6').reshape(2, 2), 'abc'),
            ('abcabc', np.array(['abc', 'x', 'z', 'bc'],
                                dtype='U3').reshape(2, 2)),
            (np.array([b'abcabc', b'xabc', b'zzzz', b'abcx'],
                      dtype='S6').reshape(2, 2), b'abc'),
            (b'abcabc', np.array([b'abc', b'x', b'z', b'bc'],
                                 dtype='S3').reshape(2, 2)),
        ]
        index_cases = [
            (np.array(['abcabc', 'xabc', 'abczz', 'abcx'],
                      dtype='U6').reshape(2, 2), 'abc'),
            ('abcabc', np.array(['abc', 'a', 'bc', 'c'],
                                dtype='U3').reshape(2, 2)),
            (np.array([b'abcabc', b'xabc', b'abczz', b'abcx'],
                      dtype='S6').reshape(2, 2), b'abc'),
            (b'abcabc', np.array([b'abc', b'a', b'bc', b'c'],
                                 dtype='S3').reshape(2, 2)),
        ]
        for api, implementations, baselines in apis:
            for name in ('count', 'find', 'rfind'):
                for value, pattern in occurrence_cases:
                    with self.subTest(api=api, method=name,
                                      value_type=type(value).__name__,
                                      pattern_type=type(pattern).__name__):
                        _assert_same_outcome(
                            self, implementations[name],
                            getattr(baselines, name),
                            lambda value=value, pattern=pattern:
                                (value, pattern),
                        )
            for name in ('index', 'rindex'):
                for value, pattern in index_cases:
                    with self.subTest(api=api, method=name,
                                      value_type=type(value).__name__,
                                      pattern_type=type(pattern).__name__):
                        _assert_same_outcome(
                            self, implementations[name],
                            getattr(baselines, name),
                            lambda value=value, pattern=pattern:
                                (value, pattern),
                        )

    @unittest.skipUnless(
        _STRING_DTYPE is not None,
        "StringDType is available on NumPy 2.x",
    )
    def test_stringdtype_na_object_none_variant(self):
        dtype = _STRING_DTYPE(na_object=None)
        values = np.array(['alpha', None, ''], dtype=dtype)
        patterns = np.array(['a', None, ''], dtype=dtype)

        for name, args in (
            ('equal', (values, values.copy())),
            ('count', (values, patterns)),
            ('str_len', (values,)),
            ('isalpha', (values,)),
        ):
            with self.subTest(method=name):
                _assert_same_outcome(
                    self, _STRINGS_JITFUNCS[name], getattr(_STRINGS, name),
                    lambda args=args: args,
                )


if __name__ == '__main__':
    unittest.main()
