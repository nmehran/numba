"""
Numba overloads for numpy.character routines
"""

from numba.np.strings import OPTIONS
from numba.np.strings._shared import (
    ensure_slice as _ensure_slice,
    equal_dispatch as _equal_dispatch,
    equal_kernel as _equal_kernel,
    fixed_width_array, fixed_width_array_pair,
    needs_nd_path,
    order_dispatch as _order_dispatch,
    register_pair as _register_pair,
    register_single as _register_single,
)
from numba.np.strings._kernels import (
    greater_equal, greater, equal,
    equal_sub32_bytes, equal_sub32_unicode,
    compare_chararrays,
    count, endswith, startswith, find, rfind, index, rindex, str_len,
    str_len_bytes, _init_sub_indices, _str_len_loop,
    isalpha, isalnum, isdecimal, isdigit, islower, isnumeric, isspace,
    istitle, isupper, scalar_bytes_len, scalar_strings_len,
    scalar_bytes_isalpha, scalar_strings_isalpha,
    scalar_bytes_isalnum, scalar_strings_isalnum, scalar_strings_isdecimal,
    scalar_bytes_isdigit, scalar_strings_isdigit, scalar_strings_isnumeric,
    scalar_bytes_isspace, scalar_strings_isspace,
    scalar_bytes_istitle, scalar_strings_istitle,
    scalar_bytes_isupper, scalar_strings_isupper,
    scalar_bytes_islower, scalar_strings_islower
)
from numba.np.strings._fixed_width import (
    OP_COUNT, OP_ENDSWITH, OP_EQUAL, OP_FIND, OP_INDEX, OP_RFIND,
    OP_RINDEX, OP_STARTSWITH, OP_GREATER, OP_GREATER_EQUAL, OP_LESS,
    OP_LESS_EQUAL, OP_ISALPHA, OP_ISALNUM, OP_ISDECIMAL, OP_ISDIGIT,
    OP_ISLOWER, OP_ISNUMERIC, OP_ISSPACE, OP_ISTITLE, OP_ISUPPER,
    _fixed_binary_bool_nd, _fixed_binary_int_nd,
    _fixed_scalar_mixed_bool_nd, _fixed_scalar_mixed_int_nd,
    fixed_property_nd, fixed_str_len_nd,
)
from numba.core import types
from numba.core.errors import NumbaTypeError
from numba.core.typing.templates import AttributeTemplate
from numba.extending import infer_getattr, overload, register_jitable
from numba import literally, njit, objmode
import numpy as np


_CHAR_INFO_SCALARS_AS_ARRAY = not isinstance(np.char.str_len, np.ufunc)


def _char_count(a, sub, start=0, end=None):
    return np.char.count(a, sub, start, end)


def _char_endswith(a, suffix, start=0, end=None):
    return np.char.endswith(a, suffix, start, end)


def _char_startswith(a, prefix, start=0, end=None):
    return np.char.startswith(a, prefix, start, end)


def _char_find(a, sub, start=0, end=None):
    return np.char.find(a, sub, start, end)


def _char_rfind(a, sub, start=0, end=None):
    return np.char.rfind(a, sub, start, end)


def _char_index(a, sub, start=0, end=None):
    return np.char.index(a, sub, start, end)


def _char_rindex(a, sub, start=0, end=None):
    return np.char.rindex(a, sub, start, end)


def _char_str_len(a):
    return np.char.str_len(a)


def _char_isalpha(a):
    return np.char.isalpha(a)


def _char_isalnum(a):
    return np.char.isalnum(a)


def _char_isdecimal(a):
    return np.char.isdecimal(a)


def _char_isdigit(a):
    return np.char.isdigit(a)


def _char_islower(a):
    return np.char.islower(a)


def _char_isnumeric(a):
    return np.char.isnumeric(a)


def _char_isspace(a):
    return np.char.isspace(a)


def _char_istitle(a):
    return np.char.istitle(a)


def _char_isupper(a):
    return np.char.isupper(a)


_CHAR_INFO_FUNCTIONS = {
    'count': _char_count,
    'endswith': _char_endswith,
    'startswith': _char_startswith,
    'find': _char_find,
    'rfind': _char_rfind,
    'index': _char_index,
    'rindex': _char_rindex,
    'str_len': _char_str_len,
    'isalpha': _char_isalpha,
    'isalnum': _char_isalnum,
    'isdecimal': _char_isdecimal,
    'isdigit': _char_isdigit,
    'islower': _char_islower,
    'isnumeric': _char_isnumeric,
    'isspace': _char_isspace,
    'istitle': _char_istitle,
    'isupper': _char_isupper,
}


# Resolve np.char.<name> to Numba wrappers inside compiled code so the
# compatibility namespace shares one overload path across NumPy versions.
@infer_getattr
class _CharModuleAttrs(AttributeTemplate):
    key = types.Module(np.char)

    def generic_resolve(self, value, attr):
        function = _CHAR_INFO_FUNCTIONS.get(attr)
        if function is not None:
            return self.context.resolve_value_type(function)


def _overload_char_function(numpy_function, wrapper_function):
    def decorate(overload_function):
        overload(numpy_function, **OPTIONS)(overload_function)
        overload(wrapper_function, **OPTIONS)(overload_function)
        return overload_function
    return decorate


@register_jitable(boundscheck=False, forceinline=True,
                  no_cpython_wrapper=True, nogil=True)
def _char_info_scalar_result(value):
    if _CHAR_INFO_SCALARS_AS_ARRAY:
        return np.array(value)
    return value


@njit(nogil=False, cache=False)
def _raise_byte_unicode_only_property_array(a, op):
    with objmode():
        if op == OP_ISDECIMAL:
            np.char.isdecimal(a)
        else:
            np.char.isnumeric(a)
    return np.empty(a.shape, np.bool_)


@njit(nogil=False, cache=False)
def _raise_byte_unicode_only_property_scalar(a, op):
    with objmode():
        if op == OP_ISDECIMAL:
            np.char.isdecimal(a)
        else:
            np.char.isnumeric(a)
    return False


def _byte_unicode_only_property_error_impl(a, op):
    if isinstance(a, types.Array):
        def impl(a):
            return _raise_byte_unicode_only_property_array(a, literally(op))
    else:
        def impl(a):
            return _char_info_scalar_result(
                _raise_byte_unicode_only_property_scalar(a, literally(op)))
    return impl


# -----------------------------------------------------------------------------
# Comparison Operators


def _fixed_width_scalar_kind(value):
    if isinstance(value, (types.Bytes, types.CharSeq)):
        return 'bytes'
    if isinstance(value, (types.UnicodeType, types.UnicodeCharSeq)):
        return 'unicode'
    return None


def _fixed_width_array_scalar_pair(left, right):
    left_kind, left_width = fixed_width_array(left)
    right_kind, right_width = fixed_width_array(right)
    left_scalar_kind = _fixed_width_scalar_kind(left)
    right_scalar_kind = _fixed_width_scalar_kind(right)

    if left_kind and right_scalar_kind == left_kind:
        return False, left_width
    if right_kind and left_scalar_kind == right_kind:
        return True, right_width
    return None, 0


def _fixed_width_bool_pair_nd_impl(x1, x2, op, rstrip, invert=False):
    if not needs_nd_path(x1, x2):
        return None
    _, width, width2 = fixed_width_array_pair(x1, x2)
    if width:
        def impl(x1, x2):
            result = _fixed_binary_bool_nd(
                x1, x2, literally(width), literally(width2), 0, 0, op, rstrip)
            return ~result if invert else result

        return impl

    scalar_left, width = _fixed_width_array_scalar_pair(x1, x2)
    if not width:
        return None

    def impl(x1, x2):
        if scalar_left:
            result = _fixed_scalar_mixed_bool_nd(
                x2, x1, literally(width), len(x1), 0, 0, op, rstrip, True)
        else:
            result = _fixed_scalar_mixed_bool_nd(
                x1, x2, literally(width), len(x2), 0, 0, op, rstrip, False)
        return ~result if invert else result

    return impl


def _fixed_width_equal_nd_impl(x1, x2, invert, rstrip):
    return _fixed_width_bool_pair_nd_impl(
        x1, x2, OP_EQUAL, rstrip, invert)


def _fixed_width_order_nd_impl(x1, x2, op, rstrip=True):
    if op == 'greater':
        op = OP_GREATER
    elif op == 'greater_equal':
        op = OP_GREATER_EQUAL
    elif op == 'less':
        op = OP_LESS
    elif op == 'less_equal':
        op = OP_LESS_EQUAL

    return _fixed_width_bool_pair_nd_impl(x1, x2, op, rstrip)


def _fixed_width_bool_nd_impl(a, sub, start, end, op):
    if not needs_nd_path(a, sub):
        return None
    _, width, width2 = fixed_width_array_pair(a, sub)
    s, e = _ensure_slice(start, end)
    if width:
        def impl(a, sub, start=0, end=None):
            start = start or s
            end = e if end is None else end
            start, end = _init_sub_indices(start, end, literally(width))
            return _fixed_binary_bool_nd(
                a, sub, literally(width), literally(width2), start, end, op,
                False)

        return impl

    scalar_left, width = _fixed_width_array_scalar_pair(a, sub)
    if not width:
        return None

    def impl(a, sub, start=0, end=None):
        start = start or s
        end = e if end is None else end
        if scalar_left:
            start, end = _init_sub_indices(start, end, len(a))
            return _fixed_scalar_mixed_bool_nd(
                sub, a, literally(width), len(a), start, end, op, False, True)
        start, end = _init_sub_indices(start, end, literally(width))
        return _fixed_scalar_mixed_bool_nd(
            a, sub, literally(width), len(sub), start, end, op, False, False)

    return impl


def _fixed_width_int_nd_impl(a, sub, start, end, op):
    if not needs_nd_path(a, sub):
        return None
    _, width, width2 = fixed_width_array_pair(a, sub)
    s, e = _ensure_slice(start, end)
    if width:
        def impl(a, sub, start=0, end=None):
            start = start or s
            end = e if end is None else end
            start, end = _init_sub_indices(start, end, literally(width))
            return _fixed_binary_int_nd(
                a, sub, literally(width), literally(width2), start, end, op)

        return impl

    scalar_left, width = _fixed_width_array_scalar_pair(a, sub)
    if not width:
        return None

    def impl(a, sub, start=0, end=None):
        start = start or s
        end = e if end is None else end
        if scalar_left:
            start, end = _init_sub_indices(start, end, len(a))
            return _fixed_scalar_mixed_int_nd(
                sub, a, literally(width), len(a), start, end, op, True)
        start, end = _init_sub_indices(start, end, literally(width))
        return _fixed_scalar_mixed_int_nd(
            a, sub, literally(width), len(sub), start, end, op, False)

    return impl


def _fixed_width_predicate_nd_impl(a, op, as_bytes=None):
    _, width = fixed_width_array(a)
    if not width or a.ndim <= 1:
        return None
    if as_bytes is None:
        as_bytes = isinstance(a.dtype, types.CharSeq)

    def impl(a):
        return fixed_property_nd(
            a, literally(width), literally(as_bytes), literally(op))

    return impl


@overload(np.char.equal, **OPTIONS)
def ov_char_equal(x1, x2):
    fixed = _fixed_width_equal_nd_impl(x1, x2, False, True)
    if fixed is not None:
        return fixed
    register_x1, register_x2, x1_dim, x2_dim = _register_pair(x1, x2)
    return _equal_dispatch(register_x1, register_x2, x1_dim, x2_dim,
                           _equal_kernel(x1, x2, equal, equal_sub32_bytes,
                                         equal_sub32_unicode), True,
                           scalar_as_array=True)


@overload(np.char.not_equal, **OPTIONS)
def ov_char_not_equal(x1, x2):
    fixed = _fixed_width_equal_nd_impl(x1, x2, True, True)
    if fixed is not None:
        return fixed
    register_x1, register_x2, x1_dim, x2_dim = _register_pair(x1, x2)
    return _equal_dispatch(register_x1, register_x2, x1_dim, x2_dim,
                           _equal_kernel(x1, x2, equal, equal_sub32_bytes,
                                         equal_sub32_unicode),
                           True, invert=True, scalar_as_array=True)


@overload(np.char.greater_equal, **OPTIONS)
def ov_char_greater_equal(x1, x2):
    fixed = _fixed_width_order_nd_impl(x1, x2, OP_GREATER_EQUAL)
    if fixed is not None:
        return fixed
    register_x1, register_x2, x1_dim, x2_dim = _register_pair(x1, x2)
    return _order_dispatch(register_x1, register_x2, x1_dim, x2_dim,
                           greater, greater_equal, 'greater_equal', True,
                           scalar_as_array=True)


@overload(np.char.greater, **OPTIONS)
def ov_char_greater(x1, x2):
    fixed = _fixed_width_order_nd_impl(x1, x2, OP_GREATER)
    if fixed is not None:
        return fixed
    register_x1, register_x2, x1_dim, x2_dim = _register_pair(x1, x2)
    return _order_dispatch(register_x1, register_x2, x1_dim, x2_dim,
                           greater, greater_equal, 'greater', True,
                           scalar_as_array=True)


@overload(np.char.less, **OPTIONS)
def ov_char_less(x1, x2):
    fixed = _fixed_width_order_nd_impl(x1, x2, OP_LESS)
    if fixed is not None:
        return fixed
    register_x1, register_x2, x1_dim, x2_dim = _register_pair(x1, x2)
    return _order_dispatch(register_x1, register_x2, x1_dim, x2_dim,
                           greater, greater_equal, 'less', True,
                           scalar_as_array=True)


@overload(np.char.less_equal, **OPTIONS)
def ov_char_less_equal(x1, x2):
    fixed = _fixed_width_order_nd_impl(x1, x2, OP_LESS_EQUAL)
    if fixed is not None:
        return fixed
    register_x1, register_x2, x1_dim, x2_dim = _register_pair(x1, x2)
    return _order_dispatch(register_x1, register_x2, x1_dim, x2_dim,
                           greater, greater_equal, 'less_equal', True,
                           scalar_as_array=True)


@overload(np.char.compare_chararrays, **OPTIONS)
def ov_char_compare_chararrays(a1, a2, cmp, rstrip):
    if not isinstance(cmp, (types.Bytes, types.UnicodeType)):
        raise NumbaTypeError(
            f'a bytes-like object is required, not {cmp.name}')

    register_a1, register_a2, a1_dim, a2_dim = _register_pair(a1, a2)
    left_scalar_like = a1_dim <= 0 < a2_dim

    if a1_dim > 0 or a2_dim > 0:
        def impl(a1, a2, cmp, rstrip):
            if left_scalar_like:
                return compare_chararrays(*register_a2(a2, False),
                                          *register_a1(a1, False),
                                          True, cmp, rstrip)
            return compare_chararrays(*register_a1(a1, False),
                                      *register_a2(a2, False),
                                      False, cmp, rstrip)
    else:
        def impl(a1, a2, cmp, rstrip):
            return np.array(compare_chararrays(*register_a1(a1, False),
                                               *register_a2(a2, False),
                                               False, cmp, rstrip)[0])
    return impl


# -----------------------------------------------------------------------------
# String Information


@_overload_char_function(np.char.count, _char_count)
def ov_char_count(a, sub, start=0, end=None):
    fixed = _fixed_width_int_nd_impl(a, sub, start, end, OP_COUNT)
    if fixed is not None:
        return fixed
    register_a, register_sub, a_dim, sub_dim = _register_pair(a, sub, 1)
    s, e = _ensure_slice(start, end)

    if a_dim > 0 or sub_dim > 0:
        def impl(a, sub, start=0, end=None):
            start = start or s
            end = e if end is None else end
            return count(*register_a(a, False),
                         *register_sub(sub, False),
                         start, end)
    else:
        def impl(a, sub, start=0, end=None):
            start = start or s
            end = e if end is None else end
            return _char_info_scalar_result(
                count(*register_a(a, False),
                      *register_sub(sub, False),
                      start, end)[0])
    return impl


@_overload_char_function(np.char.endswith, _char_endswith)
def ov_char_endswith(a, suffix, start=0, end=None):
    fixed = _fixed_width_bool_nd_impl(a, suffix, start, end, OP_ENDSWITH)
    if fixed is not None:
        return fixed
    register_a, register_sub, a_dim, sub_dim = _register_pair(a, suffix, 1)
    s, e = _ensure_slice(start, end)

    if a_dim > 0 or sub_dim > 0:
        def impl(a, suffix, start=0, end=None):
            start = start or s
            end = e if end is None else end
            return endswith(*register_a(a, False),
                            *register_sub(suffix, False),
                            start, end)
    else:
        def impl(a, suffix, start=0, end=None):
            start = start or s
            end = e if end is None else end
            return _char_info_scalar_result(
                endswith(*register_a(a, False),
                         *register_sub(suffix, False),
                         start, end)[0])
    return impl


@_overload_char_function(np.char.startswith, _char_startswith)
def ov_char_startswith(a, prefix, start=0, end=None):
    fixed = _fixed_width_bool_nd_impl(a, prefix, start, end, OP_STARTSWITH)
    if fixed is not None:
        return fixed
    register_a, register_sub, a_dim, sub_dim = _register_pair(a, prefix, 1)
    s, e = _ensure_slice(start, end)

    if a_dim > 0 or sub_dim > 0:
        def impl(a, prefix, start=0, end=None):
            start = start or s
            end = e if end is None else end
            return startswith(*register_a(a, False),
                              *register_sub(prefix, False),
                              start, end)
    else:
        def impl(a, prefix, start=0, end=None):
            start = start or s
            end = e if end is None else end
            return _char_info_scalar_result(
                startswith(*register_a(a, False),
                           *register_sub(prefix, False),
                           start, end)[0])
    return impl


@_overload_char_function(np.char.find, _char_find)
def ov_char_find(a, sub, start=0, end=None):
    fixed = _fixed_width_int_nd_impl(a, sub, start, end, OP_FIND)
    if fixed is not None:
        return fixed
    register_a, register_sub, a_dim, sub_dim = _register_pair(a, sub, 1)
    s, e = _ensure_slice(start, end)

    if a_dim > 0 or sub_dim > 0:
        def impl(a, sub, start=0, end=None):
            start = start or s
            end = e if end is None else end
            return find(*register_a(a, False),
                        *register_sub(sub, False),
                        start, end)
    else:
        def impl(a, sub, start=0, end=None):
            start = start or s
            end = e if end is None else end
            return _char_info_scalar_result(
                find(*register_a(a, False),
                     *register_sub(sub, False),
                     start, end)[0])
    return impl


@_overload_char_function(np.char.rfind, _char_rfind)
def ov_char_rfind(a, sub, start=0, end=None):
    fixed = _fixed_width_int_nd_impl(a, sub, start, end, OP_RFIND)
    if fixed is not None:
        return fixed
    register_a, register_sub, a_dim, sub_dim = _register_pair(a, sub, 1)
    s, e = _ensure_slice(start, end)

    if a_dim > 0 or sub_dim > 0:
        def impl(a, sub, start=0, end=None):
            start = start or s
            end = e if end is None else end
            return rfind(*register_a(a, False),
                         *register_sub(sub, False),
                         start, end)
    else:
        def impl(a, sub, start=0, end=None):
            start = start or s
            end = e if end is None else end
            return _char_info_scalar_result(
                rfind(*register_a(a, False),
                      *register_sub(sub, False),
                      start, end)[0])
    return impl


@_overload_char_function(np.char.index, _char_index)
def ov_char_index(a, sub, start=0, end=None):
    fixed = _fixed_width_int_nd_impl(a, sub, start, end, OP_INDEX)
    if fixed is not None:
        return fixed
    register_a, register_sub, a_dim, sub_dim = _register_pair(a, sub, 1)
    s, e = _ensure_slice(start, end)

    if a_dim > 0 or sub_dim > 0:
        def impl(a, sub, start=0, end=None):
            start = start or s
            end = e if end is None else end
            return index(*register_a(a, False),
                         *register_sub(sub, False),
                         start, end)
    else:
        def impl(a, sub, start=0, end=None):
            start = start or s
            end = e if end is None else end
            return _char_info_scalar_result(
                index(*register_a(a, False),
                      *register_sub(sub, False),
                      start, end)[0])
    return impl


@_overload_char_function(np.char.rindex, _char_rindex)
def ov_char_rindex(a, sub, start=0, end=None):
    fixed = _fixed_width_int_nd_impl(a, sub, start, end, OP_RINDEX)
    if fixed is not None:
        return fixed
    register_a, register_sub, a_dim, sub_dim = _register_pair(a, sub, 1)
    s, e = _ensure_slice(start, end)

    if a_dim > 0 or sub_dim > 0:
        def impl(a, sub, start=0, end=None):
            start = start or s
            end = e if end is None else end
            return rindex(*register_a(a, False),
                          *register_sub(sub, False),
                          start, end)
    else:
        def impl(a, sub, start=0, end=None):
            start = start or s
            end = e if end is None else end
            return _char_info_scalar_result(
                rindex(*register_a(a, False),
                       *register_sub(sub, False),
                       start, end)[0])
    return impl


@_overload_char_function(np.char.str_len, _char_str_len)
def ov_char_str_len(a):
    _, width = fixed_width_array(a)
    if width and a.ndim > 1:
        def impl(a):
            return fixed_str_len_nd(a, literally(width))

        return impl

    register_a, a_dim, as_bytes = _register_single(a)
    array_len = str_len_bytes if as_bytes else str_len
    width = a.dtype.count if isinstance(a, types.Array) else 0

    if a_dim > 0 and width and a.layout == 'C':
        if as_bytes:
            direct_len = _str_len_loop if width <= 8 else str_len_bytes

            def impl(a):
                return direct_len(a.view(np.uint8), a.size, literally(width))
        else:
            direct_len = _str_len_loop if width <= 16 else str_len

            def impl(a):
                return direct_len(a.view(np.int32), a.size, width)
    elif a_dim > 0:
        def impl(a):
            return array_len(*register_a(a, False))
    elif a_dim == -2:
        if as_bytes:
            def impl(a):
                return _char_info_scalar_result(scalar_bytes_len(a))
        else:
            def impl(a):
                return _char_info_scalar_result(scalar_strings_len(a))
    else:
        def impl(a):
            return _char_info_scalar_result(
                array_len(*register_a(a, False))[0])
    return impl


def _char_predicate_impl(a, fixed_op, array_kernel, scalar_bytes_kernel,
                         scalar_strings_kernel):
    fixed = _fixed_width_predicate_nd_impl(a, fixed_op)
    if fixed is not None:
        return fixed
    register_a, a_dim, as_bytes = _register_single(a)

    if a_dim > 0:
        def impl(a):
            return array_kernel(*register_a(a, False), as_bytes)
    elif a_dim == -2:
        scalar_kernel = scalar_bytes_kernel if as_bytes \
            else scalar_strings_kernel

        def impl(a):
            return _char_info_scalar_result(scalar_kernel(a))
    else:
        def impl(a):
            return _char_info_scalar_result(
                array_kernel(*register_a(a, False), as_bytes)[0])
    return impl


@_overload_char_function(np.char.isalpha, _char_isalpha)
def ov_char_isalpha(a):
    return _char_predicate_impl(
        a, OP_ISALPHA, isalpha, scalar_bytes_isalpha,
        scalar_strings_isalpha)


@_overload_char_function(np.char.isalnum, _char_isalnum)
def ov_char_isalnum(a):
    return _char_predicate_impl(
        a, OP_ISALNUM, isalnum, scalar_bytes_isalnum,
        scalar_strings_isalnum)


@_overload_char_function(np.char.isspace, _char_isspace)
def ov_char_isspace(a):
    return _char_predicate_impl(
        a, OP_ISSPACE, isspace, scalar_bytes_isspace,
        scalar_strings_isspace)


@_overload_char_function(np.char.isdecimal, _char_isdecimal)
def ov_char_isdecimal(a):
    catch_incompatible = NumbaTypeError("isdecimal is only available for "
                                        "Unicode strings and arrays")
    kind, _ = fixed_width_array(a)
    if kind == 'bytes':
        return _byte_unicode_only_property_error_impl(a, OP_ISDECIMAL)
    fixed = _fixed_width_predicate_nd_impl(a, OP_ISDECIMAL, False)
    if fixed is not None and not isinstance(a.dtype, types.CharSeq):
        return fixed
    register_a, a_dim, as_bytes = _register_single(a, catch_incompatible)
    if as_bytes:
        return _byte_unicode_only_property_error_impl(a, OP_ISDECIMAL)

    if a_dim > 0:
        def impl(a):
            return isdecimal(*register_a(a, False))
    elif a_dim == -2:
        def impl(a):
            return _char_info_scalar_result(scalar_strings_isdecimal(a))
    else:
        def impl(a):
            return _char_info_scalar_result(
                isdecimal(*register_a(a, False))[0])
    return impl


@_overload_char_function(np.char.isdigit, _char_isdigit)
def ov_char_isdigit(a):
    return _char_predicate_impl(
        a, OP_ISDIGIT, isdigit, scalar_bytes_isdigit,
        scalar_strings_isdigit)


@_overload_char_function(np.char.isnumeric, _char_isnumeric)
def ov_char_isnumeric(a):
    catch_incompatible = NumbaTypeError("isnumeric is only available for "
                                        "Unicode strings and arrays")
    kind, _ = fixed_width_array(a)
    if kind == 'bytes':
        return _byte_unicode_only_property_error_impl(a, OP_ISNUMERIC)
    fixed = _fixed_width_predicate_nd_impl(a, OP_ISNUMERIC, False)
    if fixed is not None and not isinstance(a.dtype, types.CharSeq):
        return fixed
    register_a, a_dim, as_bytes = _register_single(a, catch_incompatible)
    if as_bytes:
        return _byte_unicode_only_property_error_impl(a, OP_ISNUMERIC)

    if a_dim > 0:
        def impl(a):
            return isnumeric(*register_a(a, False))
    elif a_dim == -2:
        def impl(a):
            return _char_info_scalar_result(scalar_strings_isnumeric(a))
    else:
        def impl(a):
            return _char_info_scalar_result(
                isnumeric(*register_a(a, False))[0])
    return impl


@_overload_char_function(np.char.istitle, _char_istitle)
def ov_char_istitle(a):
    return _char_predicate_impl(
        a, OP_ISTITLE, istitle, scalar_bytes_istitle,
        scalar_strings_istitle)


@_overload_char_function(np.char.isupper, _char_isupper)
def ov_char_isupper(a):
    return _char_predicate_impl(
        a, OP_ISUPPER, isupper, scalar_bytes_isupper,
        scalar_strings_isupper)


@_overload_char_function(np.char.islower, _char_islower)
def ov_char_islower(a):
    return _char_predicate_impl(
        a, OP_ISLOWER, islower, scalar_bytes_islower,
        scalar_strings_islower)

# -----------------------------------------------------------------------------
