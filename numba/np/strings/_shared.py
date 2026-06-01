"""Shared helpers for NumPy string overload registration."""

from numba.np.strings import JIT_OPTIONS
from numba.np.strings._intrinsics import (
    register_array_bytes, register_array_bytes_strided,
    register_scalar_bytes, register_array_strings,
    register_array_strings_strided, register_scalar_strings,
)
from numba.core import types
from numba.core.errors import (
    NumbaError, NumbaNotImplementedError, NumbaTypeError, NumbaValueError,
)
from numba.extending import register_jitable
import numpy as np


_NUMPY_MAJOR = int(np.__version__.split('.')[0])


@register_jitable(**JIT_OPTIONS)
def unary_row_plan(value, unit_size):
    out_shape_tuple = value.shape
    out_ndim = value.ndim
    out_shape = np.empty(out_ndim, np.intp)
    value_steps = np.empty(out_ndim, np.intp)
    total = 1
    for axis in range(out_ndim):
        dim = value.shape[axis]
        out_shape[axis] = dim
        value_steps[axis] = value.strides[axis] // unit_size
        total *= dim

    co_ndim = 0
    for axis in range(out_ndim):
        dim = out_shape[axis]
        value_step = value_steps[axis]
        if co_ndim > 0:
            prev = co_ndim - 1
            if value_steps[prev] == value_step * dim:
                out_shape[prev] *= dim
                value_steps[prev] = value_step
                continue
        out_shape[co_ndim] = dim
        value_steps[co_ndim] = value_step
        co_ndim += 1
    return out_shape_tuple, out_shape, value_steps, co_ndim, total


@register_jitable(**JIT_OPTIONS)
def binary_row_plan(left, right, unit_size):
    out_shape_tuple = np.broadcast_shapes(left.shape, right.shape)
    out_ndim = len(out_shape_tuple)
    out_shape = np.empty(out_ndim, np.intp)
    left_steps = np.empty(out_ndim, np.intp)
    right_steps = np.empty(out_ndim, np.intp)
    left_shift = out_ndim - left.ndim
    right_shift = out_ndim - right.ndim
    total = 1

    for axis in range(out_ndim):
        dim = out_shape_tuple[axis]
        left_axis = axis - left_shift
        right_axis = axis - right_shift
        left_stride = 0
        right_stride = 0
        if left_axis >= 0:
            left_stride = left.strides[left_axis] // unit_size
            if left.shape[left_axis] == 1:
                left_stride = 0
        if right_axis >= 0:
            right_stride = right.strides[right_axis] // unit_size
            if right.shape[right_axis] == 1:
                right_stride = 0
        out_shape[axis] = dim
        left_steps[axis] = left_stride
        right_steps[axis] = right_stride
        total *= dim

    co_ndim = 0
    for axis in range(out_ndim):
        dim = out_shape[axis]
        left_step = left_steps[axis]
        right_step = right_steps[axis]
        if co_ndim > 0:
            prev = co_ndim - 1
            if left_steps[prev] == left_step * dim \
                    and right_steps[prev] == right_step * dim:
                out_shape[prev] *= dim
                left_steps[prev] = left_step
                right_steps[prev] = right_step
                continue
        out_shape[co_ndim] = dim
        left_steps[co_ndim] = left_step
        right_steps[co_ndim] = right_step
        co_ndim += 1
    return out_shape_tuple, out_shape, left_steps, right_steps, co_ndim, total


@register_jitable(**JIT_OPTIONS)
def next_unary_row(coords, out_shape, value_steps, out_ndim,
                   row_value_offset):
    for axis in range(out_ndim - 2, -1, -1):
        coords[axis] += 1
        row_value_offset += value_steps[axis]
        if coords[axis] < out_shape[axis]:
            break
        coords[axis] = 0
        row_value_offset -= value_steps[axis] * out_shape[axis]
    return row_value_offset


@register_jitable(**JIT_OPTIONS)
def next_binary_row(coords, out_shape, left_steps, right_steps, out_ndim,
                    row_left_offset, row_right_offset):
    for axis in range(out_ndim - 2, -1, -1):
        coords[axis] += 1
        row_left_offset += left_steps[axis]
        row_right_offset += right_steps[axis]
        if coords[axis] < out_shape[axis]:
            break
        coords[axis] = 0
        row_left_offset -= left_steps[axis] * out_shape[axis]
        row_right_offset -= right_steps[axis] * out_shape[axis]
    return row_left_offset, row_right_offset


def ensure_slice(start, end):
    """Ensure start and end slice argument is an integer type."""
    if _NUMPY_MAJOR < 2:
        start_ok = start is None or isinstance(start, (
            int, types.Integer, types.NoneType, types.Omitted
        ))
    else:
        start_ok = isinstance(start, (int, types.Integer, types.Omitted))
    end_ok = end is None or isinstance(end, (int, types.Integer,
                                             types.NoneType, types.Omitted))
    if not start_ok or not end_ok:
        raise NumbaTypeError("slice indices must be integers or None "
                             "or have an __index__ method")
    return 0, np.iinfo(np.int64).max


def _classify_string_operand(value, exception: NumbaError = None,
                             strict=True):
    if isinstance(value, types.Array):
        if value.ndim > 1:
            raise NumbaValueError(
                'unsupported fixed-width string array dimensionality')
        value_type = value.dtype
        if isinstance(value_type, (types.CharSeq, types.UnicodeCharSeq)) \
                and value_type.count:
            return value_type, value.ndim
    elif isinstance(value, (types.CharSeq, types.UnicodeCharSeq)):
        return value, -2
    elif isinstance(value, (types.Bytes, types.UnicodeType)):
        return value, -1

    if strict and isinstance(exception, NumbaError):
        raise exception
    return None, None


def ensure_type(value, exception: NumbaError = None):
    """Ensure argument is a supported fixed-width string type."""
    return _classify_string_operand(value, exception)


def str_type(value, as_np=True):
    """Infer string-type of an objects Numba instance."""
    if isinstance(value, types.Array):
        if isinstance(value.dtype, types.CharSeq):
            return 'numpy.bytes_' if as_np else 'bytes'
        if isinstance(value.dtype, types.UnicodeCharSeq):
            return 'numpy.str_' if as_np else 'str'
        return f'like {value.dtype.name}'

    if isinstance(value, types.Bytes):
        return 'numpy.bytes_' if as_np else 'bytes'
    if isinstance(value, types.UnicodeType):
        return 'numpy.str_' if as_np else 'str'
    return f'like {value.name}'


def _array_register(value, ndim, contiguous, strided, scalar):
    if ndim < 0:
        return scalar
    if isinstance(value, types.Array) and value.layout != 'C':
        return strided
    return contiguous


def register_pair(left, right, exception: (NumbaError, int) = None):
    """Choose ordinal registration functions for a pair of string operands."""
    error = exception or NumbaTypeError("comparison of non-string arrays")
    left_type, left_dim = ensure_type(left, error)
    right_type, right_dim = ensure_type(right, error)

    byte_types = (types.Bytes, types.CharSeq)
    str_types = (types.UnicodeType, types.UnicodeCharSeq)

    if isinstance(left_type, byte_types) \
            and isinstance(right_type, byte_types):
        register_left = _array_register(
            left, left_dim, register_array_bytes,
            register_array_bytes_strided, register_scalar_bytes)
        register_right = _array_register(
            right, right_dim, register_array_bytes,
            register_array_bytes_strided, register_scalar_bytes)
    elif isinstance(left_type, str_types) \
            and isinstance(right_type, str_types):
        register_left = _array_register(
            left, left_dim, register_array_strings,
            register_array_strings_strided, register_scalar_strings)
        register_right = _array_register(
            right, right_dim, register_array_strings,
            register_array_strings_strided, register_scalar_strings)
    else:
        if exception == 1:
            as_type = str_type(left, as_np=False)
            if as_type in ('str', 'bytes'):
                error = NumbaTypeError(
                    f"must be {as_type}, not {str_type(right)}")
            else:
                error = NumbaTypeError("string operation on non-string array")
        else:
            error = NumbaNotImplementedError('NotImplemented')
        raise error
    return register_left, register_right, left_dim, right_dim


def _string_type(value):
    """Return a string operand's dtype/type and dimension, or (None, None)."""
    return _classify_string_operand(value, strict=False)


def fixed_width_array(value):
    if isinstance(value, types.Array):
        if isinstance(value.dtype, types.CharSeq):
            return 'bytes', value.dtype.count
        if isinstance(value.dtype, types.UnicodeCharSeq):
            return 'unicode', value.dtype.count
    return None, 0


def fixed_width_array_pair(left, right):
    left_kind, left_width = fixed_width_array(left)
    right_kind, right_width = fixed_width_array(right)
    if left_kind is None or left_kind != right_kind:
        return None, 0, 0
    return left_kind, left_width, right_width


def needs_nd_path(*values):
    for value in values:
        if isinstance(value, types.Array) and value.ndim > 1:
            return True
    return False


def try_register_pair(left, right):
    """Choose ordinal registration for string operands, else return None."""
    left_type, left_dim = _string_type(left)
    right_type, right_dim = _string_type(right)

    if left_type is None or right_type is None:
        return None

    byte_types = (types.Bytes, types.CharSeq)
    str_types = (types.UnicodeType, types.UnicodeCharSeq)

    if isinstance(left_type, byte_types) \
            and isinstance(right_type, byte_types):
        register_left = _array_register(
            left, left_dim, register_array_bytes,
            register_array_bytes_strided, register_scalar_bytes)
        register_right = _array_register(
            right, right_dim, register_array_bytes,
            register_array_bytes_strided, register_scalar_bytes)
    elif isinstance(left_type, str_types) \
            and isinstance(right_type, str_types):
        register_left = _array_register(
            left, left_dim, register_array_strings,
            register_array_strings_strided, register_scalar_strings)
        register_right = _array_register(
            right, right_dim, register_array_strings,
            register_array_strings_strided, register_scalar_strings)
    else:
        return None
    return register_left, register_right, left_dim, right_dim


def register_single(value, exception: NumbaError = None):
    """Choose ordinal registration function for one string operand."""
    error = exception or NumbaTypeError("string operation on non-string array")
    value_type, value_dim = ensure_type(value, error)

    byte_types = (types.Bytes, types.CharSeq)
    str_types = (types.UnicodeType, types.UnicodeCharSeq)

    if isinstance(value_type, byte_types):
        register_value = _array_register(
            value, value_dim, register_array_bytes,
            register_array_bytes_strided, register_scalar_bytes)
        as_bytes = True
    elif isinstance(value_type, str_types):
        register_value = _array_register(
            value, value_dim, register_array_strings,
            register_array_strings_strided, register_scalar_strings)
        as_bytes = False
    else:
        raise error
    return register_value, value_dim, as_bytes


def equal_kernel(left, right, generic, bytes_sub32, unicode_sub32):
    """Choose the equality kernel for fixed-width operands."""
    kind_left, width_left = fixed_width_array(left)
    kind_right, width_right = fixed_width_array(right)
    array_count = int(width_left > 0) + int(width_right > 0)
    width = max(width_left, width_right)
    same_width = array_count == 2 and kind_left == kind_right \
        and width_left == width_right

    if same_width and width < 32:
        if kind_left == 'bytes':
            return bytes_sub32
        if kind_left == 'unicode':
            return unicode_sub32
    return generic


def equal_dispatch(register_left, register_right, left_dim, right_dim,
                   equal_impl, rstrip, invert=False, scalar_as_array=False):
    left_scalar_like = left_dim <= 0 < right_dim

    if left_dim > 0 or right_dim > 0:
        def impl(left, right):
            if left_scalar_like:
                result = equal_impl(*register_right(right, False),
                                    *register_left(left, False), rstrip)
            else:
                result = equal_impl(*register_left(left, False),
                                    *register_right(right, False), rstrip)
            return ~result if invert else result
    else:
        def impl(left, right):
            result = equal_impl(*register_left(left, False),
                                *register_right(right, False), rstrip)[0]
            result = ~result if invert else result
            return np.array(result) if scalar_as_array else result
    return impl


def order_dispatch(register_left, register_right, left_dim, right_dim,
                   greater_impl, greater_equal_impl, op, rstrip,
                   scalar_as_array=False):
    left_scalar_like = left_dim <= 0 < right_dim

    if op == 'greater_equal':
        kernel = greater_equal_impl
        invert_result = False
    elif op == 'greater':
        kernel = greater_impl
        invert_result = False
    elif op == 'less':
        kernel = greater_equal_impl
        invert_result = True
    else:
        kernel = greater_impl
        invert_result = True

    if left_dim > 0 or right_dim > 0:
        def impl(left, right):
            if left_scalar_like:
                result = kernel(*register_right(right, False),
                                *register_left(left, False), True, rstrip)
            else:
                result = kernel(*register_left(left, False),
                                *register_right(right, False), False, rstrip)
            return ~result if invert_result else result
    else:
        def impl(left, right):
            result = kernel(*register_left(left, False),
                            *register_right(right, False), False, rstrip)[0]
            result = ~result if invert_result else result
            return np.array(result) if scalar_as_array else result
    return impl
