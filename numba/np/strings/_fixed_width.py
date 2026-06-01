"""Raw fixed-width string kernels for strided and N-D arrays."""

from numba.np.strings import JIT_OPTIONS
from numba.np.strings._kernels import (
    _is_simple_property_ord, _islower_ord, _istitle_only_ord, _isupper_ord,
    _rstrip_ord,
)
from numba.np.strings._shared import (
    binary_row_plan, fixed_width_array, fixed_width_array_pair,
    needs_nd_path, next_binary_row, next_unary_row, unary_row_plan,
)
from llvmlite import ir
from numba import types
from numba.cpython.charseq import charseq_get_code, unicode_charseq_get_code
from numba.core import cgutils
from numba.extending import intrinsic, overload, register_jitable
import numpy as np


LOOP_JIT_OPTIONS = dict(JIT_OPTIONS, forceinline=False)


OP_STARTSWITH = 1
OP_ENDSWITH = 2
OP_FIND = 3
OP_RFIND = 4
OP_COUNT = 5
OP_INDEX = 6
OP_RINDEX = 7
OP_EQUAL = 8
OP_GREATER = 9
OP_GREATER_EQUAL = 10
OP_LESS = 11
OP_LESS_EQUAL = 12
OP_ISALPHA = 13
OP_ISALNUM = 14
OP_ISDECIMAL = 15
OP_ISDIGIT = 16
OP_ISNUMERIC = 17
OP_ISSPACE = 18
OP_ISLOWER = 19
OP_ISUPPER = 20
OP_ISTITLE = 21


@intrinsic
def _fixed_ord_array(typingctx, array, offset, unit):
    """Load one code unit from a fixed-width record at a byte offset."""
    if not isinstance(array, types.Array):
        raise TypeError('fixed-width string array required')

    if isinstance(array.dtype, types.CharSeq):
        unit_bytes = 1
        load_type = ir.IntType(8)
    elif isinstance(array.dtype, types.UnicodeCharSeq):
        unit_bytes = 4
        load_type = ir.IntType(32)
    else:
        raise TypeError('fixed-width string array required')

    def codegen(context, builder, signature, args):
        array_type, _, _ = signature.args
        array_value, offset_value, unit_value = args
        array_struct = context.make_array(array_type)(
            context, builder, array_value)
        unit_offset = builder.mul(
            unit_value, ir.Constant(unit_value.type, unit_bytes))
        offset_value = builder.add(offset_value, unit_offset)
        ptr = cgutils.pointer_add(
            builder, array_struct.data, offset_value, cgutils.voidptr_t)
        value = builder.load(builder.bitcast(ptr, load_type.as_pointer()),
                             align=1)
        result_type = context.get_value_type(types.int32)
        if load_type.width < result_type.width:
            value = builder.zext(value, result_type)
        return value

    return types.int32(array, offset, unit), codegen


def _fixed_ord(value, offset, unit):
    return 0


@overload(_fixed_ord)
def ov_fixed_ord(value, offset, unit):
    if isinstance(value, types.Array):
        def impl(value, offset, unit):
            return _fixed_ord_array(value, offset, unit)
        return impl

    if isinstance(value, types.Bytes):
        def impl(value, offset, unit):
            return value[unit]
        return impl

    if isinstance(value, types.CharSeq):
        def impl(value, offset, unit):
            return charseq_get_code(value, unit)
        return impl

    if isinstance(value, types.UnicodeCharSeq):
        def impl(value, offset, unit):
            return unicode_charseq_get_code(value, unit)
        return impl

    if isinstance(value, types.UnicodeType):
        def impl(value, offset, unit):
            return ord(str(value[unit]))
        return impl


@register_jitable(**JIT_OPTIONS)
def _fixed_record_len(values, offset, width):
    for p in range(width - 1, -1, -1):
        if _fixed_ord(values, offset, p) != 0:
            return p + 1
    return 0


@register_jitable(**JIT_OPTIONS)
def _fixed_record_compare_len(values, offset, width, rstrip):
    p = width - 1
    if rstrip:
        while p >= 0 and _rstrip_ord(_fixed_ord(values, offset, p)):
            p -= 1
    else:
        while p >= 0 and _fixed_ord(values, offset, p) == 0:
            p -= 1
    return p + 1


@register_jitable(**JIT_OPTIONS)
def _fixed_slice_bounds(n_value, start, end):
    if start < 0:
        start = start + n_value
    if start < 0:
        start = 0
    if end < 0:
        end = end + n_value
    if end < 0:
        end = 0
    if end > n_value:
        end = n_value
    return start, end


@register_jitable(**JIT_OPTIONS)
def _fixed_record_startswith(values, patterns, value_offset, pattern_offset,
                             width, pattern_width, start, end):
    n_value = _fixed_record_len(values, value_offset, width)
    n_pattern = _fixed_record_len(patterns, pattern_offset, pattern_width)
    start, end = _fixed_slice_bounds(n_value, start, end)
    if start + n_pattern > end:
        return n_pattern == 0 and start <= end
    for p in range(n_pattern):
        if _fixed_ord(values, value_offset, start + p) \
                != _fixed_ord(patterns, pattern_offset, p):
            return False
    return True


@register_jitable(**JIT_OPTIONS)
def _fixed_record_endswith(values, patterns, value_offset, pattern_offset,
                           width, pattern_width, start, end):
    n_value = _fixed_record_len(values, value_offset, width)
    n_pattern = _fixed_record_len(patterns, pattern_offset, pattern_width)
    start, end = _fixed_slice_bounds(n_value, start, end)
    if start + n_pattern > end:
        return n_pattern == 0 and start <= end
    v = end - 1
    p = n_pattern - 1
    while p >= 0:
        if _fixed_ord(values, value_offset, v) \
                != _fixed_ord(patterns, pattern_offset, p):
            return False
        v -= 1
        p -= 1
    return True


@register_jitable(**JIT_OPTIONS)
def _fixed_record_find(values, patterns, value_offset, pattern_offset,
                       width, pattern_width, start, end):
    n_value = _fixed_record_len(values, value_offset, width)
    n_pattern = _fixed_record_len(patterns, pattern_offset, pattern_width)
    start, end = _fixed_slice_bounds(n_value, start, end)
    if n_pattern == 0:
        return start if start <= end else -1
    first = _fixed_ord(patterns, pattern_offset, 0)
    while start + n_pattern <= end:
        if _fixed_ord(values, value_offset, start) != first:
            start += 1
            continue
        matched = True
        for p in range(1, n_pattern):
            if _fixed_ord(values, value_offset, start + p) \
                    != _fixed_ord(patterns, pattern_offset, p):
                matched = False
                break
        if matched:
            return start
        start += 1
    return -1


@register_jitable(**JIT_OPTIONS)
def _fixed_record_rfind(values, patterns, value_offset, pattern_offset,
                        width, pattern_width, start, end):
    n_value = _fixed_record_len(values, value_offset, width)
    n_pattern = _fixed_record_len(patterns, pattern_offset, pattern_width)
    start, end = _fixed_slice_bounds(n_value, start, end)
    if n_pattern == 0:
        return end if start <= end else -1
    last_pos = n_pattern - 1
    last = _fixed_ord(patterns, pattern_offset, last_pos)
    candidate = end - n_pattern
    while candidate >= start:
        if _fixed_ord(values, value_offset, candidate + last_pos) \
                != last:
            candidate -= 1
            continue
        matched = True
        p = last_pos - 1
        while p >= 0:
            if _fixed_ord(values, value_offset, candidate + p) \
                    != _fixed_ord(patterns, pattern_offset, p):
                matched = False
                break
            p -= 1
        if matched:
            return candidate
        candidate -= 1
    return -1


@register_jitable(**JIT_OPTIONS)
def _fixed_record_count(values, patterns, value_offset, pattern_offset,
                        width, pattern_width, start, end):
    n_value = _fixed_record_len(values, value_offset, width)
    n_pattern = _fixed_record_len(patterns, pattern_offset, pattern_width)
    start, end = _fixed_slice_bounds(n_value, start, end)
    if n_pattern == 0:
        return max(1 + end - start, 1) if start <= end else 0
    first = _fixed_ord(patterns, pattern_offset, 0)
    count = 0
    while start + n_pattern <= end:
        if _fixed_ord(values, value_offset, start) != first:
            start += 1
            continue
        matched = True
        for p in range(1, n_pattern):
            if _fixed_ord(values, value_offset, start + p) \
                    != _fixed_ord(patterns, pattern_offset, p):
                matched = False
                break
        if matched:
            count += 1
            start += n_pattern
        else:
            start += 1
    return count


@register_jitable(**JIT_OPTIONS)
def _fixed_record_equal(values, patterns, value_offset, pattern_offset,
                        width, pattern_width, rstrip):
    if not rstrip:
        common = width if width < pattern_width else pattern_width
        for p in range(common):
            if _fixed_ord(values, value_offset, p) \
                    != _fixed_ord(patterns, pattern_offset, p):
                return False
        for p in range(common, width):
            if _fixed_ord(values, value_offset, p) != 0:
                return False
        for p in range(common, pattern_width):
            if _fixed_ord(patterns, pattern_offset, p) != 0:
                return False
        return True

    n_value = _fixed_record_compare_len(values, value_offset, width, rstrip)
    n_pattern = _fixed_record_compare_len(
        patterns, pattern_offset, pattern_width, rstrip)
    if n_value != n_pattern:
        return False
    for p in range(n_value):
        if _fixed_ord(values, value_offset, p) \
                != _fixed_ord(patterns, pattern_offset, p):
            return False
    return True


@register_jitable(**JIT_OPTIONS)
def _fixed_trim_suffix(values, offset, start, end, rstrip):
    for p in range(start, end):
        ord_value = _fixed_ord(values, offset, p)
        if not (ord_value == 0 or (rstrip and _rstrip_ord(ord_value))):
            return False
    return True


@register_jitable(**JIT_OPTIONS)
def _fixed_record_compare(values, patterns, value_offset, pattern_offset,
                          width, pattern_width, rstrip):
    common = width if width < pattern_width else pattern_width
    for p in range(common):
        left = _fixed_ord(values, value_offset, p)
        right = _fixed_ord(patterns, pattern_offset, p)
        if left != right:
            if rstrip:
                left_trimmed = _rstrip_ord(left) and _fixed_trim_suffix(
                    values, value_offset, p + 1, width, rstrip)
                right_trimmed = _rstrip_ord(right) and _fixed_trim_suffix(
                    patterns, pattern_offset, p + 1, pattern_width, rstrip)
                if left_trimmed and right_trimmed:
                    return 0
                if left_trimmed:
                    return -1
                if right_trimmed:
                    return 1
            return left - right
    if width == pattern_width:
        return 0
    if width < pattern_width:
        if _fixed_trim_suffix(
                patterns, pattern_offset, common, pattern_width, rstrip):
            return 0
        return -1
    if _fixed_trim_suffix(values, value_offset, common, width, rstrip):
        return 0
    return 1


@register_jitable(**JIT_OPTIONS)
def _fixed_order_result(cmp_result, op):
    if op == OP_GREATER:
        return cmp_result > 0
    if op == OP_GREATER_EQUAL:
        return cmp_result >= 0
    if op == OP_LESS:
        return cmp_result < 0
    return cmp_result <= 0


@register_jitable(**JIT_OPTIONS)
def _fixed_record_property(values, offset, width, as_bytes, op):
    if not width:
        return False

    if op <= OP_ISSPACE:
        seen = False
        for p in range(width):
            chr_ord = _fixed_ord(values, offset, p)
            if chr_ord == 0:
                if _fixed_trim_suffix(values, offset, p + 1, width, False):
                    break
                return False
            kind = op - OP_ISALPHA
            if not _is_simple_property_ord(chr_ord, as_bytes, kind):
                return False
            seen = True
        return seen

    size = _fixed_record_len(values, offset, width)
    seen = False
    cased_state = False
    for p in range(size):
        chr_ord = _fixed_ord(values, offset, p)
        is_lower = _islower_ord(chr_ord, as_bytes)
        is_upper = _isupper_ord(chr_ord, as_bytes)
        is_title = _istitle_only_ord(chr_ord, as_bytes)

        if op == OP_ISLOWER:
            if is_upper or is_title:
                return False
            seen |= is_lower
        elif op == OP_ISUPPER:
            if is_lower or is_title:
                return False
            seen |= is_upper
        else:
            is_start = is_upper or is_title
            if cased_state:
                if is_start:
                    return False
                cased_state = is_lower
            else:
                if is_lower:
                    return False
                cased_state = is_start
                seen |= cased_state
    return seen


@register_jitable(**JIT_OPTIONS)
def _fixed_record_apply_bool(values, patterns, value_offset, pattern_offset,
                             width, pattern_width, start, end, op, rstrip):
    if op == OP_STARTSWITH:
        return _fixed_record_startswith(
            values, patterns, value_offset, pattern_offset,
            width, pattern_width, start, end)
    if op == OP_ENDSWITH:
        return _fixed_record_endswith(
            values, patterns, value_offset, pattern_offset,
            width, pattern_width, start, end)
    if op == OP_EQUAL:
        return _fixed_record_equal(
            values, patterns, value_offset, pattern_offset,
            width, pattern_width, rstrip)
    return _fixed_order_result(
        _fixed_record_compare(
            values, patterns, value_offset, pattern_offset,
            width, pattern_width, rstrip),
        op)


@register_jitable(**JIT_OPTIONS)
def _fixed_record_apply_int(values, patterns, value_offset, pattern_offset,
                            width, pattern_width, start, end, op):
    if op == OP_FIND or op == OP_INDEX:
        return _fixed_record_find(
            values, patterns, value_offset, pattern_offset,
            width, pattern_width, start, end)
    if op == OP_RFIND or op == OP_RINDEX:
        return _fixed_record_rfind(
            values, patterns, value_offset, pattern_offset,
            width, pattern_width, start, end)
    if op == OP_COUNT:
        return _fixed_record_count(
            values, patterns, value_offset, pattern_offset,
            width, pattern_width, start, end)
    return -1


@register_jitable(**LOOP_JIT_OPTIONS)
def _fixed_binary_bool_nd(values, patterns, width, pattern_width,
                          start, end, op, rstrip):
    out_shape_tuple, out_shape, value_steps, pattern_steps, out_ndim, total = \
        binary_row_plan(values, patterns, 1)
    bool_result = np.empty(total, np.bool_)
    if op != OP_EQUAL and (start > width or start > end + width):
        for i in range(total):
            bool_result[i] = False
        return bool_result.reshape(out_shape_tuple)

    _fixed_binary_bool_loop(
        values, patterns, width, pattern_width, start, end, op, rstrip,
        out_shape, value_steps, pattern_steps, out_ndim, total, bool_result)
    return bool_result.reshape(out_shape_tuple)


@register_jitable(**JIT_OPTIONS)
def _fixed_binary_bool_loop(values, patterns, width, pattern_width,
                            start, end, op, rstrip, out_shape, value_steps,
                            pattern_steps, out_ndim, total, result):
    if out_ndim == 0:
        result[0] = _fixed_record_apply_bool(
            values, patterns, 0, 0, width, pattern_width,
            start, end, op, rstrip)
        return

    inner = out_shape[out_ndim - 1]
    inner_value_step = value_steps[out_ndim - 1]
    inner_pattern_step = pattern_steps[out_ndim - 1]
    coords = np.zeros(out_ndim, np.intp)
    result_i = 0
    row_value_offset = 0
    row_pattern_offset = 0
    while result_i < total:
        value_offset = row_value_offset
        pattern_offset = row_pattern_offset
        for _ in range(inner):
            result[result_i] = _fixed_record_apply_bool(
                values, patterns, value_offset, pattern_offset,
                width, pattern_width, start, end, op, rstrip)
            result_i += 1
            value_offset += inner_value_step
            pattern_offset += inner_pattern_step

        row_value_offset, row_pattern_offset = next_binary_row(
            coords, out_shape, value_steps, pattern_steps, out_ndim,
            row_value_offset, row_pattern_offset)


@register_jitable(**JIT_OPTIONS)
def _fixed_binary_int_nd(values, patterns, width, pattern_width,
                         start, end, op):
    out_shape_tuple, out_shape, value_steps, pattern_steps, out_ndim, total = \
        binary_row_plan(values, patterns, 1)
    int_result = np.empty(total, np.int64)
    if start > width or start > end + width:
        if op == OP_INDEX or op == OP_RINDEX:
            raise ValueError('substring not found')
        fill = 0 if op == OP_COUNT else -1
        for i in range(total):
            int_result[i] = fill
        return int_result.reshape(out_shape_tuple)
    _fixed_binary_int_loop(
        values, patterns, width, pattern_width, start, end, op,
        out_shape, value_steps, pattern_steps, out_ndim, total, int_result)
    return int_result.reshape(out_shape_tuple)


@register_jitable(**JIT_OPTIONS)
def _fixed_binary_int_loop(values, patterns, width, pattern_width,
                           start, end, op, out_shape, value_steps,
                           pattern_steps, out_ndim, total, result):
    if out_ndim == 0:
        found = _fixed_record_apply_int(
            values, patterns, 0, 0, width, pattern_width,
            start, end, op)
        if found < 0 and (op == OP_INDEX or op == OP_RINDEX):
            raise ValueError('substring not found')
        result[0] = found
        return

    inner = out_shape[out_ndim - 1]
    inner_value_step = value_steps[out_ndim - 1]
    inner_pattern_step = pattern_steps[out_ndim - 1]
    coords = np.zeros(out_ndim, np.intp)
    result_i = 0
    row_value_offset = 0
    row_pattern_offset = 0
    while result_i < total:
        value_offset = row_value_offset
        pattern_offset = row_pattern_offset
        for _ in range(inner):
            found = _fixed_record_apply_int(
                values, patterns, value_offset, pattern_offset,
                width, pattern_width, start, end, op)
            if found < 0 and (op == OP_INDEX or op == OP_RINDEX):
                raise ValueError('substring not found')
            result[result_i] = found
            result_i += 1
            value_offset += inner_value_step
            pattern_offset += inner_pattern_step

        row_value_offset, row_pattern_offset = next_binary_row(
            coords, out_shape, value_steps, pattern_steps, out_ndim,
            row_value_offset, row_pattern_offset)


@register_jitable(**LOOP_JIT_OPTIONS)
def _fixed_scalar_mixed_bool_nd(array_value, scalar_value, array_width,
                                scalar_width, start, end, op, rstrip,
                                scalar_left):
    out_shape_tuple, out_shape, array_steps, out_ndim, total = \
        unary_row_plan(array_value, 1)
    result = np.empty(total, np.bool_)
    value_width = scalar_width if scalar_left else array_width
    if op != OP_EQUAL and (start > value_width or start > end + value_width):
        for i in range(total):
            result[i] = False
        return result.reshape(out_shape_tuple)

    scalar_steps = np.zeros(out_ndim, np.intp)
    if scalar_left:
        _fixed_binary_bool_loop(
            scalar_value, array_value, scalar_width, array_width, start, end,
            op, rstrip, out_shape, scalar_steps, array_steps, out_ndim, total,
            result)
    else:
        _fixed_binary_bool_loop(
            array_value, scalar_value, array_width, scalar_width, start, end,
            op, rstrip, out_shape, array_steps, scalar_steps, out_ndim, total,
            result)
    return result.reshape(out_shape_tuple)


@register_jitable(**JIT_OPTIONS)
def _fixed_scalar_mixed_int_nd(array_value, scalar_value, array_width,
                               scalar_width, start, end, op, scalar_left):
    out_shape_tuple, out_shape, array_steps, out_ndim, total = \
        unary_row_plan(array_value, 1)
    result = np.empty(total, np.int64)
    value_width = scalar_width if scalar_left else array_width
    if start > value_width or start > end + value_width:
        if op == OP_INDEX or op == OP_RINDEX:
            raise ValueError('substring not found')
        fill = 0 if op == OP_COUNT else -1
        for i in range(total):
            result[i] = fill
        return result.reshape(out_shape_tuple)

    scalar_steps = np.zeros(out_ndim, np.intp)
    if scalar_left:
        _fixed_binary_int_loop(
            scalar_value, array_value, scalar_width, array_width, start, end,
            op, out_shape, scalar_steps, array_steps, out_ndim, total, result)
    else:
        _fixed_binary_int_loop(
            array_value, scalar_value, array_width, scalar_width, start, end,
            op, out_shape, array_steps, scalar_steps, out_ndim, total, result)
    return result.reshape(out_shape_tuple)


@register_jitable(**JIT_OPTIONS)
def fixed_str_len_nd(values, width):
    out_shape_tuple, out_shape, value_steps, out_ndim, total = \
        unary_row_plan(values, 1)
    result = np.empty(total, np.int64)

    if out_ndim == 0:
        result[0] = _fixed_record_len(values, 0, width)
        return result.reshape(out_shape_tuple)

    inner = out_shape[out_ndim - 1]
    inner_value_step = value_steps[out_ndim - 1]
    coords = np.zeros(out_ndim, np.intp)
    result_i = 0
    row_value_offset = 0
    while result_i < total:
        value_offset = row_value_offset
        for _ in range(inner):
            result[result_i] = _fixed_record_len(values, value_offset, width)
            result_i += 1
            value_offset += inner_value_step

        row_value_offset = next_unary_row(
            coords, out_shape, value_steps, out_ndim, row_value_offset)

    return result.reshape(out_shape_tuple)


@register_jitable(**JIT_OPTIONS)
def fixed_property_nd(values, width, as_bytes, op):
    out_shape_tuple, out_shape, value_steps, out_ndim, total = \
        unary_row_plan(values, 1)
    result = np.empty(total, np.bool_)

    if out_ndim == 0:
        result[0] = _fixed_record_property(values, 0, width, as_bytes, op)
        return result.reshape(out_shape_tuple)

    inner = out_shape[out_ndim - 1]
    inner_value_step = value_steps[out_ndim - 1]
    coords = np.zeros(out_ndim, np.intp)
    result_i = 0
    row_value_offset = 0
    while result_i < total:
        value_offset = row_value_offset
        for _ in range(inner):
            result[result_i] = _fixed_record_property(
                values, value_offset, width, as_bytes, op)
            result_i += 1
            value_offset += inner_value_step

        row_value_offset = next_unary_row(
            coords, out_shape, value_steps, out_ndim, row_value_offset)

    return result.reshape(out_shape_tuple)
