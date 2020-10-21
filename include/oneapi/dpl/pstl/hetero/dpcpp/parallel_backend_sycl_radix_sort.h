// -*- C++ -*-
//===-- parallel_backend_sycl_radix_sort.h --------------------------------===//
//
// Copyright (C) 2020 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This file incorporates work covered by the following copyright and permission
// notice:
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef _ONEDPL_parallel_backend_sycl_radix_sort_H
#define _ONEDPL_parallel_backend_sycl_radix_sort_H

#include <CL/sycl.hpp>
#include <climits>

#include "parallel_backend_sycl_utils.h"
#include "execution_sycl_defs.h"

namespace oneapi
{
namespace dpl
{
namespace __par_backend_hetero
{

namespace sycl = cl::sycl;

//------------------------------------------------------------------------
// radix sort: kernel names
//------------------------------------------------------------------------

template <typename _DerivedKernelName>
class __kernel_name_base;

template <typename... _Name>
class __radix_sort_count_kernel : public __kernel_name_base<__radix_sort_count_kernel<_Name...>>
{
};

template <typename... _Name>
class __radix_sort_scan_kernel : public __kernel_name_base<__radix_sort_scan_kernel<_Name...>>
{
};

template <typename... _Name>
class __radix_sort_reorder_kernel : public __kernel_name_base<__radix_sort_reorder_kernel<_Name...>>
{
};

//------------------------------------------------------------------------
// radix sort: ordered traits for a given size and integral/float flag
//------------------------------------------------------------------------

template <::std::size_t __type_size, bool __is_integral_type>
struct __get_ordered
{
};

template <>
struct __get_ordered<1, true>
{
    using _type = uint8_t;
    constexpr static ::std::int8_t __mask = 0x80;
};

template <>
struct __get_ordered<2, true>
{
    using _type = uint16_t;
    constexpr static ::std::int16_t __mask = 0x8000;
};

template <>
struct __get_ordered<4, true>
{
    using _type = uint32_t;
    constexpr static ::std::int32_t __mask = 0x80000000;
};

template <>
struct __get_ordered<8, true>
{
    using _type = uint64_t;
    constexpr static ::std::int64_t __mask = 0x8000000000000000;
};

template <>
struct __get_ordered<4, false>
{
    using _type = uint64_t;
    constexpr static ::std::uint64_t __nmask = 0xFFFFFFFFull; // for negative numbers
    constexpr static ::std::uint64_t __pmask = 0x80000000ull; // for positive numbers
};

template <>
struct __get_ordered<8, false>
{
    using _type = uint64_t;
    constexpr static ::std::uint64_t __nmask = 0xFFFFFFFFFFFFFFFFull; // for negative numbers
    constexpr static ::std::uint64_t __pmask = 0x8000000000000000ull; // for positive numbers
};

//------------------------------------------------------------------------
// radix sort: ordered type for a given type
//------------------------------------------------------------------------

// for unknown/unsupported type we do not have any trait
template <typename _T, typename _Dummy = void>
struct __ordered
{
};

// for unsigned integrals we use the same type
template <typename _T>
struct __ordered<_T, __enable_if_t<::std::is_integral<_T>::value&& ::std::is_unsigned<_T>::value>>
{
    using _type = _T;
};

// for signed integrals or floatings we map: size -> corresponding unsigned integral
template <typename _T>
struct __ordered<_T, __enable_if_t<(::std::is_integral<_T>::value && ::std::is_signed<_T>::value) ||
                                   ::std::is_floating_point<_T>::value>>
{
    using _type = typename __get_ordered<sizeof(_T), ::std::is_integral<_T>::value>::_type;
};

// shorthands
template <typename _T>
using __ordered_t = typename __ordered<_T>::_type;

//------------------------------------------------------------------------
// radix sort: functions for conversion to ordered type
//------------------------------------------------------------------------

// for already ordered types (any uints) we use the same type
template <typename _T>
inline __enable_if_t<::std::is_same<_T, __ordered_t<_T>>::value, __ordered_t<_T>>
__convert_to_ordered(_T __value)
{
    return __value;
}

// converts integral type to ordered (in terms of bitness) type
template <typename _T>
inline __enable_if_t<!::std::is_same<_T, __ordered_t<_T>>::value && !::std::is_floating_point<_T>::value,
                     __ordered_t<_T>>
__convert_to_ordered(_T __value)
{
    _T __result = __value ^ __get_ordered<sizeof(_T), true>::__mask;
    return *reinterpret_cast<__ordered_t<_T>*>(&__result);
}

// converts floating type to ordered (in terms of bitness) type
template <typename _T>
inline __enable_if_t<!::std::is_same<_T, __ordered_t<_T>>::value && ::std::is_floating_point<_T>::value,
                     __ordered_t<_T>>
__convert_to_ordered(_T __value)
{
    // represent as uint64_t
    __ordered_t<_T> __uvalue = *reinterpret_cast<__ordered_t<_T>*>(&__value);
    // check if value negative
    __ordered_t<_T> __is_negative = __uvalue >> (sizeof(_T) * CHAR_BIT - 1);
    // for positive: 00..00 -> 00..00 -> 10..00
    // for negative: 00..01 -> 11..11 -> 11..11
    __ordered_t<_T> __ordered_mask =
        (__is_negative * __get_ordered<sizeof(_T), false>::__nmask) | __get_ordered<sizeof(_T), false>::__pmask;
    return __uvalue ^ __ordered_mask;
}

//------------------------------------------------------------------------
// radix sort: run-time device info functions
//------------------------------------------------------------------------

// get item id in sub-group
inline ::std::uint32_t
__get_sg_item_idx(const sycl::nd_item<1>& __idx)
{
    // technically sycl::id<1>::operator[int] returns a value that always fits in uint8_t (no overflow)
    // and since 64-bit arithmetic is more expensive, the return type is set to ::std::uint32_t
    return static_cast<::std::uint32_t>(__idx.get_sub_group().get_local_id()[0]);
}

// get number of items in sub-group
inline ::std::uint32_t
__get_sg_item_num(const sycl::nd_item<1>& __idx)
{
    return __idx.get_sub_group().get_local_range()[0];
}

// get rounded up result of (__number / __divisor)
template <typename _T1, typename _T2>
inline auto
__get_roundedup_div(_T1 __number, _T2 __divisor) -> decltype((__number - 1) / __divisor + 1)
{
    return (__number - 1) / __divisor + 1;
}

//------------------------------------------------------------------------
// radix sort: bit pattern functions
//------------------------------------------------------------------------

// get number of states radix bits can represent
constexpr ::std::uint32_t
__get_states_in_bits(::std::uint32_t __radix_bits)
{
    return (1 << __radix_bits);
}

// get number of buckets (size of radix bits) in T
template <typename _T>
constexpr ::std::uint32_t
__get_buckets_in_type(::std::uint32_t __radix_bits)
{
    return (sizeof(_T) * CHAR_BIT) / __radix_bits;
}

// required for descending comparator support
template <bool __flag>
struct __invert_if
{
    template <typename _T>
    _T
    operator()(_T __value)
    {
        return __value;
    }
};

// invert value if descending comparator is passed
template <>
struct __invert_if<true>
{
    template <typename _T>
    _T
    operator()(_T __value)
    {
        return ~__value;
    }

    // invertation for bool type have to be logical, rather than bit
    bool
    operator()(bool __value)
    {
        return !__value;
    }
};

// get bit values in a certain bucket of a value
template <::std::uint32_t __radix_bits, bool __is_comp_asc, typename _T>
::std::uint32_t
__get_bucket_value(_T __value, ::std::uint32_t __radix_iter)
{
    // invert value if we need to sort in descending order
    __value = __invert_if<!__is_comp_asc>{}(__value);

    // get bucket offset idx from the end of bit type (least significant bits)
    ::std::uint32_t __bucket_offset = __radix_iter * __radix_bits;

    // get offset mask for one bucket, e.g.
    // radix_bits=2: 0000 0001 -> 0000 0100 -> 0000 0011
    __ordered_t<_T> __bucket_mask = (1u << __radix_bits) - 1u;

    // get bits under bucket mask
    return (__value >> __bucket_offset) & __bucket_mask;
}

//-----------------------------------------------------------------------
// radix sort: count kernel (per iteration)
//-----------------------------------------------------------------------

template <typename _KernelName, typename _Iterator, ::std::uint32_t __radix_bits, bool __is_comp_asc,
          typename _ExecutionPolicy, typename _ValBuf, typename _CountBuf
#if _PSTL_COMPILE_KERNEL
          ,
          typename _Kernel
#endif
          >
sycl::event
__radix_sort_count_submit(_ExecutionPolicy&& __exec, ::std::size_t __segments, ::std::size_t __block_size,
                          ::std::uint32_t __radix_iter, _ValBuf& __val_buf, ::std::size_t __val_buf_size,
                          _CountBuf& __count_buf, ::std::size_t __count_buf_size, sycl::event __dependency_event
#if _PSTL_COMPILE_KERNEL
                          ,
                          _Kernel& __kernel
#endif
)
{
    // typedefs
    using _ValueT = __value_t<_ValBuf>;
    using _CountT = __value_t<_CountBuf>;

    // radix states used for an array storing bucket state counters
    const ::std::uint32_t __radix_states = __get_states_in_bits(__radix_bits);

    // correct __block_size according to local memory limit
    const auto __max_allocation_size = oneapi::dpl::__internal::__max_local_allocation_size<_ExecutionPolicy, _CountT>(
        ::std::forward<_ExecutionPolicy>(__exec), __block_size * __radix_states);
    __block_size = __get_roundedup_div(__max_allocation_size, __radix_states);

    // iteration space info
    const ::std::size_t __blocks_total = __get_roundedup_div(__val_buf_size, __block_size);
    const ::std::size_t __blocks_per_segment = __get_roundedup_div(__blocks_total, __segments);

    // submit to compute arrays with local count values
    sycl::event __count_levent = __exec.queue().submit([&](sycl::handler& __hdl) {
        __hdl.depends_on(__dependency_event);

        // an accessor with values to count
        auto __val_gacc = __internal::get_access<_Iterator>(__hdl)(__val_buf);
        // an accessor with value counter from each work_group
        auto __count_gacc = __count_buf.template get_access<access_mode::write>(__hdl);
        // an accessor per work-group with value counters from each work-item
        auto __count_lacc = sycl::accessor<_CountT, 1, access_mode::read_write, access_target::local>(
            __block_size * __radix_states, __hdl);

        __hdl.parallel_for<_KernelName>(
#if _PSTL_COMPILE_KERNEL
            __kernel,
#endif
            sycl::nd_range<1>(__segments * __block_size, __block_size), [=](sycl::nd_item<1> __self_item) {
                // item info
                const ::std::size_t __self_lidx = __self_item.get_local_id(0);
                const ::std::size_t __wgroup_idx = __self_item.get_group(0);
                const ::std::size_t __start_idx = __blocks_per_segment * __block_size * __wgroup_idx + __self_lidx;

                // 1.1. count per witem: create a private array for storing count values
                _CountT __count_arr[__radix_states] = {0};
                // 1.2. count per witem: count values and write result to private count array
                for (::std::size_t __block_idx = 0; __block_idx < __blocks_per_segment; ++__block_idx)
                {
                    const ::std::size_t __val_idx = __start_idx + __block_size * __block_idx;
                    // TODO: profile how it affects performance
                    if (__val_idx < __val_buf_size)
                    {
                        // get value, convert it to ordered (in terms of bitness)
                        __ordered_t<_ValueT> __val = __convert_to_ordered(__val_gacc[__val_idx]);
                        // get bit values in a certain bucket of a value
                        ::std::uint32_t __bucket_val =
                            __get_bucket_value<__radix_bits, __is_comp_asc>(__val, __radix_iter);
                        // increment counter for this bit bucket
                        ++__count_arr[__bucket_val];
                    }
                }
                // 1.3. count per witem: write private count array to local count array
                const ::std::uint32_t __count_start_idx = __radix_states * __self_lidx;
                for (::std::uint32_t __radix_state_idx = 0; __radix_state_idx < __radix_states; ++__radix_state_idx)
                    __count_lacc[__count_start_idx + __radix_state_idx] = __count_arr[__radix_state_idx];
                __self_item.barrier(sycl::access::fence_space::local_space);

                // 2.1. count per wgroup: reduce till __count_lacc[] size > __block_size (all threads work)
                for (::std::uint32_t __i = 1; __i < __radix_states; ++__i)
                    __count_lacc[__self_lidx] += __count_lacc[__block_size * __i + __self_lidx];
                __self_item.barrier(sycl::access::fence_space::local_space);
                // 2.2. count per wgroup: reduce until __count_lacc[] size > __radix_states (threads /= 2 per iteration)
                for (::std::uint32_t __active_ths = __block_size >> 1; __active_ths >= __radix_states;
                     __active_ths >>= 1)
                {
                    if (__self_lidx < __active_ths)
                        __count_lacc[__self_lidx] += __count_lacc[__active_ths + __self_lidx];
                    __self_item.barrier(sycl::access::fence_space::local_space);
                }
                // 2.3. count per wgroup: write local count array to global count array
                if (__self_lidx < __radix_states)
                    __count_gacc[__radix_states * __wgroup_idx + __self_lidx] = __count_lacc[__self_lidx];
            });
    });

    return __count_levent;
}

//-----------------------------------------------------------------------
// radix sort: scan kernel (per iteration)
//-----------------------------------------------------------------------

template <typename _KernelName, typename _Iterator, ::std::uint32_t __radix_bits, typename _ExecutionPolicy,
          typename _CountBuf>
sycl::event
__radix_sort_scan_submit(_ExecutionPolicy&& __exec, ::std::size_t __segments, _CountBuf& __count_buf,
                         ::std::size_t __count_buf_size, sycl::event __dependency_event)
{
    // typedefs
    using _CountT = __value_t<_CountBuf>;

    // radix states used for an array storing bucket state counters
    const ::std::uint32_t __radix_states = __get_states_in_bits(__radix_bits);

    sycl::event __scan_event = __exec.queue().submit([&](sycl::handler& __hdl) {
        __hdl.depends_on(__dependency_event);

        // an accessor with value counter from each work_group
        auto __count_gacc = __count_buf.template get_access<access_mode::read_write>(__hdl);

        __hdl.parallel_for<_KernelName>(
            sycl::nd_range<1>(__radix_states, __radix_states), [=](sycl::nd_item<1> __self_item) {
                // item info
                const ::std::size_t __self_lidx = __self_item.get_local_id(0);

                // exclusive scan (but in last __radix_states we write total sum)
                ::std::uint32_t __part_sum = 0u;
                for (::std::uint32_t __i = 0; __i <= __segments; ++__i)
                {
                    ::std::size_t __count_val_idx = __i * __radix_states + __self_lidx;
                    ::std::uint32_t __prev_val = __count_gacc[__count_val_idx];
                    __count_gacc[__count_val_idx] = __part_sum;
                    __part_sum += __prev_val;
                }

                // exclusive scan over total radix sums
                ::std::size_t __total_sum_idx = __segments * __radix_states + __self_lidx;
                __count_gacc[__total_sum_idx] = sycl::ONEAPI::exclusive_scan(
                    __self_item.get_group(), __count_gacc[__total_sum_idx], sycl::ONEAPI::plus<_CountT>());
            });
    });

    return __scan_event;
}

//-----------------------------------------------------------------------
// radix sort: a function for reorder phase of one iteration
//-----------------------------------------------------------------------

template <typename _KernelName, typename _Iterator, ::std::uint32_t __radix_bits, bool __is_comp_asc,
          typename _ExecutionPolicy, typename _InBuf, typename _OutBuf, typename _OffsetBuf
#if _PSTL_COMPILE_KERNEL
          ,
          typename _Kernel
#endif
          >
sycl::event
__radix_sort_reorder_submit(_ExecutionPolicy&& __exec, ::std::size_t __segments, ::std::size_t __block_size,
                            ::std::size_t __sg_size, ::std::uint32_t __radix_iter, _InBuf& __input_buf,
                            _OutBuf& __output_buf, ::std::size_t __inout_buf_size, _OffsetBuf& __offset_buf,
                            ::std::size_t __offset_buf_size, sycl::event __dependency_event
#if _PSTL_COMPILE_KERNEL
                            ,
                            _Kernel& __kernel
#endif
)
{
    // typedefs
    using _InputT = __value_t<_InBuf>;
    using _OffsetT = __value_t<_OffsetBuf>;

    // item info
    const ::std::size_t __it_size = __block_size / __sg_size;

    // iteration space info
    const ::std::uint32_t __radix_states = __get_states_in_bits(__radix_bits);
    const ::std::size_t __blocks_total = __get_roundedup_div(__inout_buf_size, __block_size);
    const ::std::size_t __blocks_per_segment = __get_roundedup_div(__blocks_total, __segments);

    // submit to reorder values
    sycl::event __reorder_event = __exec.queue().submit([&](sycl::handler& __hdl) {
        __hdl.depends_on(__dependency_event);

        // an accessor with offsets from each work group
        auto __offset_gacc = __offset_buf.template get_access<access_mode::read>(__hdl);
        // an accessor with values to reorder
        auto __input_gacc = __internal::get_access<_Iterator>(__hdl)(__input_buf);
        // an accessor for reordered values
        auto __output_gacc = __internal::get_access<_Iterator>(__hdl)(__output_buf);

        __hdl.parallel_for<_KernelName>(
#if _PSTL_COMPILE_KERNEL
            __kernel,
#endif
            sycl::nd_range<1>(__segments * __sg_size, __sg_size), [=](sycl::nd_item<1> __self_item) {
                // item info
                const ::std::size_t __self_lidx = __self_item.get_local_id(0);
                const ::std::size_t __wgroup_idx = __self_item.get_group(0);
                const ::std::size_t __start_idx = __blocks_per_segment * __block_size * __wgroup_idx + __self_lidx;

                // 1. create a private array for storing offset values
                //    and add total offset and offset for compute unit for a certain radix state
                _OffsetT __offset_arr[__radix_states];
                const ::std::uint32_t __global_offset_start_idx = __segments * __radix_states;
                const ::std::uint32_t __local_offset_start_idx = __wgroup_idx * __radix_states;
                for (::std::uint32_t __radix_state_idx = 0; __radix_state_idx < __radix_states; ++__radix_state_idx)
                    __offset_arr[__radix_state_idx] = __offset_gacc[__global_offset_start_idx + __radix_state_idx] +
                                                      __offset_gacc[__local_offset_start_idx + __radix_state_idx];

                for (::std::size_t __block_idx = 0; __block_idx < __blocks_per_segment * __it_size; ++__block_idx)
                {
                    const ::std::size_t __val_idx = __start_idx + __sg_size * __block_idx;
                    // TODO: profile how it affects performance
                    if (__val_idx < __inout_buf_size)
                    {
                        // get value, convert it to ordered (in terms of bitness)
                        __ordered_t<_InputT> __batch_val = __convert_to_ordered(__input_gacc[__val_idx]);
                        // get bit values in a certain bucket of a value
                        ::std::uint32_t __bucket_val =
                            __get_bucket_value<__radix_bits, __is_comp_asc>(__batch_val, __radix_iter);

                        _OffsetT __new_offset_idx = 0;
                        // TODO: most computation-heavy code segment - find a better optimized solution
                        for (::std::uint32_t __radix_state_idx = 0; __radix_state_idx < __radix_states;
                             ++__radix_state_idx)
                        {
                            ::std::uint32_t __is_current_bucket = __bucket_val == __radix_state_idx;
                            ::std::uint32_t __sg_item_offset =
                                sycl::ONEAPI::exclusive_scan(__self_item.get_sub_group(), __is_current_bucket,
                                                             sycl::ONEAPI::plus<::std::uint32_t>());

                            __new_offset_idx |=
                                __is_current_bucket * (__offset_arr[__radix_state_idx] + __sg_item_offset);
                            ::std::uint32_t __sg_total_offset =
                                sycl::ONEAPI::reduce(__self_item.get_sub_group(), __is_current_bucket,
                                                     sycl::ONEAPI::plus<::std::uint32_t>());

                            __offset_arr[__radix_state_idx] = __offset_arr[__radix_state_idx] + __sg_total_offset;
                        }

                        __output_gacc[__new_offset_idx] = __input_gacc[__val_idx];
                    }
                }
            });
    });

    return __reorder_event;
}

//-----------------------------------------------------------------------
// radix sort: a function for one iteration
//-----------------------------------------------------------------------

template <typename _Iterator, ::std::uint32_t __radix_bits, bool __is_comp_asc, typename _ExecutionPolicy,
          typename _InBuf, typename _OutBuf, typename _TmpBuf>
sycl::event
__parallel_radix_sort_iteration(_ExecutionPolicy&& __exec, ::std::size_t __segments, ::std::uint32_t __radix_iter,
                                _InBuf& __in_buf, _OutBuf& __out_buf, ::std::size_t __inout_buf_size,
                                _TmpBuf& __tmp_buf, ::std::size_t __tmp_buf_size, sycl::event __dependency_event)
{
    using __count_kernel_name =
        __radix_sort_count_kernel<_InBuf, _TmpBuf, typename __decay_t<_ExecutionPolicy>::kernel_name>;
    using __scan_kernel_name = __radix_sort_scan_kernel<_TmpBuf, typename __decay_t<_ExecutionPolicy>::kernel_name>;
    using __reorder_kernel_name =
        __radix_sort_reorder_kernel<_InBuf, _OutBuf, ::std::size_t, typename __decay_t<_ExecutionPolicy>::kernel_name>;
    using _KernelName = typename __decay_t<_ExecutionPolicy>::kernel_name;
    using __count_kernel_name = __radix_sort_count_kernel<_InBuf, _TmpBuf, _KernelName>;
    using __scan_kernel_name = __radix_sort_scan_kernel<_TmpBuf, _KernelName>;
    using __reorder_kernel_name = __radix_sort_reorder_kernel<_InBuf, _OutBuf, ::std::size_t, _KernelName>;
    ::std::size_t __max_sg_size = oneapi::dpl::__internal::__max_sub_group_size(__exec);
    ::std::size_t __block_size = __max_sg_size;
    ::std::size_t __reorder_sg_size = __max_sg_size;
#if _PSTL_COMPILE_KERNEL
    auto __count_kernel = __count_kernel_name::__compile_kernel(::std::forward<_ExecutionPolicy>(__exec));
    auto __reorder_kernel = __reorder_kernel_name::__compile_kernel(::std::forward<_ExecutionPolicy>(__exec));
    ::std::size_t __count_sg_size = oneapi::dpl::__internal::__kernel_sub_group_size(__exec, __count_kernel);
    __reorder_sg_size = oneapi::dpl::__internal::__kernel_sub_group_size(__exec, __reorder_kernel);
    __block_size = sycl::max(__count_sg_size, __reorder_sg_size);
#endif
    // TODO: block size mustn't be less than number of states now. Check how to get rid of that restriction.
    const ::std::uint32_t __radix_states = __get_states_in_bits(__radix_bits);
    if (__block_size < __radix_states)
        __block_size = __radix_states;

    // 1. Count Phase
    sycl::event __count_event = __radix_sort_count_submit<__count_kernel_name, _Iterator, __radix_bits, __is_comp_asc>(
        __exec, __segments, __block_size, __radix_iter, __in_buf, __inout_buf_size, __tmp_buf, __tmp_buf_size,
        __dependency_event
#if _PSTL_COMPILE_KERNEL
        ,
        __count_kernel
#endif
    );

    // 2. Scan Phase
    sycl::event __scan_event = __radix_sort_scan_submit<__scan_kernel_name, _Iterator, __radix_bits>(
        __exec, __segments, __tmp_buf, __tmp_buf_size, __count_event);

    // 3. Reorder Phase
    sycl::event __reorder_event =
        __radix_sort_reorder_submit<__reorder_kernel_name, _Iterator, __radix_bits, __is_comp_asc>(
            __exec, __segments, __block_size, __reorder_sg_size, __radix_iter, __in_buf, __out_buf, __inout_buf_size,
            __tmp_buf, __tmp_buf_size, __scan_event
#if _PSTL_COMPILE_KERNEL
            ,
            __reorder_kernel
#endif
        );

    return __reorder_event;
}

//-----------------------------------------------------------------------
// radix sort: main function
//-----------------------------------------------------------------------

template <bool __is_comp_asc, typename _Iterator, typename _ExecutionPolicy>
void
__parallel_radix_sort(_ExecutionPolicy&& __exec, _Iterator __first, _Iterator __last)
{
    const ::std::size_t __inout_buf_size = __last - __first;
    if (__inout_buf_size <= 1)
        return;

    auto __in_buf = __internal::get_buffer()(__first, __last);

    // typedefs
    using _Buffer = __decay_t<decltype(__in_buf)>;
    using _DecExecutionPolicy = __decay_t<_ExecutionPolicy>;
    using _T = __value_t<_Iterator>;

    const ::std::size_t __wg_size = oneapi::dpl::__internal::__max_work_group_size(__exec);
    const ::std::size_t __cunits = oneapi::dpl::__internal::__max_compute_units(__exec);

    ::std::size_t __seg_k = 12;
    if (__inout_buf_size < __cunits * __wg_size * __seg_k)
        __seg_k = 1;
    const ::std::size_t __segments = __get_roundedup_div(__inout_buf_size, __wg_size * __seg_k);

    // radix bits represent number of processed bits in each value during one iteration
    const ::std::uint32_t __radix_bits = 4;
    const ::std::uint32_t __radix_iters = __get_buckets_in_type<_T>(__radix_bits);
    const ::std::uint32_t __radix_states = __get_states_in_bits(__radix_bits);

    // additional __radix_states elements are used for storing total scan sums
    const ::std::size_t __tmp_buf_size = __segments * __radix_states + __radix_states;
    // memory for storing count and offset values
    auto __tmp_buf = sycl::buffer<::std::uint32_t, 1>(sycl::range<1>(__tmp_buf_size));
    // memory for storing values sorted for an iteration
    __internal::__buffer<_DecExecutionPolicy, _T, _Buffer> __out_buffer_holder{__exec, __inout_buf_size};
    auto __out_buf = __out_buffer_holder.get_buffer();

    // iterations per each bucket
    // TODO: radix for bool can be made using 1 iteration (x2 speedup against current implementation)
    sycl::event __iteration_event{};
    for (::std::uint32_t __radix_iter = 0; __radix_iter < __radix_iters; ++__radix_iter)
    {
        // TODO: convert to ordered type once at the first iteration and convert back at the last one
        __iteration_event = __parallel_radix_sort_iteration<_Iterator, __radix_bits, __is_comp_asc>(
            __exec, __segments, __radix_iter, __in_buf, __out_buf, __inout_buf_size, __tmp_buf, __tmp_buf_size,
            __iteration_event);

        ::std::swap(__in_buf, __out_buf);

        // TODO: since reassign to __iteration_event does not work, we have to make explicit wait on the event
        explicit_wait_if<::std::is_pointer<_Iterator>::value>{}(__iteration_event);
    }

    return;
}

} // namespace __par_backend_hetero
} // namespace dpl
} // namespace oneapi

#endif /* _ONEDPL_parallel_backend_sycl_radix_sort_H */
