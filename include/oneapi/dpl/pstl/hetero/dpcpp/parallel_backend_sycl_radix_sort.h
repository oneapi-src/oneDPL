// -*- C++ -*-
//===-- parallel_backend_sycl_radix_sort.h --------------------------------===//
//
// Copyright (C) Intel Corporation
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
class __radix_sort_scan_kernel_1 : public __kernel_name_base<__radix_sort_scan_kernel_1<_Name...>>
{
};

template <typename... _Name>
class __radix_sort_scan_kernel_2 : public __kernel_name_base<__radix_sort_scan_kernel_2<_Name...>>
{
};

template <typename... _Name>
class __radix_sort_reorder_kernel : public __kernel_name_base<__radix_sort_reorder_kernel<_Name...>>
{
};

template <typename _Name>
class __odd_iteration;

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
    using _type = uint32_t;
    constexpr static ::std::uint32_t __nmask = 0xFFFFFFFF; // for negative numbers
    constexpr static ::std::uint32_t __pmask = 0x80000000; // for positive numbers
};

template <>
struct __get_ordered<8, false>
{
    using _type = uint64_t;
    constexpr static ::std::uint64_t __nmask = 0xFFFFFFFFFFFFFFFF; // for negative numbers
    constexpr static ::std::uint64_t __pmask = 0x8000000000000000; // for positive numbers
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

// get rounded up result of (__number / __divisor)
template <typename _T1, typename _T2>
inline auto
__get_roundedup_div(_T1 __number, _T2 __divisor) -> decltype((__number - 1) / __divisor + 1)
{
    return (__number - 1) / __divisor + 1;
}

template <typename _T>
inline _T
__get_rounded_down_power2(_T __x)
{
    _T __val = 1;
    while (__x >= 2 * __val)
        __val <<= 1;
    return __val;
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

template <typename _T, bool __is_comp_asc>
inline __enable_if_t<__is_comp_asc, _T>
__get_last_value()
{
    return ::std::numeric_limits<_T>::max();
};

template <typename _T, bool __is_comp_asc>
inline __enable_if_t<!__is_comp_asc, _T>
__get_last_value()
{
    return ::std::numeric_limits<_T>::min();
};

//-----------------------------------------------------------------------
// radix sort: count kernel (per iteration)
//-----------------------------------------------------------------------

template <typename _KernelName, ::std::uint32_t __radix_bits, bool __is_comp_asc, typename _ExecutionPolicy,
          typename _ValRange, typename _CountBuf
#if _ONEDPL_COMPILE_KERNEL
          ,
          typename _Kernel
#endif
          >
sycl::event
__radix_sort_count_submit(_ExecutionPolicy&& __exec, ::std::size_t __segments, ::std::size_t __block_size,
                          ::std::uint32_t __radix_iter, _ValRange&& __val_rng, _CountBuf& __count_buf,
                          sycl::event __dependency_event
#if _ONEDPL_COMPILE_KERNEL
                          ,
                          _Kernel __kernel
#endif
)
{
    // typedefs
    using _ValueT = oneapi::dpl::__internal::__value_t<_ValRange>;
    using _CountT = typename _CountBuf::value_type;

    // radix states used for an array storing bucket state counters
    const ::std::uint32_t __radix_states = __get_states_in_bits(__radix_bits);

    const auto __val_buf_size = __val_rng.size();
    // iteration space info
    const ::std::size_t __blocks_total = __get_roundedup_div(__val_buf_size, __block_size);
    const ::std::size_t __blocks_per_segment = __get_roundedup_div(__blocks_total, __segments);

    auto __count_rng =
        oneapi::dpl::__ranges::all_view<_CountT, __par_backend_hetero::access_mode::read_write>(__count_buf);

    // submit to compute arrays with local count values
    sycl::event __count_levent = __exec.queue().submit([&](sycl::handler& __hdl) {
        __hdl.depends_on(__dependency_event);

        oneapi::dpl::__ranges::__require_access(__hdl, __val_rng,
                                                __count_rng); //get an access to data under SYCL buffer
        // an accessor per work-group with value counters from each work-item
        auto __count_lacc = sycl::accessor<_CountT, 1, access_mode::read_write, access_target::local>(
            __block_size * __radix_states, __hdl);

#if _ONEDPL_KERNEL_BUNDLE_PRESENT
        __hdl.use_kernel_bundle(__kernel);
#endif
        __hdl.parallel_for<_KernelName>(
#if _ONEDPL_COMPILE_KERNEL && !_ONEDPL_KERNEL_BUNDLE_PRESENT
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
                        __ordered_t<_ValueT> __val = __convert_to_ordered(__val_rng[__val_idx]);
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
                {
                    // move buckets with the same id to adjacent positions,
                    // thus splitting __count_rng into __radix_states regions
                    __count_rng[(__segments + 1) * __self_lidx + __wgroup_idx] = __count_lacc[__self_lidx];
                }
            });
    });

    return __count_levent;
}

//-----------------------------------------------------------------------
// radix sort: scan kernel (per iteration)
//-----------------------------------------------------------------------

template <typename _KernelName1, typename _KernelName2, ::std::uint32_t __radix_bits, typename _ExecutionPolicy,
          typename _CountBuf>
sycl::event
__radix_sort_scan_submit(_ExecutionPolicy&& __exec, ::std::size_t __scan_wg_size, ::std::size_t __segments,
                         _CountBuf& __count_buf, sycl::event __dependency_event)
{
    using _CountT = typename _CountBuf::value_type;

    auto __count_rng =
        oneapi::dpl::__ranges::all_view<_CountT, __par_backend_hetero::access_mode::read_write>(__count_buf);

    // there are no local offsets for the first segment, but the rest segments should be scanned
    // with respect to the count value in the first segment what requires n + 1 positions
    const ::std::size_t __scan_size = __segments + 1;

    __scan_wg_size = ::std::min(__scan_size, __scan_wg_size);

    const ::std::uint32_t __radix_states = __get_states_in_bits(__radix_bits);
    const ::std::size_t __global_scan_begin = __count_buf.get_count() - __radix_states;

    // 1. Local scan: produces local offsets using count values
    sycl::event __scan_event = __exec.queue().submit([&](sycl::handler& __hdl) {
        __hdl.depends_on(__dependency_event);
        // an accessor with value counter from each work_group
        oneapi::dpl::__ranges::__require_access(__hdl, __count_rng); //get an access to data under SYCL buffer
        __hdl.parallel_for<_KernelName1>(
            sycl::nd_range<1>(__radix_states * __scan_wg_size, __scan_wg_size), [=](sycl::nd_item<1> __self_item) {
                // find borders of a region with a specific bucket id
                sycl::global_ptr<_CountT> __begin = __count_rng.begin() + __scan_size * __self_item.get_group(0);
                // TODO: consider another approach with use of local memory
                sycl::ONEAPI::exclusive_scan(__self_item.get_group(), __begin, __begin + __scan_size, __begin,
                                             _CountT(0), sycl::ONEAPI::plus<_CountT>{});
            });
    });

    // 2. Global scan: produces global offsets using local offsets
    __scan_event = __exec.queue().submit([&](sycl::handler& __hdl) {
        __hdl.depends_on(__scan_event);
        oneapi::dpl::__ranges::__require_access(__hdl, __count_rng);
        __hdl.parallel_for<_KernelName2>(
            sycl::nd_range<1>(__radix_states, __radix_states), [=](sycl::nd_item<1> __self_item) {
                ::std::size_t __self_lidx = __self_item.get_local_id(0);

                ::std::size_t __global_offset_idx = __global_scan_begin + __self_lidx;
                ::std::size_t __last_segment_bucket_idx = (__self_lidx + 1) * __scan_size - 1;

                // copy buckets from the last segment, scan them to get global offsets
                __count_rng[__global_offset_idx] = sycl::ONEAPI::exclusive_scan(
                    __self_item.get_group(), __count_rng[__last_segment_bucket_idx], sycl::ONEAPI::plus<_CountT>{});
            });
    });

    return __scan_event;
}

//-----------------------------------------------------------------------
// radix sort: a function for reorder phase of one iteration
//-----------------------------------------------------------------------

template <typename _KernelName, ::std::uint32_t __radix_bits, bool __is_comp_asc, typename _ExecutionPolicy,
          typename _InRange, typename _OutRange, typename _OffsetBuf
#if _ONEDPL_COMPILE_KERNEL
          ,
          typename _Kernel
#endif
          >
sycl::event
__radix_sort_reorder_submit(_ExecutionPolicy&& __exec, ::std::size_t __segments, ::std::size_t __block_size,
                            ::std::size_t __sg_size, ::std::uint32_t __radix_iter, _InRange&& __input_rng,
                            _OutRange&& __output_rng, _OffsetBuf& __offset_buf, sycl::event __dependency_event
#if _ONEDPL_COMPILE_KERNEL
                            ,
                            _Kernel __kernel
#endif
)
{
    // typedefs
    using _InputT = oneapi::dpl::__internal::__value_t<_InRange>;
    using _OffsetT = typename _OffsetBuf::value_type;

    // item info
    const ::std::size_t __it_size = __block_size / __sg_size;
    const auto __inout_buf_size = __output_rng.size();

    // iteration space info
    const ::std::uint32_t __radix_states = __get_states_in_bits(__radix_bits);
    const ::std::size_t __blocks_total = __get_roundedup_div(__inout_buf_size, __block_size);
    const ::std::size_t __blocks_per_segment = __get_roundedup_div(__blocks_total, __segments);

    auto __offset_rng =
        oneapi::dpl::__ranges::all_view<::std::uint32_t, __par_backend_hetero::access_mode::read>(__offset_buf);

    // submit to reorder values
    sycl::event __reorder_event = __exec.queue().submit([&](sycl::handler& __hdl) {
        __hdl.depends_on(__dependency_event);

        // access with offsets from each work group
        oneapi::dpl::__ranges::__require_access(__hdl, __offset_rng);

        // access with values to reorder and reordered values
        oneapi::dpl::__ranges::__require_access(__hdl, __input_rng, __output_rng);

#if _ONEDPL_KERNEL_BUNDLE_PRESENT
        __hdl.use_kernel_bundle(__kernel);
#endif
        __hdl.parallel_for<_KernelName>(
#if _ONEDPL_COMPILE_KERNEL && !_ONEDPL_KERNEL_BUNDLE_PRESENT
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
                const ::std::uint32_t __global_offset_start_idx = (__segments + 1) * __radix_states;
                for (::std::uint32_t __radix_state_idx = 0; __radix_state_idx < __radix_states; ++__radix_state_idx)
                {
                    const ::std::uint32_t __global_offset_idx = __global_offset_start_idx + __radix_state_idx;
                    const ::std::uint32_t __local_offset_idx = __wgroup_idx + (__segments + 1) * __radix_state_idx;
                    __offset_arr[__radix_state_idx] =
                        __offset_rng[__global_offset_idx] + __offset_rng[__local_offset_idx];
                }

                // find offsets for the same values within a segment and fill the resulting buffer
                for (::std::size_t __block_idx = 0; __block_idx < __blocks_per_segment * __it_size; ++__block_idx)
                {
                    const ::std::size_t __val_idx = __start_idx + __sg_size * __block_idx;

                    // get value, convert it to ordered (in terms of bitness)
                    // if the index is outside of the range, use fake value which will not affect other values
                    __ordered_t<_InputT> __batch_val = __val_idx < __inout_buf_size
                                                           ? __convert_to_ordered(__input_rng[__val_idx])
                                                           : __get_last_value<__ordered_t<_InputT>, __is_comp_asc>();

                    // get bit values in a certain bucket of a value
                    ::std::uint32_t __bucket_val =
                        __get_bucket_value<__radix_bits, __is_comp_asc>(__batch_val, __radix_iter);

                    _OffsetT __new_offset_idx = 0;
                    // TODO: most computation-heavy code segment - find a better optimized solution
                    for (::std::uint32_t __radix_state_idx = 0; __radix_state_idx < __radix_states; ++__radix_state_idx)
                    {
                        ::std::uint32_t __is_current_bucket = __bucket_val == __radix_state_idx;
                        ::std::uint32_t __sg_item_offset = sycl::ONEAPI::exclusive_scan(
                            __self_item.get_sub_group(), __is_current_bucket, sycl::ONEAPI::plus<::std::uint32_t>());

                        __new_offset_idx |= __is_current_bucket * (__offset_arr[__radix_state_idx] + __sg_item_offset);
                        // the last scanned value may not contain number of all copies, thus adding __is_current_bucket
                        ::std::uint32_t __sg_total_offset = sycl::ONEAPI::broadcast(
                            __self_item.get_sub_group(), __sg_item_offset + __is_current_bucket, __sg_size - 1);

                        __offset_arr[__radix_state_idx] = __offset_arr[__radix_state_idx] + __sg_total_offset;
                    }

                    if (__val_idx < __inout_buf_size)
                        __output_rng[__new_offset_idx] = __input_rng[__val_idx];
                }
            });
    });

    return __reorder_event;
}

//-----------------------------------------------------------------------
// radix sort: a function for one iteration
//-----------------------------------------------------------------------

template <::std::uint32_t __radix_bits, bool __is_comp_asc, typename _ExecutionPolicy, typename _InRange,
          typename _OutRange, typename _TmpBuf>
sycl::event
__parallel_radix_sort_iteration(_ExecutionPolicy&& __exec, ::std::size_t __segments, ::std::uint32_t __radix_iter,
                                _InRange&& __in_rng, _OutRange&& __out_rng, _TmpBuf& __tmp_buf,
                                sycl::event __dependency_event)
{
    using _CustomName = typename __decay_t<_ExecutionPolicy>::kernel_name;
    using _RadixCountKernel =
        oneapi::dpl::__par_backend_hetero::__internal::_KernelName_t<__radix_sort_count_kernel, _CustomName,
                                                                     __decay_t<_InRange>, __decay_t<_TmpBuf>>;
    using _RadixLocalScanKernel =
        oneapi::dpl::__par_backend_hetero::__internal::_KernelName_t<__radix_sort_scan_kernel_1, _CustomName,
                                                                     __decay_t<_TmpBuf>>;
    using _RadixGlobalScanKernel =
        oneapi::dpl::__par_backend_hetero::__internal::_KernelName_t<__radix_sort_scan_kernel_2, _CustomName,
                                                                     __decay_t<_TmpBuf>>;
    using _RadixReorderKernel =
        oneapi::dpl::__par_backend_hetero::__internal::_KernelName_t<__radix_sort_reorder_kernel, _CustomName,
                                                                     __decay_t<_InRange>, __decay_t<_OutRange>>;

    ::std::size_t __max_sg_size = oneapi::dpl::__internal::__max_sub_group_size(__exec);
    ::std::size_t __scan_wg_size = oneapi::dpl::__internal::__max_work_group_size(__exec);
    ::std::size_t __block_size = __max_sg_size;
    ::std::size_t __reorder_sg_size = __max_sg_size;
#if _ONEDPL_COMPILE_KERNEL
    auto __count_kernel_stuff = _RadixCountKernel::create(__exec.queue().get_context());
    auto __count_kernel = __count_kernel_stuff.__compile_kernel();
    auto __reorder_kernel_stuff = _RadixReorderKernel::create(__exec.queue().get_context());
    auto __reorder_kernel = __reorder_kernel_stuff.__compile_kernel();
    ::std::size_t __count_sg_size = oneapi::dpl::__internal::__kernel_sub_group_size(__exec, __count_kernel);
    __reorder_sg_size = oneapi::dpl::__internal::__kernel_sub_group_size(__exec, __reorder_kernel);
    __block_size = sycl::max(__count_sg_size, __reorder_sg_size);
#endif
    const ::std::uint32_t __radix_states = __get_states_in_bits(__radix_bits);

    // correct __block_size according to local memory limit in count phase
    const auto __max_allocation_size =
        oneapi::dpl::__internal::__max_local_allocation_size<_ExecutionPolicy, typename __decay_t<_TmpBuf>::value_type>(
            ::std::forward<_ExecutionPolicy>(__exec), __block_size * __radix_states);
    __block_size = __get_roundedup_div(__max_allocation_size, __radix_states);

    // TODO: block size must be power of 2 and more than number of states. Check how to get rid of that restriction.
    __block_size = __get_rounded_down_power2(__block_size);
    if (__block_size < __radix_states)
        __block_size = __radix_states;

    // 1. Count Phase
    sycl::event __count_event = __radix_sort_count_submit<_RadixCountKernel, __radix_bits, __is_comp_asc>(
        ::std::forward<_ExecutionPolicy>(__exec), __segments, __block_size, __radix_iter,
        ::std::forward<_InRange>(__in_rng), __tmp_buf, __dependency_event
#if _ONEDPL_COMPILE_KERNEL
        ,
#    if _ONEDPL_KERNEL_BUNDLE_PRESENT
        __count_kernel_stuff.kernel_bundle()
#    else
        __count_kernel
#    endif
#endif //_ONEDPL_COMPILE_KERNEL
    );

    // 2. Scan Phase
    sycl::event __scan_event = __radix_sort_scan_submit<_RadixLocalScanKernel, _RadixGlobalScanKernel, __radix_bits>(
        ::std::forward<_ExecutionPolicy>(__exec), __scan_wg_size, __segments, __tmp_buf, __count_event);

    // 3. Reorder Phase
    sycl::event __reorder_event = __radix_sort_reorder_submit<_RadixReorderKernel, __radix_bits, __is_comp_asc>(
        ::std::forward<_ExecutionPolicy>(__exec), __segments, __block_size, __reorder_sg_size, __radix_iter,
        ::std::forward<_InRange>(__in_rng), ::std::forward<_OutRange>(__out_rng), __tmp_buf, __scan_event
#if _ONEDPL_COMPILE_KERNEL
        ,
#    if _ONEDPL_KERNEL_BUNDLE_PRESENT
        __reorder_kernel_stuff.kernel_bundle()
#    else
        __reorder_kernel
#    endif
#endif //_ONEDPL_COMPILE_KERNEL
    );

    return __reorder_event;
}

//-----------------------------------------------------------------------
// radix sort: main function
//-----------------------------------------------------------------------

template <bool __is_comp_asc, typename _Range, typename _ExecutionPolicy>
__future<void>
__parallel_radix_sort(_ExecutionPolicy&& __exec, _Range&& __in_rng)
{
    const ::std::size_t __n = __in_rng.size();
    assert(__n > 1);

    // types
    using _DecExecutionPolicy = __decay_t<_ExecutionPolicy>;
    using _T = oneapi::dpl::__internal::__value_t<_Range>;

    const ::std::size_t __wg_size = oneapi::dpl::__internal::__max_work_group_size(__exec);
    const ::std::size_t __segments = __get_roundedup_div(__n, __wg_size);

    // radix bits represent number of processed bits in each value during one iteration
    const ::std::uint32_t __radix_bits = 4;
    const ::std::uint32_t __radix_iters = __get_buckets_in_type<_T>(__radix_bits);
    const ::std::uint32_t __radix_states = __get_states_in_bits(__radix_bits);

    // additional 2 * __radix_states elements are used for getting local and global offsets from count values
    const ::std::size_t __tmp_buf_size = __segments * __radix_states + 2 * __radix_states;
    // memory for storing count and offset values
    auto __tmp_buf = sycl::buffer<::std::uint32_t, 1>(sycl::range<1>(__tmp_buf_size));

    // memory for storing values sorted for an iteration
    __internal::__buffer<_DecExecutionPolicy, _T> __out_buffer_holder{__exec, __n};
    auto __out_rng = oneapi::dpl::__ranges::all_view<_T, __par_backend_hetero::access_mode::read_write>(
        __out_buffer_holder.get_buffer());

    // iterations per each bucket
    assert("Number of iterations must be even" && __radix_iters % 2 == 0);
    // TODO: radix for bool can be made using 1 iteration (x2 speedup against current implementation)
    sycl::event __iteration_event{};
    for (::std::uint32_t __radix_iter = 0; __radix_iter < __radix_iters; ++__radix_iter)
    {
        // TODO: convert to ordered type once at the first iteration and convert back at the last one
        if (__radix_iter % 2 == 0)
            __iteration_event = __parallel_radix_sort_iteration<__radix_bits, __is_comp_asc>(
                ::std::forward<_ExecutionPolicy>(__exec), __segments, __radix_iter, ::std::forward<_Range>(__in_rng),
                __out_rng, __tmp_buf, __iteration_event);
        else //swap __in_rng and __out_rng
            __iteration_event = __parallel_radix_sort_iteration<__radix_bits, __is_comp_asc>(
                make_wrapped_policy<__odd_iteration>(::std::forward<_ExecutionPolicy>(__exec)), __segments,
                __radix_iter, __out_rng, ::std::forward<_Range>(__in_rng), __tmp_buf, __iteration_event);

        // TODO: since reassign to __iteration_event does not work, we have to make explicit wait on the event
        explicit_wait_if<::std::is_pointer<decltype(__in_rng.begin())>::value>{}(__iteration_event);
    }

    return __future<void>(__iteration_event, __tmp_buf, __out_buffer_holder.get_buffer());
}

} // namespace __par_backend_hetero
} // namespace dpl
} // namespace oneapi

#endif /* _ONEDPL_parallel_backend_sycl_radix_sort_H */
