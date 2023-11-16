// -*- C++ -*-
//===-- parallel_backend_sycl_reduce.h --------------------------------===//
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

#ifndef _ONEDPL_PARALLEL_BACKEND_SYCL_HISTOGRAM_H
#define _ONEDPL_PARALLEL_BACKEND_SYCL_HISTOGRAM_H

#include <algorithm>
#include <cstdint>
#include <type_traits>

#include "sycl_defs.h"
#include "parallel_backend_sycl_utils.h"
#include "execution_sycl_defs.h"
#include "unseq_backend_sycl.h"
#include "utils_ranges_sycl.h"
#include "oneapi/dpl/internal/histogram_binhash_utils.h"
#include "oneapi/dpl/internal/async_impl/async_impl_hetero.h"

namespace oneapi
{
namespace dpl
{
namespace __par_backend_hetero
{

//no boost baseline
template <typename _BinHash>
struct __binhash_SLM_wrapper
{
    _BinHash __bin_hash;
    __binhash_SLM_wrapper(_BinHash __bin_hash) : __bin_hash(__bin_hash) {}

    template <typename _T2>
    inline ::std::uint32_t
    get_bin(_T2&& __value) const
    {
        return __bin_hash.get_bin(::std::forward<_T2>(__value));
    }

    template <typename _T2>
    inline bool
    is_valid(_T2&& __value) const
    {
        return __bin_hash.is_valid(::std::forward<_T2>(__value));
    }

    inline ::std::size_t
    get_required_SLM_memory() const
    {
        return 0;
    }

    inline void
    init_SLM_memory(void* __boost_mem, const sycl::nd_item<1>& __self_item) const
    {
    }

    template <typename _T2>
    inline ::std::uint32_t
    get_bin(_T2&& __value, void* __boost_mem) const
    {
        return get_bin(::std::forward<_T2>(__value));
    }

    template <typename _T2>
    inline bool
    is_valid(_T2&& __value, void* __boost_mem) const
    {
        return is_valid(::std::forward<_T2>(__value));
    }
};

template <typename _Range>
struct __binhash_SLM_wrapper<oneapi::dpl::__internal::__custom_range_binhash<_Range>>
{
    using _BinHashType = typename oneapi::dpl::__internal::__custom_range_binhash<_Range>;
    _BinHashType __bin_hash;

    using __boundary_type = typename _BinHashType::__boundary_type;

    __binhash_SLM_wrapper(_BinHashType __bin_hash) : __bin_hash(__bin_hash) {}

    template <typename _T2>
    inline ::std::uint32_t
    get_bin(_T2&& __value) const
    {
        return __bin_hash.get_bin(::std::forward<_T2>(__value));
    }

    template <typename _T2>
    inline bool
    is_valid(_T2&& __value) const
    {
        return __bin_hash.is_valid(::std::forward<_T2>(__value));
    }

    inline ::std::size_t
    get_required_SLM_memory()
    {
        return sizeof(__boundary_type) * __bin_hash.__boundaries.size();
    }

    inline void
    init_SLM_memory(void* __boost_mem, const sycl::nd_item<1>& __self_item) const
    {
        ::std::uint32_t __gSize = __self_item.get_local_range()[0];
        ::std::uint32_t __self_lidx = __self_item.get_local_id(0);
        ::std::uint8_t __factor = oneapi::dpl::__internal::__dpl_ceiling_div(__bin_hash.__boundaries.size(), __gSize);
        ::std::uint8_t __k = 0;
        __boundary_type* __d_boundaries = static_cast<__boundary_type*>(__boost_mem);
        for (__k; __k < __factor - 1; __k++)
        {
            __d_boundaries[__gSize * __k + __self_lidx] = __bin_hash.__boundaries[__gSize * __k + __self_lidx];
        }
        // residual
        if (__gSize * __k + __self_lidx < __bin_hash.__boundaries.size())
        {
            __d_boundaries[__gSize * __k + __self_lidx] = __bin_hash.__boundaries[__gSize * __k + __self_lidx];
        }
    }

    template <typename _T2>
    ::std::uint32_t inline get_bin(_T2&& __value, void* __boost_mem) const
    {
        __boundary_type* __d_boundaries = static_cast<__boundary_type*>(__boost_mem);
        return (::std::upper_bound(__d_boundaries, __d_boundaries + __bin_hash.__boundaries.size(),
                                   ::std::forward<_T2>(__value)) -
                __d_boundaries) -
               1;
    }

    template <typename _T2>
    bool inline is_valid(const _T2& __value, void* __boost_mem) const
    {
        __boundary_type* __d_boundaries = static_cast<__boundary_type*>(__boost_mem);
        return (__value >= __d_boundaries[0]) && (__value < __d_boundaries[__bin_hash.__boundaries.size()]);
    }
};

template <typename _HistAccessor, typename _OffsetT, typename _Size>
inline void
__clear_wglocal_histograms(const _HistAccessor& __local_histogram, const _OffsetT& __offset, const _Size& __num_bins,
                           const sycl::nd_item<1>& __self_item)
{
    ::std::uint32_t __gSize = __self_item.get_local_range()[0];
    ::std::uint32_t __self_lidx = __self_item.get_local_id(0);
    ::std::uint8_t __factor = oneapi::dpl::__internal::__dpl_ceiling_div(__num_bins, __gSize);
    ::std::uint8_t __k = 0;

    for (__k; __k < __factor - 1; __k++)
    {
        __local_histogram[__offset + __gSize * __k + __self_lidx] = 0;
    }
    // residual
    if (__gSize * __k + __self_lidx < __num_bins)
    {
        __local_histogram[__offset + __gSize * __k + __self_lidx] = 0;
    }
    __dpl_sycl::__group_barrier(__self_item);
}

template <typename _BinIdxType, typename _Iter1, typename _HistReg, typename _BinFunc>
inline void
__accum_local_register_iter(const _Iter1& __in_acc, const ::std::size_t& __index, _HistReg* __histogram,
                            _BinFunc __func, void* __boost_mem)
{
    const auto& __x = __in_acc[__index];
    if (__func.is_valid(__x))
    {
        _BinIdxType c = __func.get_bin(__x, __boost_mem);
        __histogram[c]++;
    }
}

template <typename _BinIdxType, sycl::access::address_space _AddressSpace, typename _Iter1, typename _HistAccessor,
          typename _OffsetT, typename _BinFunc, typename... _VoidType>
inline void
__accum_local_atomics_iter(const _Iter1& __in_acc, const ::std::size_t& __index,
                           const _HistAccessor& __wg_local_histogram, const _OffsetT& __offset, _BinFunc __func,
                           _VoidType... __boost_mem)
{
    using _histo_value_type = typename _HistAccessor::value_type;
    const auto& __x = __in_acc[__index];
    if (__func.is_valid(__x))
    {
        _BinIdxType __c = __func.get_bin(__x, __boost_mem...);
        __dpl_sycl::__atomic_ref<_histo_value_type, _AddressSpace> __local_bin(__wg_local_histogram[__offset + __c]);
        __local_bin++;
    }
}

template <typename _BinType, typename _HistAccessorIn, typename _OffsetT, typename _HistAccessorOut, typename _Size>
inline void
__reduce_out_histograms(const _HistAccessorIn& __in_histogram, const _OffsetT& __offset,
                        const _HistAccessorOut& __out_histogram, const _Size& __num_bins,
                        const sycl::nd_item<1>& __self_item)
{
    ::std::uint32_t __gSize = __self_item.get_local_range()[0];
    ::std::uint32_t __self_lidx = __self_item.get_local_id(0);
    ::std::uint8_t __factor = oneapi::dpl::__internal::__dpl_ceiling_div(__num_bins, __gSize);
    ::std::uint8_t __k = 0;

    for (__k; __k < __factor - 1; __k++)
    {
        __dpl_sycl::__atomic_ref<_BinType, sycl::access::address_space::global_space> __global_bin(
            __out_histogram[__gSize * __k + __self_lidx]);
        __global_bin += __in_histogram[__offset + __gSize * __k + __self_lidx];
    }
    // residual
    if (__gSize * __k + __self_lidx < __num_bins)
    {
        __dpl_sycl::__atomic_ref<_BinType, sycl::access::address_space::global_space> __global_bin(
            __out_histogram[__gSize * __k + __self_lidx]);
        __global_bin += __in_histogram[__offset + __gSize * __k + __self_lidx];
    }
}

template <::std::uint16_t __iters_per_work_item, ::std::uint8_t __bins_per_work_item, typename _ExecutionPolicy,
          typename _Range1, typename _Range2, typename _Size, typename _IdxHashFunc, typename... _Range3>
inline auto
__histogram_general_registers_local_reduction(_ExecutionPolicy&& __exec, const sycl::event& __init_e,
                                              ::std::uint16_t __work_group_size, _Range1&& __input, _Range2&& __bins,
                                              const _Size& __num_bins, _IdxHashFunc __func, _Range3&&... __opt_range)
{
    const ::std::size_t __n = __input.size();
    using _local_histogram_type = ::std::uint32_t;
    using _private_histogram_type = ::std::uint16_t;
    using _histogram_index_type = ::std::uint8_t;
    using _bin_type = oneapi::dpl::__internal::__value_t<_Range2>;

    ::std::size_t __required_slm_bytes = __func.get_required_SLM_memory();
    ::std::size_t __extra = 0;
    if (__required_slm_bytes != 0)
    {
        __extra = oneapi::dpl::__internal::__dpl_ceiling_div(__required_slm_bytes, sizeof(_local_histogram_type));
    }
    ::std::size_t __segments =
        oneapi::dpl::__internal::__dpl_ceiling_div(__n, __work_group_size * __iters_per_work_item);
    return __exec.queue().submit([&](auto& __h) {
        __h.depends_on(__init_e);
        oneapi::dpl::__ranges::__require_access(__h, __input, __bins, __opt_range...);
        __dpl_sycl::__local_accessor<_local_histogram_type> __local_histogram(sycl::range(__num_bins + __extra), __h);
        __h.parallel_for(
            sycl::nd_range<1>(__segments * __work_group_size, __work_group_size), [=](sycl::nd_item<1> __self_item) {
                const ::std::size_t __self_lidx = __self_item.get_local_id(0);
                const ::std::size_t __wgroup_idx = __self_item.get_group(0);
                const ::std::size_t __seg_start = __work_group_size * __iters_per_work_item * __wgroup_idx;
                void* __boost_mem = static_cast<void*>(&(__local_histogram[0]) + __num_bins);
                __func.init_SLM_memory(__boost_mem, __self_item);
                __clear_wglocal_histograms(__local_histogram, 0, __num_bins, __self_item);
                _private_histogram_type __histogram[__bins_per_work_item] = {0};

                if (__seg_start + __work_group_size * __iters_per_work_item < __n)
                {
                    _ONEDPL_PRAGMA_UNROLL
                    for (_histogram_index_type __idx = 0; __idx < __iters_per_work_item; __idx++)
                    {
                        __accum_local_register_iter<_histogram_index_type>(
                            __input, __seg_start + __idx * __work_group_size + __self_lidx, __histogram, __func,
                            __boost_mem);
                    }
                }
                else
                {
                    _ONEDPL_PRAGMA_UNROLL
                    for (_histogram_index_type __idx = 0; __idx < __iters_per_work_item; __idx++)
                    {
                        ::std::size_t __val_idx = __seg_start + __idx * __work_group_size + __self_lidx;
                        if (__val_idx < __n)
                        {
                            __accum_local_register_iter<_histogram_index_type>(__input, __val_idx, __histogram, __func,
                                                                               __boost_mem);
                        }
                    }
                }

                for (_histogram_index_type __k = 0; __k < __num_bins; __k++)
                {
                    __dpl_sycl::__atomic_ref<_local_histogram_type, sycl::access::address_space::local_space>
                        __local_bin(__local_histogram[__k]);
                    __local_bin += __histogram[__k];
                }

                __dpl_sycl::__group_barrier(__self_item);

                __reduce_out_histograms<_bin_type>(__local_histogram, 0, __bins, __num_bins, __self_item);
            });
    });
}

template <::std::uint16_t __iters_per_work_item, typename _ExecutionPolicy, typename _Range1, typename _Range2,
          typename _Size, typename _IdxHashFunc, typename... _Range3>
inline auto
__histogram_general_local_atomics(_ExecutionPolicy&& __exec, const sycl::event& __init_e,
                                  ::std::uint16_t __work_group_size, _Range1&& __input, _Range2&& __bins,
                                  const _Size& __num_bins, _IdxHashFunc __func, _Range3&&... __opt_range)
{
    using _local_histogram_type = ::std::uint32_t;
    using _bin_type = oneapi::dpl::__internal::__value_t<_Range2>;
    using _histogram_index_type = ::std::uint16_t;

    ::std::size_t __extra =
        oneapi::dpl::__internal::__dpl_ceiling_div(__func.get_required_SLM_memory(), sizeof(_local_histogram_type));
    const ::std::size_t __n = __input.size();
    ::std::size_t __segments =
        oneapi::dpl::__internal::__dpl_ceiling_div(__n, __work_group_size * __iters_per_work_item);
    return __exec.queue().submit([&](auto& __h) {
        __h.depends_on(__init_e);
        oneapi::dpl::__ranges::__require_access(__h, __input, __bins, __opt_range...);
        // minimum type size for atomics
        __dpl_sycl::__local_accessor<_local_histogram_type> __local_histogram(sycl::range(__num_bins + __extra), __h);
        __h.parallel_for(sycl::nd_range<1>(__segments * __work_group_size, __work_group_size),
                         [=](sycl::nd_item<1> __self_item) {
                             constexpr auto _atomic_address_space = sycl::access::address_space::local_space;
                             const ::std::size_t __self_lidx = __self_item.get_local_id(0);
                             const ::std::uint32_t __wgroup_idx = __self_item.get_group(0);
                             const ::std::size_t __seg_start = __work_group_size * __wgroup_idx * __iters_per_work_item;
                             void* __boost_mem = static_cast<void*>(&(__local_histogram[0]) + __num_bins);
                             __func.init_SLM_memory(__boost_mem, __self_item);

                             __clear_wglocal_histograms(__local_histogram, 0, __num_bins, __self_item);

                             if (__seg_start + __work_group_size * __iters_per_work_item < __n)
                             {
                                 _ONEDPL_PRAGMA_UNROLL
                                 for (::std::uint8_t __idx = 0; __idx < __iters_per_work_item; __idx++)
                                 {
                                     ::std::size_t __val_idx = __seg_start + __idx * __work_group_size + __self_lidx;
                                     __accum_local_atomics_iter<_histogram_index_type, _atomic_address_space>(
                                         __input, __val_idx, __local_histogram, 0, __func, __boost_mem);
                                 }
                             }
                             else
                             {
                                 _ONEDPL_PRAGMA_UNROLL
                                 for (::std::uint8_t __idx = 0; __idx < __iters_per_work_item; __idx++)
                                 {
                                     ::std::size_t __val_idx = __seg_start + __idx * __work_group_size + __self_lidx;
                                     if (__val_idx < __n)
                                     {
                                         __accum_local_atomics_iter<_histogram_index_type, _atomic_address_space>(
                                             __input, __val_idx, __local_histogram, 0, __func, __boost_mem);
                                     }
                                 }
                             }
                             __dpl_sycl::__group_barrier(__self_item);

                             __reduce_out_histograms<_bin_type>(__local_histogram, 0, __bins, __num_bins, __self_item);
                         });
    });
}

template <::std::uint16_t __min_iters_per_work_item, typename _ExecutionPolicy, typename _Range1, typename _Range2,
          typename _Size, typename _IdxHashFunc, typename... _Range3>
inline auto
__histogram_general_private_global_atomics(_ExecutionPolicy&& __exec, const sycl::event& __init_e,
                                           ::std::uint16_t __work_group_size, _Range1&& __input, _Range2&& __bins,
                                           const _Size& __num_bins, _IdxHashFunc __func, _Range3&&... __opt_range)
{
    const ::std::size_t __n = __input.size();
    using _bin_type = oneapi::dpl::__internal::__value_t<_Range2>;
    using _histogram_index_type = ::std::uint32_t;

    auto __global_mem_size = __exec.queue().get_device().template get_info<sycl::info::device::global_mem_size>();
    const ::std::size_t __max_segments =
        ::std::min(__global_mem_size / (__num_bins * sizeof(_bin_type)),
                   oneapi::dpl::__internal::__dpl_ceiling_div(__n, __work_group_size * __min_iters_per_work_item));
    const ::std::size_t __iters_per_work_item =
        oneapi::dpl::__internal::__dpl_ceiling_div(__n, __max_segments * __work_group_size);
    ::std::size_t __segments =
        oneapi::dpl::__internal::__dpl_ceiling_div(__n, __work_group_size * __iters_per_work_item);

    auto __private_histograms =
        oneapi::dpl::__par_backend_hetero::__buffer<_ExecutionPolicy, _bin_type>(__exec, __segments * __num_bins)
            .get_buffer();

    return __exec.queue().submit([&](auto& __h) {
        __h.depends_on(__init_e);
        oneapi::dpl::__ranges::__require_access(__h, __input, __bins, __opt_range...);
        sycl::accessor __hacc_private{__private_histograms, __h, sycl::read_write, sycl::no_init};
        __h.parallel_for(
            sycl::nd_range<1>(__segments * __work_group_size, __work_group_size), [=](sycl::nd_item<1> __self_item) {
                constexpr auto _atomic_address_space = sycl::access::address_space::global_space;
                const ::std::size_t __self_lidx = __self_item.get_local_id(0);
                const ::std::size_t __wgroup_idx = __self_item.get_group(0);
                const ::std::size_t __seg_start = __work_group_size * __iters_per_work_item * __wgroup_idx;

                __clear_wglocal_histograms(__hacc_private, __wgroup_idx * __num_bins, __num_bins, __self_item);

                if (__seg_start + __work_group_size * __iters_per_work_item < __n)
                {
                    for (::std::size_t __idx = 0; __idx < __iters_per_work_item; __idx++)
                    {
                        ::std::size_t __val_idx = __seg_start + __idx * __work_group_size + __self_lidx;
                        __accum_local_atomics_iter<_histogram_index_type, _atomic_address_space>(
                            __input, __val_idx, __hacc_private, __wgroup_idx * __num_bins, __func);
                    }
                }
                else
                {
                    for (::std::size_t __idx = 0; __idx < __iters_per_work_item; __idx++)
                    {
                        ::std::size_t __val_idx = __seg_start + __idx * __work_group_size + __self_lidx;
                        if (__val_idx < __n)
                        {
                            __accum_local_atomics_iter<_histogram_index_type, _atomic_address_space>(
                                __input, __val_idx, __hacc_private, __wgroup_idx * __num_bins, __func);
                        }
                    }
                }

                __dpl_sycl::__group_barrier(__self_item);

                __reduce_out_histograms<_bin_type>(__hacc_private, __wgroup_idx * __num_bins, __bins, __num_bins,
                                                   __self_item);
            });
    });
}

template <::std::uint16_t __iters_per_work_item, typename _ExecutionPolicy, typename _Iter1, typename _Iter2,
          typename _Size, typename _IdxHashFunc, typename... _Range>
inline auto
__parallel_histogram_sycl_impl(_ExecutionPolicy&& __exec, _Iter1 __first, _Iter1 __last, _Iter2 __histogram_first,
                               const _Size& __num_bins, _IdxHashFunc __func, _Range&&... __opt_range)
{
    using __local_histogram_type = ::std::uint32_t;
    using __global_histogram_type = typename ::std::iterator_traits<_Iter2>::value_type;

    ::std::size_t __max_wgroup_size = oneapi::dpl::__internal::__max_work_group_size(__exec);

    ::std::uint16_t __work_group_size = ::std::min(::std::size_t(1024), __max_wgroup_size);

    auto __local_mem_size = __exec.queue().get_device().template get_info<sycl::info::device::local_mem_size>();
    constexpr ::std::uint8_t __max_work_item_private_bins = 16;

    auto keep_bins =
        oneapi::dpl::__ranges::__get_sycl_range<oneapi::dpl::__par_backend_hetero::access_mode::write, _Iter2>();
    auto bins_buf = keep_bins(__histogram_first, __histogram_first + __num_bins);

    auto __f = oneapi::dpl::__internal::fill_functor<__global_histogram_type>{__global_histogram_type{0}};
    //fill histogram bins with zeros
    auto init_e = oneapi::dpl::__par_backend_hetero::__parallel_for(
        __exec, unseq_backend::walk_n<_ExecutionPolicy, decltype(__f)>{__f}, __num_bins,
        bins_buf.all_view());
    auto n = __last - __first;

    if (n > 0)
    {
        auto keep_input =
            oneapi::dpl::__ranges::__get_sycl_range<oneapi::dpl::__par_backend_hetero::access_mode::read, _Iter1>();
        auto input_buf = keep_input(__first, __last);

        ::std::size_t req_SLM_bytes = __func.get_required_SLM_memory();
        ::std::size_t extra_SLM_elements =
            oneapi::dpl::__internal::__dpl_ceiling_div(req_SLM_bytes, sizeof(__local_histogram_type));
        // if bins fit into registers, use register private accumulation

        sycl::event e;
        if (__num_bins < __max_work_item_private_bins)
        {
            e = __histogram_general_registers_local_reduction<__iters_per_work_item, __max_work_item_private_bins>(
                ::std::forward<_ExecutionPolicy>(__exec), init_e, __work_group_size, input_buf.all_view(),
                bins_buf.all_view(), __num_bins, __func, ::std::forward<_Range...>(__opt_range)...);
        }
        // if bins fit into SLM, use local atomics
        else if ((__num_bins + extra_SLM_elements) * sizeof(__local_histogram_type) < __local_mem_size)
        {
            e = __histogram_general_local_atomics<__iters_per_work_item>(
                ::std::forward<_ExecutionPolicy>(__exec), init_e, __work_group_size, input_buf.all_view(),
                bins_buf.all_view(), __num_bins, __func, ::std::forward<_Range...>(__opt_range)...);
        }
        else // otherwise, use global atomics (private copies per workgroup)
        {
            e = __histogram_general_private_global_atomics<__iters_per_work_item>(
                ::std::forward<_ExecutionPolicy>(__exec), init_e, __work_group_size, input_buf.all_view(),
                bins_buf.all_view(), __num_bins, __func, ::std::forward<_Range...>(__opt_range)...);
        }
        return __future(e);
    }
    else
    {
        return init_e;
    }
}

template <::std::uint16_t __iters_per_work_item, typename _ExecutionPolicy, typename _Iter1, typename _Iter2,
          typename _Size, typename _IdxHashFunc, typename... _Range>
inline auto
__parallel_histogram_impl(_ExecutionPolicy&& __exec, _Iter1 __first, _Iter1 __last, _Iter2 __histogram_first,
                          const _Size& __num_bins, _IdxHashFunc __func, /*req_sycl_conversion = */ ::std::false_type,
                          _Range&&... __opt_range)
{
    //wrap binhash in a wrapper to allow shared memory boost where available
    return __parallel_histogram_sycl_impl<__iters_per_work_item>(
        ::std::forward<_ExecutionPolicy>(__exec), __first, __last, __histogram_first, __num_bins,
        __binhash_SLM_wrapper(__func), ::std::forward<_Range...>(__opt_range)...);
}

template <::std::uint16_t __iters_per_work_item, typename _ExecutionPolicy, typename _Iter1, typename _Iter2,
          typename _Size, typename _InternalRange>
inline auto
__parallel_histogram_impl(_ExecutionPolicy&& __exec, _Iter1 __first, _Iter1 __last, _Iter2 __histogram_first,
                          const _Size& __num_bins,
                          oneapi::dpl::__internal::__custom_range_binhash<_InternalRange> __func,
                          /*req_sycl_conversion = */ ::std::true_type)
{
    auto __range_to_upg = __func.get_range();
    //required to have this in the call stack to keep any created buffers alive
    auto __keep_boundaries =
        oneapi::dpl::__ranges::__get_sycl_range<oneapi::dpl::__par_backend_hetero::access_mode::read,
                                                decltype(__range_to_upg.begin())>();
    auto __boundary_buf = __keep_boundaries(__range_to_upg.begin(), __range_to_upg.end());
    auto __boundary_view = __boundary_buf.all_view();
    auto __bin_hash = oneapi::dpl::__internal::__custom_range_binhash(__boundary_view);
    return __parallel_histogram_impl<__iters_per_work_item>(
        ::std::forward<_ExecutionPolicy>(__exec), __first, __last, __histogram_first, __num_bins, __bin_hash,
        /*req_sycl_conversion = */ ::std::false_type{}, __boundary_view);
}

template <typename _ExecutionPolicy, typename _Iter1, typename _Iter2, typename _Size, typename _IdxHashFunc>
inline void
__parallel_histogram(_ExecutionPolicy&& __exec, _Iter1 __first, _Iter1 __last, _Iter2 __histogram_first,
                     const _Size& __num_bins, _IdxHashFunc __func)
{
    using _DoSyclConversion = typename _IdxHashFunc::req_sycl_range_conversion;
    __parallel_histogram_impl</*iters_per_workitem = */ 4>(::std::forward<_ExecutionPolicy>(__exec), __first, __last,
                                                           __histogram_first, __num_bins, __func, _DoSyclConversion{})
        .wait();
}

} // namespace __par_backend_hetero
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_PARALLEL_BACKEND_SYCL_HISTOGRAM_H
