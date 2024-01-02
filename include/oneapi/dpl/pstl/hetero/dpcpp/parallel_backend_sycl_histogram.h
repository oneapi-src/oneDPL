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

// Baseline wrapper which provides no acceleration via SLM memory, but still
// allows generic calls to a wrapped binhash structure from within the kernels
template <typename _BinHash>
struct __binhash_SLM_wrapper
{
    //will always be empty, but just to have some type
    using extra_memory_type = typename ::std::uint8_t;
    _BinHash __bin_hash;
    __binhash_SLM_wrapper(_BinHash __bin_hash_) : __bin_hash(__bin_hash_) {}

    template <typename _T2>
    ::std::uint32_t
    get_bin(_T2&& __value) const
    {
        return __bin_hash.get_bin(::std::forward<_T2>(__value));
    }

    template <typename _T2>
    bool
    is_valid(_T2&& __value) const
    {
        return __bin_hash.is_valid(::std::forward<_T2>(__value));
    }

    ::std::size_t
    get_required_SLM_elements() const
    {
        return 0;
    }

    template <typename _ExtraMemAccessor>
    void
    init_SLM_memory(_ExtraMemAccessor /*__SLM_mem*/, const sycl::nd_item<1>& /*__self_item*/) const
    {
    }

    template <typename _T2, typename _ExtraMemAccessor>
    ::std::uint32_t
    get_bin(_T2&& __value, _ExtraMemAccessor /*__SLM_mem*/) const
    {
        return get_bin(::std::forward<_T2>(__value));
    }

    template <typename _T2, typename _ExtraMemAccessor>
    bool
    is_valid(_T2&& __value, _ExtraMemAccessor /*__SLM_mem*/) const
    {
        return is_valid(::std::forward<_T2>(__value));
    }

    void
    require_access(sycl::handler& __cgh)
    {
    }
};

// Specialization for custom range binhash function which stores boundary data
// into SLM for quick repeated usage
template <typename _Range>
struct __binhash_SLM_wrapper<oneapi::dpl::__internal::__custom_range_binhash<_Range>>
{
    using _BinHashType = typename oneapi::dpl::__internal::__custom_range_binhash<_Range>;
    using extra_memory_type = typename _BinHashType::__boundary_type;
    _BinHashType __bin_hash;

    __binhash_SLM_wrapper(_BinHashType __bin_hash_) : __bin_hash(__bin_hash_) {}

    template <typename _T2>
    ::std::uint32_t
    get_bin(_T2&& __value) const
    {
        return __bin_hash.get_bin(::std::forward<_T2>(__value));
    }

    template <typename _T2>
    bool
    is_valid(_T2&& __value) const
    {
        return __bin_hash.is_valid(::std::forward<_T2>(__value));
    }

    ::std::size_t
    get_required_SLM_elements()
    {
        return __bin_hash.__boundaries.size();
    }

    template <typename _ExtraMemAccessor>
    void
    init_SLM_memory(_ExtraMemAccessor __d_boundaries, const sycl::nd_item<1>& __self_item) const
    {
        ::std::uint32_t __gSize = __self_item.get_local_range()[0];
        ::std::uint32_t __self_lidx = __self_item.get_local_id(0);
        ::std::uint8_t __factor = oneapi::dpl::__internal::__dpl_ceiling_div(__bin_hash.__boundaries.size(), __gSize);
        ::std::uint8_t __k = 0;
        for (; __k < __factor - 1; ++__k)
        {
            __d_boundaries[__gSize * __k + __self_lidx] = __bin_hash.__boundaries[__gSize * __k + __self_lidx];
        }
        // residual
        if (__gSize * __k + __self_lidx < __bin_hash.__boundaries.size())
        {
            __d_boundaries[__gSize * __k + __self_lidx] = __bin_hash.__boundaries[__gSize * __k + __self_lidx];
        }
    }

    template <typename _T2, typename _ExtraMemAccessor>
    ::std::uint32_t
    get_bin(_T2&& __value, _ExtraMemAccessor __d_boundaries) const
    {
        return (::std::upper_bound(__d_boundaries.begin(), __d_boundaries.begin() + __bin_hash.__boundaries.size(),
                                   ::std::forward<_T2>(__value)) -
                __d_boundaries.begin()) -
               1;
    }

    template <typename _T2, typename _ExtraMemAccessor>
    bool
    is_valid(const _T2& __value, _ExtraMemAccessor __d_boundaries) const
    {
        return (__value >= __d_boundaries[0]) && (__value < __d_boundaries[__bin_hash.__boundaries.size() - 1]);
    }

    void
    require_access(sycl::handler& __cgh)
    {
        oneapi::dpl::__ranges::__require_access(__cgh, __bin_hash.get_range());
    }
};

//This wrapper is required to keep the buffer alive until the kernel has been completed (waited on)
template <typename _BinHash, typename _BufferType = int>
struct __binhash_buffer_holder
{
    _BufferType __buffer;
    _BinHash __bin_hash;

    //used for binhash with a buffer to keep alive
    __binhash_buffer_holder(_BinHash __bin_hash_, _BufferType __buffer_) : __bin_hash(__bin_hash_), __buffer(__buffer_)
    {
    }

    //used for binhash without a buffer to keep alive
    __binhash_buffer_holder(_BinHash __bin_hash_) : __bin_hash(__bin_hash_), __buffer() {}

    auto
    get_device_copyable_binhash()
    {
        return __bin_hash;
    }
};

template <typename _BinHash>
struct __make_sycl_upgraded_binhash
{
    auto
    operator()(_BinHash __bin_hash)
    {
        return __binhash_buffer_holder(__binhash_SLM_wrapper(__bin_hash));
    }
};

template <typename _Range>
struct __make_sycl_upgraded_binhash<oneapi::dpl::__internal::__custom_range_binhash<_Range>>
{
    auto
    operator()(oneapi::dpl::__internal::__custom_range_binhash<_Range> __bin_hash)
    {
        auto __range_to_upg = __bin_hash.get_range();
        auto __keep_boundaries =
            oneapi::dpl::__ranges::__get_sycl_range<oneapi::dpl::__par_backend_hetero::access_mode::read,
                                                    decltype(__range_to_upg.begin())>();
        auto __buffer = __keep_boundaries(__range_to_upg.begin(), __range_to_upg.end());
        return __binhash_buffer_holder(
            __binhash_SLM_wrapper(oneapi::dpl::__internal::__custom_range_binhash(__buffer.all_view())), __buffer);
    }
};

template <typename... _Name>
class __histo_kernel_register_local_red;

template <typename... _Name>
class __histo_kernel_local_atomics;

template <typename... _Name>
class __histo_kernel_private_glocal_atomics;

template <typename _HistAccessor, typename _OffsetT, typename _Size>
void
__clear_wglocal_histograms(const _HistAccessor& __local_histogram, const _OffsetT& __offset, _Size __num_bins,
                           const sycl::nd_item<1>& __self_item)
{
    ::std::uint32_t __gSize = __self_item.get_local_range()[0];
    ::std::uint32_t __self_lidx = __self_item.get_local_id(0);
    ::std::uint8_t __factor = oneapi::dpl::__internal::__dpl_ceiling_div(__num_bins, __gSize);
    ::std::uint8_t __k = 0;

    for (; __k < __factor - 1; ++__k)
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

template <typename _BinIdxType, typename _Range, typename _HistReg, typename _BinFunc, typename _ExtraMemAccessor>
void
__accum_local_register_iter(_Range&& __input, const ::std::size_t& __index, _HistReg* __histogram, _BinFunc __func,
                            _ExtraMemAccessor __SLM_mem)
{
    const auto& __x = __input[__index];
    if (__func.is_valid(__x, __SLM_mem))
    {
        _BinIdxType c = __func.get_bin(__x, __SLM_mem);
        ++__histogram[c];
    }
}

template <typename _BinIdxType, sycl::access::address_space _AddressSpace, typename _Range, typename _HistAccessor,
          typename _OffsetT, typename _BinFunc, typename... _ExtraMemType>
void
__accum_local_atomics_iter(_Range&& __input, const ::std::size_t& __index, const _HistAccessor& __wg_local_histogram,
                           const _OffsetT& __offset, _BinFunc __func, _ExtraMemType... __SLM_mem)
{
    using _histo_value_type = typename _HistAccessor::value_type;
    const auto& __x = __input[__index];
    if (__func.is_valid(__x, __SLM_mem...))
    {
        _BinIdxType __c = __func.get_bin(__x, __SLM_mem...);
        __dpl_sycl::__atomic_ref<_histo_value_type, _AddressSpace> __local_bin(__wg_local_histogram[__offset + __c]);
        ++__local_bin;
    }
}

template <typename _BinType, typename _FactorType, typename _HistAccessorIn, typename _OffsetT,
          typename _HistAccessorOut, typename _Size>
void
__reduce_out_histograms(const _HistAccessorIn& __in_histogram, const _OffsetT& __offset,
                        const _HistAccessorOut& __out_histogram, _Size __num_bins, const sycl::nd_item<1>& __self_item)
{
    ::std::uint32_t __gSize = __self_item.get_local_range()[0];
    ::std::uint32_t __self_lidx = __self_item.get_local_id(0);
    _FactorType __factor = oneapi::dpl::__internal::__dpl_ceiling_div(__num_bins, __gSize);
    _FactorType __k = 0;

    for (; __k < __factor - 1; ++__k)
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

template <::std::uint16_t __iters_per_work_item, ::std::uint8_t __bins_per_work_item, typename _KernelName>
struct __histogram_general_registers_local_reduction_submitter;

template <::std::uint16_t __iters_per_work_item, ::std::uint8_t __bins_per_work_item, typename... _KernelName>
struct __histogram_general_registers_local_reduction_submitter<__iters_per_work_item, __bins_per_work_item,
                                                               __internal::__optional_kernel_name<_KernelName...>>
{
    template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _IdxHashFunc>
    auto
    operator()(_ExecutionPolicy&& __exec, const sycl::event& __init_e, ::std::uint16_t __work_group_size,
               _Range1&& __input, _Range2&& __bins, _IdxHashFunc __func)
    {
        const ::std::size_t __n = __input.size();
        const ::std::uint8_t __num_bins = __bins.size();
        using _local_histogram_type = ::std::uint32_t;
        using _private_histogram_type = ::std::uint16_t;
        using _histogram_index_type = ::std::uint8_t;
        using _bin_type = oneapi::dpl::__internal::__value_t<_Range2>;
        using _extra_memory_type = typename _IdxHashFunc::extra_memory_type;

        ::std::size_t __extra_SLM_elements = __func.get_required_SLM_elements();
        ::std::size_t __segments =
            oneapi::dpl::__internal::__dpl_ceiling_div(__n, __work_group_size * __iters_per_work_item);
        return __exec.queue().submit([&](auto& __h) {
            __h.depends_on(__init_e);
            __func.require_access(__h);
            oneapi::dpl::__ranges::__require_access(__h, __input, __bins);
            __dpl_sycl::__local_accessor<_local_histogram_type> __local_histogram(sycl::range(__num_bins), __h);
            __dpl_sycl::__local_accessor<_extra_memory_type> __extra_SLM(sycl::range(__extra_SLM_elements), __h);
            __h.template parallel_for<_KernelName...>(
                sycl::nd_range<1>(__segments * __work_group_size, __work_group_size),
                [=](sycl::nd_item<1> __self_item) {
                    const ::std::size_t __self_lidx = __self_item.get_local_id(0);
                    const ::std::size_t __wgroup_idx = __self_item.get_group(0);
                    const ::std::size_t __seg_start = __work_group_size * __iters_per_work_item * __wgroup_idx;
                    __func.init_SLM_memory(__extra_SLM, __self_item);
                    __clear_wglocal_histograms(__local_histogram, 0, __num_bins, __self_item);
                    _private_histogram_type __histogram[__bins_per_work_item] = {0};

                    if (__seg_start + __work_group_size * __iters_per_work_item < __n)
                    {
                        _ONEDPL_PRAGMA_UNROLL
                        for (::std::uint8_t __idx = 0; __idx < __iters_per_work_item; ++__idx)
                        {
                            __accum_local_register_iter<_histogram_index_type>(
                                __input, __seg_start + __idx * __work_group_size + __self_lidx, __histogram, __func,
                                __extra_SLM);
                        }
                    }
                    else
                    {
                        _ONEDPL_PRAGMA_UNROLL
                        for (::std::uint8_t __idx = 0; __idx < __iters_per_work_item; ++__idx)
                        {
                            ::std::size_t __val_idx = __seg_start + __idx * __work_group_size + __self_lidx;
                            if (__val_idx < __n)
                            {
                                __accum_local_register_iter<_histogram_index_type>(__input, __val_idx, __histogram,
                                                                                   __func, __extra_SLM);
                            }
                        }
                    }

                    for (_histogram_index_type __k = 0; __k < __num_bins; ++__k)
                    {
                        __dpl_sycl::__atomic_ref<_local_histogram_type, sycl::access::address_space::local_space>
                            __local_bin(__local_histogram[__k]);
                        __local_bin += __histogram[__k];
                    }

                    __dpl_sycl::__group_barrier(__self_item);

                    __reduce_out_histograms<_bin_type, ::std::uint8_t>(__local_histogram, 0, __bins, __num_bins,
                                                                       __self_item);
                });
        });
    }
};

template <::std::uint16_t __iters_per_work_item, ::std::uint8_t __bins_per_work_item, typename _ExecutionPolicy,
          typename _Range1, typename _Range2, typename _IdxHashFunc>
auto
__histogram_general_registers_local_reduction(_ExecutionPolicy&& __exec, const sycl::event& __init_e,
                                              ::std::uint16_t __work_group_size, _Range1&& __input, _Range2&& __bins,
                                              _IdxHashFunc __func)
{
    using _KernelBaseName = typename ::std::decay_t<_ExecutionPolicy>::kernel_name;

    using _iters_per_work_item_t = ::std::integral_constant<::std::uint16_t, __iters_per_work_item>;

    // Required to include _iters_per_work_item_t in kernel name because we compile multiple kernels and decide between
    // them at runtime.  Other compile time arguments aren't required as it is the user's responsibility to provide a
    // unique kernel name to the policy for each call when using no-unamed-lambdas
    using _RegistersLocalReducName = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
        __histo_kernel_register_local_red<_iters_per_work_item_t, _KernelBaseName>>;

    return __histogram_general_registers_local_reduction_submitter<__iters_per_work_item, __bins_per_work_item,
                                                                   _RegistersLocalReducName>()(
        ::std::forward<_ExecutionPolicy>(__exec), __init_e, __work_group_size, ::std::forward<_Range1>(__input),
        ::std::forward<_Range2>(__bins), __func);
}

template <::std::uint16_t __iters_per_work_item, typename _KernelName>
struct __histogram_general_local_atomics_submitter;

template <::std::uint16_t __iters_per_work_item, typename... _KernelName>
struct __histogram_general_local_atomics_submitter<__iters_per_work_item,
                                                   __internal::__optional_kernel_name<_KernelName...>>
{
    template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _IdxHashFunc>
    auto
    operator()(_ExecutionPolicy&& __exec, const sycl::event& __init_e, ::std::uint16_t __work_group_size,
               _Range1&& __input, _Range2&& __bins, _IdxHashFunc __func)
    {
        using _local_histogram_type = ::std::uint32_t;
        using _bin_type = oneapi::dpl::__internal::__value_t<_Range2>;
        using _histogram_index_type = ::std::uint16_t;
        using _extra_memory_type = typename _IdxHashFunc::extra_memory_type;

        ::std::size_t __extra_SLM_elements = __func.get_required_SLM_elements();
        const ::std::size_t __n = __input.size();
        const ::std::size_t __num_bins = __bins.size();

        ::std::size_t __segments =
            oneapi::dpl::__internal::__dpl_ceiling_div(__n, __work_group_size * __iters_per_work_item);
        return __exec.queue().submit([&](auto& __h) {
            __h.depends_on(__init_e);
            __func.require_access(__h);
            oneapi::dpl::__ranges::__require_access(__h, __input, __bins);
            // minimum type size for atomics
            __dpl_sycl::__local_accessor<_local_histogram_type> __local_histogram(sycl::range(__num_bins), __h);
            __dpl_sycl::__local_accessor<_extra_memory_type> __extra_SLM(sycl::range(__extra_SLM_elements), __h);
            __h.template parallel_for<_KernelName...>(
                sycl::nd_range<1>(__segments * __work_group_size, __work_group_size),
                [=](sycl::nd_item<1> __self_item) {
                    constexpr auto _atomic_address_space = sycl::access::address_space::local_space;
                    const ::std::size_t __self_lidx = __self_item.get_local_id(0);
                    const ::std::uint32_t __wgroup_idx = __self_item.get_group(0);
                    const ::std::size_t __seg_start = __work_group_size * __wgroup_idx * __iters_per_work_item;
                    __func.init_SLM_memory(__extra_SLM, __self_item);

                    __clear_wglocal_histograms(__local_histogram, 0, __num_bins, __self_item);

                    if (__seg_start + __work_group_size * __iters_per_work_item < __n)
                    {
                        _ONEDPL_PRAGMA_UNROLL
                        for (::std::uint8_t __idx = 0; __idx < __iters_per_work_item; ++__idx)
                        {
                            ::std::size_t __val_idx = __seg_start + __idx * __work_group_size + __self_lidx;
                            __accum_local_atomics_iter<_histogram_index_type, _atomic_address_space>(
                                __input, __val_idx, __local_histogram, 0, __func, __extra_SLM);
                        }
                    }
                    else
                    {
                        _ONEDPL_PRAGMA_UNROLL
                        for (::std::uint8_t __idx = 0; __idx < __iters_per_work_item; ++__idx)
                        {
                            ::std::size_t __val_idx = __seg_start + __idx * __work_group_size + __self_lidx;
                            if (__val_idx < __n)
                            {
                                __accum_local_atomics_iter<_histogram_index_type, _atomic_address_space>(
                                    __input, __val_idx, __local_histogram, 0, __func, __extra_SLM);
                            }
                        }
                    }
                    __dpl_sycl::__group_barrier(__self_item);

                    __reduce_out_histograms<_bin_type, ::std::uint16_t>(__local_histogram, 0, __bins, __num_bins,
                                                                        __self_item);
                });
        });
    }
};

template <::std::uint16_t __iters_per_work_item, typename _ExecutionPolicy, typename _Range1, typename _Range2,
          typename _IdxHashFunc>
auto
__histogram_general_local_atomics(_ExecutionPolicy&& __exec, const sycl::event& __init_e,
                                  ::std::uint16_t __work_group_size, _Range1&& __input, _Range2&& __bins,
                                  _IdxHashFunc __func)
{
    using _KernelBaseName = typename ::std::decay_t<_ExecutionPolicy>::kernel_name;

    using _iters_per_work_item_t = ::std::integral_constant<::std::uint16_t, __iters_per_work_item>;

    // Required to include _iters_per_work_item_t in kernel name because we compile multiple kernels and decide between
    // them at runtime.  Other compile time arguments aren't required as it is the user's responsibility to provide a
    // unique kernel name to the policy for each call when using no-unamed-lambdas
    using _LocalAtomicsName = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
        __histo_kernel_local_atomics<_iters_per_work_item_t, _KernelBaseName>>;

    return __histogram_general_local_atomics_submitter<__iters_per_work_item, _LocalAtomicsName>()(
        ::std::forward<_ExecutionPolicy>(__exec), __init_e, __work_group_size, ::std::forward<_Range1>(__input),
        ::std::forward<_Range2>(__bins), __func);
}

template <typename _KernelName>
struct __histogram_general_private_global_atomics_submitter;

template <typename... _KernelName>
struct __histogram_general_private_global_atomics_submitter<__internal::__optional_kernel_name<_KernelName...>>
{
    template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _IdxHashFunc>
    auto
    operator()(_ExecutionPolicy&& __exec, const sycl::event& __init_e, ::std::uint16_t __min_iters_per_work_item,
               ::std::uint16_t __work_group_size, _Range1&& __input, _Range2&& __bins, _IdxHashFunc __func)
    {
        const ::std::size_t __n = __input.size();
        const ::std::size_t __num_bins = __bins.size();
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
            __func.require_access(__h);
            oneapi::dpl::__ranges::__require_access(__h, __input, __bins);
            sycl::accessor __hacc_private{__private_histograms, __h, sycl::read_write, sycl::no_init};
            __h.template parallel_for<_KernelName...>(
                sycl::nd_range<1>(__segments * __work_group_size, __work_group_size),
                [=](sycl::nd_item<1> __self_item) {
                    constexpr auto _atomic_address_space = sycl::access::address_space::global_space;
                    const ::std::size_t __self_lidx = __self_item.get_local_id(0);
                    const ::std::size_t __wgroup_idx = __self_item.get_group(0);
                    const ::std::size_t __seg_start = __work_group_size * __iters_per_work_item * __wgroup_idx;

                    __clear_wglocal_histograms(__hacc_private, __wgroup_idx * __num_bins, __num_bins, __self_item);

                    if (__seg_start + __work_group_size * __iters_per_work_item < __n)
                    {
                        for (::std::size_t __idx = 0; __idx < __iters_per_work_item; ++__idx)
                        {
                            ::std::size_t __val_idx = __seg_start + __idx * __work_group_size + __self_lidx;
                            __accum_local_atomics_iter<_histogram_index_type, _atomic_address_space>(
                                __input, __val_idx, __hacc_private, __wgroup_idx * __num_bins, __func);
                        }
                    }
                    else
                    {
                        for (::std::size_t __idx = 0; __idx < __iters_per_work_item; ++__idx)
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

                    __reduce_out_histograms<_bin_type, ::std::uint32_t>(__hacc_private, __wgroup_idx * __num_bins,
                                                                        __bins, __num_bins, __self_item);
                });
        });
    }
};
template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _IdxHashFunc>
auto
__histogram_general_private_global_atomics(_ExecutionPolicy&& __exec, const sycl::event& __init_e,
                                           ::std::uint16_t __min_iters_per_work_item, ::std::uint16_t __work_group_size,
                                           _Range1&& __input, _Range2&& __bins, _IdxHashFunc __func)
{
    using _KernelBaseName = typename ::std::decay_t<_ExecutionPolicy>::kernel_name;

    using _GlobalAtomicsName = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
        __histo_kernel_private_glocal_atomics<_KernelBaseName>>;

    return __histogram_general_private_global_atomics_submitter<_GlobalAtomicsName>()(
        ::std::forward<_ExecutionPolicy>(__exec), __init_e, __min_iters_per_work_item, __work_group_size,
        ::std::forward<_Range1>(__input), ::std::forward<_Range2>(__bins), __func);
}

template <::std::uint16_t __iters_per_work_item, typename _ExecutionPolicy, typename _Range1, typename _Range2,
          typename _IdxHashFunc>
auto
__parallel_histogram_select_kernel(_ExecutionPolicy&& __exec, const sycl::event& __init_e, _Range1&& __input,
                                   _Range2&& __bins, _IdxHashFunc __func)
{
    using _private_histogram_type = ::std::uint16_t;
    using _local_histogram_type = ::std::uint32_t;
    using _extra_memory_type = typename _IdxHashFunc::extra_memory_type;

    const auto __num_bins = __bins.size();
    ::std::size_t __max_wgroup_size = oneapi::dpl::__internal::__max_work_group_size(__exec);
    ::std::uint16_t __work_group_size = ::std::min(::std::size_t(1024), __max_wgroup_size);

    auto __local_mem_size = __exec.queue().get_device().template get_info<sycl::info::device::local_mem_size>();
    constexpr ::std::uint8_t __max_work_item_private_bins = 16 / sizeof(_private_histogram_type);

    // if bins fit into registers, use register private accumulation
    if (__num_bins <= __max_work_item_private_bins)
    {
        return __future(
            __histogram_general_registers_local_reduction<__iters_per_work_item, __max_work_item_private_bins>(
                ::std::forward<_ExecutionPolicy>(__exec), __init_e, __work_group_size, ::std::forward<_Range1>(__input),
                ::std::forward<_Range2>(__bins), __func));
    }
    // if bins fit into SLM, use local atomics
    else if (__num_bins * sizeof(_local_histogram_type) +
                 __func.get_required_SLM_elements() * sizeof(_extra_memory_type) <
             __local_mem_size)
    {
        return __future(__histogram_general_local_atomics<__iters_per_work_item>(
            ::std::forward<_ExecutionPolicy>(__exec), __init_e, __work_group_size, ::std::forward<_Range1>(__input),
            ::std::forward<_Range2>(__bins), __func));
    }
    else // otherwise, use global atomics (private copies per workgroup)
    {
        //Use __iters_per_work_item here as a runtime parameter, because only one kernel is created for
        // private_global_atomics with a variable number of iterations per workitem. __iters_per_work_item is just a
        // suggestion which but global memory limitations may increase this value to be able to fit the workgroup
        // private copies of the histogram bins in global memory.  No unrolling is taken advantage of here because it
        // is a runtime argument.
        return __future(__histogram_general_private_global_atomics(
            ::std::forward<_ExecutionPolicy>(__exec), __init_e, __iters_per_work_item, __work_group_size,
            ::std::forward<_Range1>(__input), ::std::forward<_Range2>(__bins), __func));
    }
}

template <typename _Name>
struct __hist_fill_zeros_wrapper
{
};

template <typename _ExecutionPolicy, typename _Iter1, typename _Size, typename _IdxHashFunc, typename _Iter2>
void
__parallel_histogram(_ExecutionPolicy&& __exec, _Iter1 __first, _Iter1 __last, _Size __num_bins, _IdxHashFunc __func,
                     _Iter2 __histogram_first)
{

    using _global_histogram_type = typename ::std::iterator_traits<_Iter2>::value_type;
    const auto __n = __last - __first;

    // The access mode we we want here is "read_write" + no_init property to cover the reads required by the main
    //  kernel, but also to avoid copying the data in unnecessarily.  In practice, this "write" access mode should
    //  accomplish this as write implies read, and we avoid a copy-in from the host for "write" access mode.
    // TODO: Add no_init property to get_sycl_range to allow expressivity we need here.
    auto __keep_bins =
        oneapi::dpl::__ranges::__get_sycl_range<oneapi::dpl::__par_backend_hetero::access_mode::write, _Iter2>();
    auto __bins_buf = __keep_bins(__histogram_first, __histogram_first + __num_bins);
    auto __bins = __bins_buf.all_view();

    auto __f = oneapi::dpl::__internal::fill_functor<_global_histogram_type>{_global_histogram_type{0}};
    //fill histogram bins with zeros

    auto __init_event = oneapi::dpl::__par_backend_hetero::__parallel_for(
        oneapi::dpl::__par_backend_hetero::make_wrapped_policy<__hist_fill_zeros_wrapper>(__exec),
        unseq_backend::walk_n<_ExecutionPolicy, decltype(__f)>{__f}, __num_bins, __bins);

    if (__n > 0)
    {
        //need __func_sycl_buffer_wrap to stay in scope until the kernel completes to keep the buffer alive
        auto __func_sycl_buffer_wrap = __make_sycl_upgraded_binhash<decltype(__func)>()(__func);
        auto __func_sycl = __func_sycl_buffer_wrap.get_device_copyable_binhash();
        auto __keep_input =
            oneapi::dpl::__ranges::__get_sycl_range<oneapi::dpl::__par_backend_hetero::access_mode::read, _Iter1>();
        auto __input_buf = __keep_input(__first, __last);

        if (__n < 1048576)
        {
            __parallel_histogram_select_kernel</*iters_per_workitem = */ 4>(::std::forward<_ExecutionPolicy>(__exec),
                                                                            __init_event, __input_buf.all_view(),
                                                                            ::std::move(__bins), __func_sycl)
                .wait();
        }
        else
        {
            __parallel_histogram_select_kernel</*iters_per_workitem = */ 32>(::std::forward<_ExecutionPolicy>(__exec),
                                                                             __init_event, __input_buf.all_view(),
                                                                             ::std::move(__bins), __func_sycl)
                .wait();
        }
    }
    else
    {
        __init_event.wait();
    }
}

} // namespace __par_backend_hetero
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_PARALLEL_BACKEND_SYCL_HISTOGRAM_H
