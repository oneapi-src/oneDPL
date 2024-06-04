// -*- C++ -*-
//===----------------------------------------------------------------------===//
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

#ifndef _ONEDPL_KT_SINGLE_PASS_SCAN_H
#define _ONEDPL_KT_SINGLE_PASS_SCAN_H

#include "../../pstl/hetero/dpcpp/sycl_defs.h"
#include "../../pstl/hetero/dpcpp/unseq_backend_sycl.h"
#include "../../pstl/hetero/dpcpp/parallel_backend_sycl.h"
#include "../../pstl/hetero/dpcpp/execution_sycl_defs.h"
#include "../../pstl/utils.h"

#include <cstdint>
#include <cassert>
#include <cstddef>
#include <type_traits>

namespace oneapi::dpl::experimental::kt
{

namespace gpu
{

namespace __impl
{

template <typename... _Name>
class __copy_if_kernel;

template <typename... _Name>
class __lookback_init_kernel;

template <typename... _Name>
class __lookback_kernel;

static constexpr int SUBGROUP_SIZE = 32;

template <typename _ScanStatusFlag>
struct __scan_lookback_mem_mgr
{
    using _FlagStorageType = typename _ScanStatusFlag::_FlagStorageType;
    using _ValueType = typename _ScanStatusFlag::_ValueType;

    __scan_lookback_mem_mgr(sycl::queue __q, std::size_t __num_wgs) : __queue{__q}, __num_workgroups(__num_wgs){};

    void
    allocate()
    {
        __scan_memory_begin = sycl::malloc_device<std::uint8_t>(get_memory_size(), __queue);
        if (!__scan_memory_begin)
            throw std::bad_alloc();
    }

    sycl::event
    async_free(sycl::event __dependency)
    {
        return __queue.submit([__e = __dependency, __ptr = __scan_memory_begin, __q = __queue](sycl::handler& __hdl) {
            __hdl.depends_on(__e);
            __hdl.host_task([=]() { sycl::free(__ptr, __q); });
        });
    }

    void
    free()
    {
        sycl::free(__scan_memory_begin, __queue);
    }

    _FlagStorageType*
    get_flags_begin()
    {
        // Aligned flags
        void* __base_flags = reinterpret_cast<void*>(__scan_memory_begin + get_tile_values_bytes());
        auto __remainder = get_padded_flag_bytes();
        return reinterpret_cast<_FlagStorageType*>(
            ::std::align(::std::alignment_of_v<_FlagStorageType>, get_flag_bytes(), __base_flags, __remainder));
    }

    _ValueType*
    get_partial_values_begin()
    {
        return reinterpret_cast<_ValueType*>(__scan_memory_begin);
    }

    _ValueType*
    get_full_values_begin()
    {
        return reinterpret_cast<_ValueType*>(__scan_memory_begin + get_num_elements() * sizeof(_ValueType));
    }

    std::size_t
    get_num_elements()
    {
        return _ScanStatusFlag::__padding + __num_workgroups;
    }

  private:
    std::size_t
    get_tile_values_bytes()
    {
        return (2 * get_num_elements()) * sizeof(_ValueType);
    }

    std::size_t
    get_flag_bytes()
    {
        return get_num_elements() * sizeof(_FlagStorageType);
    }

    std::size_t
    get_padded_flag_bytes()
    {
        // sizeof(_FlagStorageType) extra bytes for possible intenal alignment
        return get_flag_bytes() + sizeof(_FlagStorageType);
    }

    std::size_t
    get_memory_size()
    {
        return get_tile_values_bytes() + get_padded_flag_bytes();
    }

    std::uint8_t* __scan_memory_begin = nullptr;
    std::size_t __num_workgroups;

    sycl::queue __queue;
};

template <typename _T>
struct __scan_status_flag
{
    using _FlagStorageType = uint32_t;
    using _ValueType = _T;
    using _AtomicFlagT = sycl::atomic_ref<_FlagStorageType, sycl::memory_order::acq_rel, sycl::memory_scope::device,
                                          sycl::access::address_space::global_space>;
    using _AtomicValueT = sycl::atomic_ref<_T, sycl::memory_order::acq_rel, sycl::memory_scope::device,
                                           sycl::access::address_space::global_space>;

    static constexpr _FlagStorageType __initialized_status = 0;
    static constexpr _FlagStorageType __partial_status = 1;
    static constexpr _FlagStorageType __full_status = 2;
    static constexpr _FlagStorageType __oob_status = 3;

    static constexpr int __padding = SUBGROUP_SIZE;

    __scan_status_flag(_FlagStorageType* __flags, _T* __full_vals, _T* __partial_vals, const std::uint32_t __tile_id)
        : __tile_id(__tile_id), __flags_begin(__flags), __full_vals_begin(__full_vals),
          __partial_vals_begin(__partial_vals), __atomic_flag(*(__flags + __tile_id + __padding)),
          __atomic_partial_value(*(__partial_vals + __tile_id + __padding)),
          __atomic_full_value(*(__full_vals + __tile_id + __padding))
    {
    }

    void
    set_partial(const _T __val)
    {
        __atomic_partial_value.store(__val, sycl::memory_order::release);
        __atomic_flag.store(__partial_status, sycl::memory_order::release);
    }

    void
    set_full(const _T __val)
    {
        __atomic_full_value.store(__val, sycl::memory_order::release);
        __atomic_flag.store(__full_status, sycl::memory_order::release);
    }

    template <typename _Subgroup, typename _BinaryOp>
    _T
    cooperative_lookback(const _Subgroup& __subgroup, _BinaryOp __binary_op)
    {
        _T __running = oneapi::dpl::unseq_backend::__known_identity<_BinaryOp, _T>;
        auto __local_id = __subgroup.get_local_id();

        for (int __tile = static_cast<int>(__tile_id) - 1; __tile >= 0; __tile -= SUBGROUP_SIZE)
        {
            _AtomicFlagT __tile_flag_atomic(*(__flags_begin + __tile + __padding - __local_id));
            _T __tile_flag = __initialized_status;

            // Load flag from a previous tile based on my local id.
            // Spin until every work-item in this subgroup reads a valid status
            do
            {
                __tile_flag = __tile_flag_atomic.load(sycl::memory_order::acquire);
            } while (!sycl::all_of_group(__subgroup, __tile_flag != __initialized_status));

            bool __is_full = __tile_flag == __full_status;
            auto __is_full_ballot = sycl::ext::oneapi::group_ballot(__subgroup, __is_full);
            std::uint32_t __is_full_ballot_bits{};
            __is_full_ballot.extract_bits(__is_full_ballot_bits);

            _AtomicValueT __tile_value_atomic(
                *((__is_full ? __full_vals_begin : __partial_vals_begin) + __tile + __padding - __local_id));
            _T __tile_val = __tile_value_atomic.load(sycl::memory_order::acquire);

            auto __lowest_item_with_full = sycl::ctz(__is_full_ballot_bits);
            _T __contribution = __local_id <= __lowest_item_with_full
                                    ? __tile_val
                                    : oneapi::dpl::unseq_backend::__known_identity<_BinaryOp, _T>;

            // Running reduction of all of the partial results from the tiles found, as well as the full contribution from the closest tile (if any)
            __running = __binary_op(__running, sycl::reduce_over_group(__subgroup, __contribution, __binary_op));

            // If we found a full value, we can stop looking at previous tiles. Otherwise,
            // keep going through tiles until we either find a full tile or we've completely
            // recomputed the prefix using partial values
            if (__is_full_ballot_bits)
                break;
        }
        return __running;
    }

    const uint32_t __tile_id;
    _FlagStorageType* __flags_begin;
    _T* __full_vals_begin;
    _T* __partial_vals_begin;
    _AtomicFlagT __atomic_flag;
    _AtomicValueT __atomic_partial_value;
    _AtomicValueT __atomic_full_value;
};

template <typename _FlagType, typename _Type, typename _BinaryOp, typename _KernelName>
struct __lookback_init_submitter;

template <typename _FlagType, typename _Type, typename _BinaryOp, typename... _Name>
struct __lookback_init_submitter<_FlagType, _Type, _BinaryOp,
                                 oneapi::dpl::__par_backend_hetero::__internal::__optional_kernel_name<_Name...>>
{
    template <typename _StatusFlags, typename _PartialValues>
    sycl::event
    operator()(sycl::queue __q, _StatusFlags&& __status_flags, _PartialValues&& __partial_values,
               std::size_t __status_flags_size, std::uint16_t __status_flag_padding) const
    {
        return __q.submit([&](sycl::handler& __hdl) {
            __hdl.parallel_for<_Name...>(sycl::range<1>{__status_flags_size}, [=](const sycl::item<1>& __item) {
                auto __id = __item.get_linear_id();
                __status_flags[__id] =
                    __id < __status_flag_padding ? _FlagType::__oob_status : _FlagType::__initialized_status;
                __partial_values[__id] = oneapi::dpl::unseq_backend::__known_identity<_BinaryOp, _Type>;
            });
        });
    }
};

template <typename _FlagType, typename _Group, typename _SubGroup, typename _StatusFlags, typename _StatusValues,
          typename _Type, typename _BinaryOp>
void
__lookback_phase(const _Group& __group, const _SubGroup& __subgroup, _StatusFlags __status_flags,
                 _StatusValues __status_vals_full, _StatusValues __status_vals_partial, std::uint32_t __tile_id,
                 const _Type& __local_reduction, _Type& __prev_tile_reduction, _BinaryOp __binary_op)
{
    // The last sub-group will query the previous tiles to find a prefix
    if (__subgroup.get_group_id() == (__subgroup.get_group_range()[0] - 1))
    {
        _FlagType __flag(__status_flags, __status_vals_full, __status_vals_partial, __tile_id);

        if (__subgroup.get_local_id() == __subgroup.get_local_range()[0] - 1)
        {
            __flag.set_partial(__local_reduction);
        }

        __prev_tile_reduction = __flag.cooperative_lookback(__subgroup, __binary_op);

        if (__subgroup.get_local_id() == __subgroup.get_local_range()[0] - 1)
        {
            __flag.set_full(__binary_op(__prev_tile_reduction, __local_reduction));
        }
    }
    __prev_tile_reduction = sycl::group_broadcast(__group, __prev_tile_reduction, __group.get_local_range()[0] - 1);
}

template <std::uint16_t __data_per_workitem, std::uint16_t __workgroup_size, typename _Type, typename _FlagType,
          typename _InRng, typename _OutRng, typename _BinaryOp, typename _StatusFlags, typename _StatusValues,
          typename _TileVals>
struct __lookback_kernel_func
{
    using _FlagStorageType = typename _FlagType::_FlagStorageType;
    static constexpr std::uint32_t __elems_in_tile = __workgroup_size * __data_per_workitem;

    _InRng __in_rng;
    _OutRng __out_rng;
    _BinaryOp __binary_op;
    std::size_t __n;
    _StatusFlags __status_flags;
    std::size_t __status_flags_size;
    _StatusValues __status_vals_full;
    _StatusValues __status_vals_partial;
    _TileVals __tile_vals;

    [[sycl::reqd_sub_group_size(SUBGROUP_SIZE)]] void
    operator()(const sycl::nd_item<1>& __item) const
    {
        auto __group = __item.get_group();
        auto __subgroup = __item.get_sub_group();
        auto __local_id = __item.get_local_id(0);

        std::uint32_t __tile_id = 0;

        // Obtain unique ID for this work-group that will be used in decoupled lookback
        if (__group.leader())
        {
            sycl::atomic_ref<_FlagStorageType, sycl::memory_order::relaxed, sycl::memory_scope::device,
                             sycl::access::address_space::global_space>
                __idx_atomic(__status_flags[__status_flags_size - 1]);
            __tile_id = __idx_atomic.fetch_add(1);
        }

        __tile_id = sycl::group_broadcast(__group, __tile_id, 0);

        std::size_t __current_offset = static_cast<std::size_t>(__tile_id) * __elems_in_tile;
        auto __out_begin = __out_rng.begin() + __current_offset;

        if (__current_offset >= __n)
            return;

        // Global load into local
        auto __wg_current_offset = (__tile_id * __elems_in_tile);
        auto __wg_next_offset = ((__tile_id + 1) * __elems_in_tile);
        auto __wg_local_memory_size = __elems_in_tile;

        if (__wg_next_offset > __n)
            __wg_local_memory_size = __n - __wg_current_offset;

        if (__wg_next_offset <= __n)
        {
            _ONEDPL_PRAGMA_UNROLL
            for (std::uint32_t __i = 0; __i < __data_per_workitem; ++__i)
            {
                __tile_vals[__local_id + __workgroup_size * __i] =
                    __in_rng[__wg_current_offset + __local_id + __workgroup_size * __i];
            }
        }
        else
        {
            _ONEDPL_PRAGMA_UNROLL
            for (std::uint32_t __i = 0; __i < __data_per_workitem; ++__i)
            {
                if (__wg_current_offset + __local_id + __workgroup_size * __i < __n)
                {
                    __tile_vals[__local_id + __workgroup_size * __i] =
                        __in_rng[__wg_current_offset + __local_id + __workgroup_size * __i];
                }
            }
        }

        auto __tile_vals_ptr = __dpl_sycl::__get_accessor_ptr(__tile_vals);
        _Type __local_reduction =
            sycl::joint_reduce(__group, __tile_vals_ptr, __tile_vals_ptr + __wg_local_memory_size, __binary_op);
        _Type __prev_tile_reduction{};

        __lookback_phase<_FlagType>(__group, __subgroup, __status_flags, __status_vals_full, __status_vals_partial,
                                    __tile_id, __local_reduction, __prev_tile_reduction, __binary_op);

        sycl::joint_inclusive_scan(__group, __tile_vals_ptr, __tile_vals_ptr + __wg_local_memory_size, __out_begin,
                                   __binary_op, __prev_tile_reduction);
    }
};

template <std::uint16_t __data_per_workitem, std::uint16_t __workgroup_size, typename _Type, typename _FlagType,
          typename _KernelName>
struct __lookback_scan_submitter;

template <std::uint16_t __data_per_workitem, std::uint16_t __workgroup_size, typename _Type, typename _FlagType,
          typename... _Name>
struct __lookback_scan_submitter<__data_per_workitem, __workgroup_size, _Type, _FlagType,
                                 oneapi::dpl::__par_backend_hetero::__internal::__optional_kernel_name<_Name...>>
{

    template <typename _InRng, typename _OutRng, typename _BinaryOp, typename _StatusFlags, typename _StatusValues>
    sycl::event
    operator()(sycl::queue __q, sycl::event __prev_event, _InRng&& __in_rng, _OutRng&& __out_rng, _BinaryOp __binary_op,
               std::size_t __n, _StatusFlags&& __status_flags, std::size_t __status_flags_size,
               _StatusValues&& __status_vals_full, _StatusValues&& __status_vals_partial,
               std::size_t __current_num_items) const
    {
        using _LocalAccessorType = sycl::local_accessor<_Type, 1>;
        using _KernelFunc =
            __lookback_kernel_func<__data_per_workitem, __workgroup_size, _Type, _FlagType, std::decay_t<_InRng>,
                                   std::decay_t<_OutRng>, std::decay_t<_BinaryOp>, std::decay_t<_StatusFlags>,
                                   std::decay_t<_StatusValues>, std::decay_t<_LocalAccessorType>>;

        static constexpr std::uint32_t __elems_in_tile = __workgroup_size * __data_per_workitem;

        return __q.submit([&](sycl::handler& __hdl) {
            auto __tile_vals = _LocalAccessorType(sycl::range<1>{__elems_in_tile}, __hdl);
            __hdl.depends_on(__prev_event);

            oneapi::dpl::__ranges::__require_access(__hdl, __in_rng, __out_rng);
            __hdl.parallel_for<_Name...>(sycl::nd_range<1>(__current_num_items, __workgroup_size),
                                         _KernelFunc{__in_rng, __out_rng, __binary_op, __n, __status_flags,
                                                     __status_flags_size, __status_vals_full, __status_vals_partial,
                                                     __tile_vals});
        });
    }
};

template <bool _Inclusive, typename _InRng, typename _OutRng, typename _BinaryOp, typename _KernelParam>
sycl::event
__single_pass_scan(sycl::queue __queue, _InRng&& __in_rng, _OutRng&& __out_rng, _BinaryOp __binary_op, _KernelParam)
{
    using _Type = oneapi::dpl::__internal::__value_t<_InRng>;
    using _FlagType = __scan_status_flag<_Type>;
    using _FlagStorageType = typename _FlagType::_FlagStorageType;

    using _KernelName = typename _KernelParam::kernel_name;
    using _LookbackInitKernel = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
        __lookback_init_kernel<_KernelName, _Type, _BinaryOp>>;
    using _LookbackKernel = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
        __lookback_kernel<_KernelName, _Type, _BinaryOp>>;

    const std::size_t __n = __in_rng.size();

    if (__n == 0)
        return sycl::event{};

    static_assert(_Inclusive, "Single-pass scan only available for inclusive scan");
    static_assert(oneapi::dpl::unseq_backend::__has_known_identity<_BinaryOp, _Type>::value,
                  "Only binary operators with known identity values are supported");

    assert("This device does not support 64-bit atomics" &&
           (sizeof(_Type) < 64 || __queue.get_device().has(sycl::aspect::atomic64)));

    // Next power of 2 greater than or equal to __n
    auto __n_uniform = ::oneapi::dpl::__internal::__dpl_bit_ceil(__n);

    if constexpr (std::negation_v<typename _KernelParam::single_wg_opt_out>)
    {
        // Perform a single-work group scan if the input is small
        if (oneapi::dpl::__par_backend_hetero::__group_scan_fits_in_slm<_Type>(__queue, __n, __n_uniform))
        {
            return oneapi::dpl::__par_backend_hetero::__parallel_transform_scan_single_group(
                oneapi::dpl::__internal::__device_backend_tag{},
                oneapi::dpl::execution::__dpl::make_device_policy<typename _KernelParam::kernel_name>(__queue),
                std::forward<_InRng>(__in_rng), std::forward<_OutRng>(__out_rng), __n,
                oneapi::dpl::__internal::__no_op{}, unseq_backend::__no_init_value<_Type>{}, __binary_op,
                std::true_type{});
        }
    }

    constexpr std::size_t __workgroup_size = _KernelParam::workgroup_size;
    constexpr std::size_t __data_per_workitem = _KernelParam::data_per_workitem;

    // Avoid non_uniform n by padding up to a multiple of workgroup_size
    std::size_t __elems_in_tile = __workgroup_size * __data_per_workitem;
    std::size_t __num_wgs = oneapi::dpl::__internal::__dpl_ceiling_div(__n, __elems_in_tile);

    __scan_lookback_mem_mgr<_FlagType> __device_mem_mgr(__queue, __num_wgs);

    __device_mem_mgr.allocate();

    _Type* __status_vals_full = __device_mem_mgr.get_full_values_begin();
    _Type* __status_vals_partial = __device_mem_mgr.get_partial_values_begin();
    _FlagStorageType* __status_flags = __device_mem_mgr.get_flags_begin();
    //adding 1 to the number elements to account for the tile id
    std::size_t __status_flags_size = __device_mem_mgr.get_num_elements() + 1;

    auto __fill_event = __lookback_init_submitter<_FlagType, _Type, _BinaryOp, _LookbackInitKernel>{}(
        __queue, __status_flags, __status_vals_partial, __status_flags_size, _FlagType::__padding);

    std::size_t __current_num_wgs = oneapi::dpl::__internal::__dpl_ceiling_div(__n, __elems_in_tile);
    std::size_t __current_num_items = __current_num_wgs * __workgroup_size;

    auto __prev_event =
        __lookback_scan_submitter<__data_per_workitem, __workgroup_size, _Type, _FlagType, _LookbackKernel>{}(
            __queue, __fill_event, __in_rng, __out_rng, __binary_op, __n, __status_flags, __status_flags_size,
            __status_vals_full, __status_vals_partial, __current_num_items);

    // TODO: Currently, the following portion of code makes this entire function synchronous.
    // Ideally, we should be able to use the asynchronous free below, but we have found that doing
    // so introduces a large unexplainable slowdown. Once this slowdown has been identified and corrected,
    // we should replace this code with the asynchronous version below.
    if (0)
    {
        return __device_mem_mgr.async_free(__prev_event);
    }
    else
    {
        __prev_event.wait();
        __device_mem_mgr.free();
        return __prev_event;
    }
}

template <std::uint16_t __data_per_workitem, std::uint16_t __workgroup_size, typename _FlagType, typename _InRng,
          typename _OutRng, typename _NumRng, typename _UnaryPredicate, typename _StatusFlags, typename _StatusValues,
          typename _TileValues>
struct __copy_if_kernel_func
{
    static constexpr std::uint32_t __elems_in_tile = __workgroup_size * __data_per_workitem;
    using _SizeT = std::size_t;
    using _BinaryOp = std::plus<_SizeT>;
    using _Type = oneapi::dpl::__internal::__value_t<_InRng>;
    using _FlagStorageType = typename _FlagType::_FlagStorageType;

    _InRng __in_rng;
    _OutRng __out_rng;
    _NumRng __num_rng;
    _SizeT __n;
    _UnaryPredicate __pred;
    _StatusFlags __status_flags;
    std::size_t __status_flags_size;
    _StatusValues __status_vals_full;
    _StatusValues __status_vals_partial;
    _TileValues __wg_copy_if_values;
    std::size_t __current_num_wgs;

    [[sycl::reqd_sub_group_size(SUBGROUP_SIZE)]] void
    operator()(const sycl::nd_item<1>& __item) const
    {
        auto __group = __item.get_group();
        auto __wg_local_id = __item.get_local_id(0);
        auto __sg = __item.get_sub_group();

        std::uint32_t __tile_id = 0;

        // Obtain unique ID for this work-group that will be used in decoupled lookback
        if (__group.leader())
        {
            sycl::atomic_ref<_FlagStorageType, sycl::memory_order::relaxed, sycl::memory_scope::device,
                             sycl::access::address_space::global_space>
                __idx_atomic(__status_flags[__status_flags_size - 1]);
            __tile_id = __idx_atomic.fetch_add(1);
        }

        __tile_id = sycl::group_broadcast(__group, __tile_id, 0);

        std::size_t __current_offset = static_cast<std::size_t>(__tile_id) * __elems_in_tile;

        std::uint16_t __wi_count = 0;
        // Phase 1: Create __wg_count and construct in-order __wg_copy_if_values

        if ((__tile_id + 1) * __elems_in_tile <= __n)
        {
#pragma unroll
            for (std::uint16_t __i = 0; __i < __data_per_workitem; ++__i)
            {
                _Type __val = __in_rng[__i + __wg_local_id * __data_per_workitem + __elems_in_tile * __tile_id];

                if (__pred(__val))
                {
                    __wg_copy_if_values[__wi_count + __wg_local_id * __data_per_workitem] = __val;
                    ++__wi_count;
                }
            }
        }
        else
        {
            // Edge of input, have to handle memory bounds
            for (std::uint16_t __i = 0; __i + __wg_local_id * __data_per_workitem + __elems_in_tile * __tile_id < __n;
                 ++__i)
            {
                _Type __val = __in_rng[__i + __wg_local_id * __data_per_workitem + __elems_in_tile * __tile_id];

                if (__pred(__val))
                {
                    __wg_copy_if_values[__wi_count + __wg_local_id * __data_per_workitem] = __val;
                    ++__wi_count;
                }
            }
        }
        _SizeT __wg_count = __wi_count;
        __wg_count = sycl::exclusive_scan_over_group(__group, __wg_count, _BinaryOp{});

        // Phase 2: Global scan across __wg_count
        _SizeT __copied_elements = 0;

        __lookback_phase<_FlagType>(__group, __sg, __status_flags, __status_vals_full, __status_vals_partial, __tile_id,
                                    __wg_count + __wi_count, __copied_elements, _BinaryOp{});

        // Phase 3: copy values to global memory
        for (std::uint16_t __i = 0; __i < __wi_count; ++__i)
        {
            __out_rng[__copied_elements + __wg_count + __i] =
                __wg_copy_if_values[__i + __wg_local_id * __data_per_workitem];
        }
        if (__tile_id == (__current_num_wgs - 1) && __wg_local_id == (__workgroup_size - 1))
            __num_rng[0] = __copied_elements + __wg_count + __wi_count;
    }
};

template <std::uint16_t __data_per_workitem, std::uint16_t __workgroup_size, typename _FlagType, typename _KernelName>
struct __copy_if_submitter;

template <std::uint16_t __data_per_workitem, std::uint16_t __workgroup_size, typename _FlagType, typename... _Name>
struct __copy_if_submitter<__data_per_workitem, __workgroup_size, _FlagType,
                           oneapi::dpl::__par_backend_hetero::__internal::__optional_kernel_name<_Name...>>
{

    template <typename _Event, typename _InRng, typename _OutRng, typename _NumCopiedRng, typename _UnaryPredicate,
              typename _StatusFlags, typename _StatusValues>
    sycl::event
    operator()(sycl::queue __q, _Event __fill_event, _InRng&& __in_rng, _OutRng&& __out_rng, _NumCopiedRng&& __num_rng,
               std::size_t __n, _UnaryPredicate __pred, _StatusFlags&& __status_flags, std::size_t __status_flags_size,
               _StatusValues&& __status_vals_full, _StatusValues&& __status_vals_partial,
               std::size_t __current_num_items, std::size_t __current_num_wgs) const
    {
        using _Type = oneapi::dpl::__internal::__value_t<_InRng>;
        using _LocalAccessorType = sycl::local_accessor<_Type, 1>;
        using _KernelFunc =
            __copy_if_kernel_func<__data_per_workitem, __workgroup_size, _FlagType, std::decay_t<_InRng>,
                                  std::decay_t<_OutRng>, std::decay_t<_NumCopiedRng>, _UnaryPredicate,
                                  std::decay_t<_StatusFlags>, std::decay_t<_StatusValues>,
                                  std::decay_t<_LocalAccessorType>>;

        static constexpr std::uint32_t __elems_in_tile = __workgroup_size * __data_per_workitem;

        return __q.submit([&](sycl::handler& __hdl) {
            auto __wg_copy_if_values = sycl::local_accessor<_Type, 1>(sycl::range<1>{__elems_in_tile}, __hdl);
            __hdl.depends_on(__fill_event);

            auto __tile_vals = _LocalAccessorType(sycl::range<1>{__elems_in_tile}, __hdl);
            oneapi::dpl::__ranges::__require_access(__hdl, __in_rng, __out_rng, __num_rng);
            __hdl.parallel_for<_Name...>(sycl::nd_range<1>(__current_num_items, __workgroup_size),
                                         _KernelFunc{__in_rng, __out_rng, __num_rng, __n, __pred, __status_flags,
                                                     __status_flags_size, __status_vals_full, __status_vals_partial,
                                                     __tile_vals, __current_num_wgs});
        });
    }
};

template <typename _InRng, typename _OutRng, typename _NumCopiedRng, typename _UnaryPredicate, typename _KernelParam>
sycl::event
single_pass_copy_if_impl(sycl::queue __queue, _InRng&& __in_rng, _OutRng&& __out_rng, _NumCopiedRng __num_rng,
                         _UnaryPredicate __pred, _KernelParam)
{
    using _SizeT = uint64_t;
    using _KernelName = __copy_if_kernel<typename _KernelParam::kernel_name>;
    using _Type = oneapi::dpl::__internal::__value_t<_InRng>;
    using _FlagType = __scan_status_flag<_SizeT>;

    using _BinaryOp = std::plus<_SizeT>;

    using _LookbackInitKernel = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
        __lookback_init_kernel<_KernelName, _SizeT, _BinaryOp>>;

    using _CopyIfKernel =
        oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<__copy_if_kernel<_KernelName, _Type>>;

    const std::size_t __n = __in_rng.size();

    // Next power of 2 greater than or equal to __n
    auto __n_uniform = ::oneapi::dpl::__internal::__dpl_bit_ceil(__n);

    if constexpr (std::negation_v<typename _KernelParam::single_wg_opt_out>)
    {
        //If we fit in a single WG SLM, use the single wg version from oneDPL main
        if (oneapi::dpl::__par_backend_hetero::__group_copy_if_fits_in_slm(__queue, __n, __n_uniform))
        {
            return oneapi::dpl::__par_backend_hetero::__dispatch_small_copy_if(
                oneapi::dpl::execution::__dpl::make_device_policy<_KernelName>(__queue), __n,
                std::forward<_InRng>(__in_rng), std::forward<_OutRng>(__out_rng),
                std::forward<_NumCopiedRng>(__num_rng), __pred);
        }
    }
    constexpr std::size_t __workgroup_size = _KernelParam::workgroup_size;
    constexpr std::size_t __elems_per_workitem = _KernelParam::data_per_workitem;

    // Avoid non_uniform n by padding up to a multiple of __workgroup_size
    constexpr std::uint32_t __elems_in_tile = __workgroup_size * __elems_per_workitem;
    std::size_t __current_num_wgs = oneapi::dpl::__internal::__dpl_ceiling_div(__n, __elems_in_tile);
    std::size_t __current_num_items = __current_num_wgs * __workgroup_size;

    __scan_lookback_mem_mgr<_FlagType> __device_mem_mgr(__queue, __current_num_wgs);
    __device_mem_mgr.allocate();

    // Memory Structure:
    // [Lookback Scan Memory, Tile Id Counter]
    auto __status_vals_full = __device_mem_mgr.get_full_values_begin();
    auto __status_vals_partial = __device_mem_mgr.get_partial_values_begin();
    auto __status_flags = __device_mem_mgr.get_flags_begin();
    //adding 1 to the number elements to account for the tile id
    std::size_t __status_flags_size = __device_mem_mgr.get_num_elements() + 1;

    auto __fill_event = __lookback_init_submitter<_FlagType, _SizeT, _BinaryOp, _LookbackInitKernel>{}(
        __queue, __status_flags, __status_vals_partial, __status_flags_size, _FlagType::__padding);

    sycl::event __prev_event = __copy_if_submitter<__elems_per_workitem, __workgroup_size, _FlagType, _CopyIfKernel>{}(
        __queue, __fill_event, __in_rng, __out_rng, __num_rng, __n, __pred, __status_flags, __status_flags_size,
        __status_vals_full, __status_vals_partial, __current_num_items, __current_num_wgs);

    // TODO: Currently, the following portion of code makes this entire function synchronous.
    // Ideally, we should be able to use the asynchronous free below, but we have found that doing
    // so introduces a large unexplainable slowdown. Once this slowdown has been identified and corrected,
    // we should replace this code with the asynchronous version below.
    if (0)
    {
        return __device_mem_mgr.async_free(__prev_event);
    }
    else
    {
        __prev_event.wait();
        __device_mem_mgr.free();
        return __prev_event;
    }
}

} // namespace __impl

template <typename _InRng, typename _OutRng, typename _NumCopiedRng, typename _UnaryPredicate, typename _KernelParam>
sycl::event
copy_if(sycl::queue __queue, _InRng&& __in_rng, _OutRng&& __out_rng, _NumCopiedRng&& __num_rng, _UnaryPredicate __pred,
        _KernelParam __param = {})
{
    auto __in_view = oneapi::dpl::__ranges::views::all(std::forward<_InRng>(__in_rng));
    auto __out_view = oneapi::dpl::__ranges::views::all(std::forward<_OutRng>(__out_rng));
    auto __num_view = oneapi::dpl::__ranges::views::all(std::forward<_OutRng>(__num_rng));

    return __impl::single_pass_copy_if_impl(__queue, std::move(__in_view), std::move(__out_view), std::move(__num_view),
                                            __pred, __param);
}

template <typename _InIterator, typename _OutIterator, typename _NumCopiedIterator, typename _UnaryPredicate,
          typename _KernelParam>
sycl::event
copy_if(sycl::queue __queue, _InIterator __in_begin, _InIterator __in_end, _OutIterator __out_begin,
        _NumCopiedIterator __num_begin, _UnaryPredicate __pred, _KernelParam __param = {})
{
    auto __n = __in_end - __in_begin;

    auto __keep1 = oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read, _InIterator>();
    auto __buf_in = __keep1(__in_begin, __in_end);
    auto __keep2 = oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::write, _OutIterator>();
    auto __buf_out = __keep2(__out_begin, __out_begin + __n);

    auto __keep_num =
        oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::write, _NumCopiedIterator>();
    auto __buf_num = __keep2(__num_begin, __num_begin + 1);

    return __impl::single_pass_copy_if_impl(__queue, __buf_in.all_view(), __buf_out.all_view(), __buf_num.all_view(),
                                            __pred, __param);
}

template <typename _InRng, typename _OutRng, typename _BinaryOp, typename _KernelParam>
sycl::event
inclusive_scan(sycl::queue __queue, _InRng&& __in_rng, _OutRng&& __out_rng, _BinaryOp __binary_op,
               _KernelParam __param = {})
{
    auto __in_view = oneapi::dpl::__ranges::views::all(std::forward<_InRng>(__in_rng));
    auto __out_view = oneapi::dpl::__ranges::views::all(std::forward<_OutRng>(__out_rng));

    return __impl::__single_pass_scan<true>(__queue, std::move(__in_view), std::move(__out_view), __binary_op, __param);
}

template <typename _InIterator, typename _OutIterator, typename _BinaryOp, typename _KernelParam>
sycl::event
inclusive_scan(sycl::queue __queue, _InIterator __in_begin, _InIterator __in_end, _OutIterator __out_begin,
               _BinaryOp __binary_op, _KernelParam __param = {})
{
    auto __n = __in_end - __in_begin;

    auto __keep1 = oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read, _InIterator>();
    auto __buf1 = __keep1(__in_begin, __in_end);
    auto __keep2 = oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::write, _OutIterator>();
    auto __buf2 = __keep2(__out_begin, __out_begin + __n);

    return __impl::__single_pass_scan<true>(__queue, __buf1.all_view(), __buf2.all_view(), __binary_op, __param);
}

} // namespace gpu

} // namespace oneapi::dpl::experimental::kt

#endif /* _ONEDPL_KT_SINGLE_PASS_SCAN_H */
