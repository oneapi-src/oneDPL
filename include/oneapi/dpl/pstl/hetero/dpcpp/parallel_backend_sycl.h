// -*- C++ -*-
//===-- parallel_backend_sycl.h -------------------------------------------===//
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

//!!! NOTE: This file should be included under the macro _ONEDPL_BACKEND_SYCL

// This header guard is used to check inclusion of DPC++ backend.
// Changing this macro may result in broken tests.
#ifndef _ONEDPL_PARALLEL_BACKEND_SYCL_H
#define _ONEDPL_PARALLEL_BACKEND_SYCL_H

#include <cassert>
#include <algorithm>
#include <type_traits>
#include <functional>
#include <utility>
#include <cmath>
#include <limits>
#include <cstdint>

#include "../../iterator_impl.h"
#include "../../execution_impl.h"
#include "../../utils_ranges.h"

#include "sycl_defs.h"
#include "parallel_backend_sycl_utils.h"
#include "parallel_backend_sycl_reduce.h"
#include "execution_sycl_defs.h"
#include "sycl_iterator.h"
#include "unseq_backend_sycl.h"
#include "utils_ranges_sycl.h"

#if _USE_RADIX_SORT
#    include "parallel_backend_sycl_radix_sort.h"
#endif

#include "sycl_traits.h" //SYCL traits specialization for some oneDPL types.

namespace oneapi
{
namespace dpl
{
namespace __par_backend_hetero
{

//-----------------------------------------------------------------------------
//- iter_mode_resolver
//-----------------------------------------------------------------------------

// iter_mode_resolver resolves the situations when
// the access mode provided by a user differs (inMode) from
// the access mode required by an algorithm (outMode).
// In general case iter_mode_resolver accepts the only situations
// when inMode == outMode,
// whereas the template specializations describe cases with specific
// inMode and outMode and the preferred access mode between the two.
template <access_mode inMode, access_mode outMode>
struct iter_mode_resolver
{
    static_assert(inMode == outMode, "Access mode provided by user conflicts with the one required by the algorithm");
    static constexpr access_mode value = inMode;
};

template <>
struct iter_mode_resolver<access_mode::read, access_mode::read_write>
{
    static constexpr access_mode value = access_mode::read;
};

template <>
struct iter_mode_resolver<access_mode::write, access_mode::read_write>
{
    static constexpr access_mode value = access_mode::write;
};

template <>
struct iter_mode_resolver<access_mode::read_write, access_mode::read>
{
    //TODO: warn user that the access mode is changed
    static constexpr access_mode value = access_mode::read;
};

template <>
struct iter_mode_resolver<access_mode::read_write, access_mode::write>
{
    //TODO: warn user that the access mode is changed
    static constexpr access_mode value = access_mode::write;
};

template <>
struct iter_mode_resolver<access_mode::discard_write, access_mode::write>
{
    static constexpr access_mode value = access_mode::discard_write;
};

template <>
struct iter_mode_resolver<access_mode::discard_read_write, access_mode::write>
{
    //TODO: warn user that the access mode is changed
    static constexpr access_mode value = access_mode::write;
};

template <>
struct iter_mode_resolver<access_mode::discard_read_write, access_mode::read_write>
{
    static constexpr access_mode value = access_mode::discard_read_write;
};

//-----------------------------------------------------------------------------
//- iter_mode
//-----------------------------------------------------------------------------

// create iterator with different access mode
template <access_mode outMode>
struct iter_mode
{
    // for common heterogeneous iterator
    template <template <access_mode, typename...> class Iter, access_mode inMode, typename... Types>
    Iter<iter_mode_resolver<inMode, outMode>::value, Types...>
    operator()(const Iter<inMode, Types...>& it)
    {
        constexpr access_mode preferredMode = iter_mode_resolver<inMode, outMode>::value;
        if (inMode == preferredMode)
            return it;
        return Iter<preferredMode, Types...>(it);
    }
    // for ounting_iterator
    template <typename T>
    oneapi::dpl::counting_iterator<T>
    operator()(const oneapi::dpl::counting_iterator<T>& it)
    {
        return it;
    }
    // for zip_iterator
    template <typename... Iters>
    auto
    operator()(const oneapi::dpl::zip_iterator<Iters...>& it)
        -> decltype(oneapi::dpl::__internal::map_zip(*this, it.base()))
    {
        return oneapi::dpl::__internal::map_zip(*this, it.base());
    }
    // for common iterator
    template <typename Iter>
    Iter
    operator()(const Iter& it1)
    {
        return it1;
    }
    // for raw pointers
    template <typename T>
    T*
    operator()(T* ptr)
    {
        // it does not have any iter mode because of two factors:
        //   - since it is a raw pointer, kernel can read/write despite of access_mode
        //   - access_mode also serves for implicit synchronization for buffers to build graph dependency
        //     and since usm have only explicit synchronization and does not provide dependency resolution mechanism
        //     it does not require access_mode
        return ptr;
    }

    template <typename T>
    const T*
    operator()(const T* ptr)
    {
        return ptr;
    }
};

template <access_mode outMode, typename _Iterator>
auto
make_iter_mode(const _Iterator& __it) -> decltype(iter_mode<outMode>()(__it))
{
    return iter_mode<outMode>()(__it);
}

// set of class templates to name kernels

template <typename... _Name>
class __scan_local_kernel;

template <typename... _Name>
class __scan_group_kernel;

template <typename... _Name>
class __find_or_kernel;

template <typename... _Name>
class __scan_propagate_kernel;

template <typename... _Name>
class __sort_leaf_kernel;

template <typename... _Name>
class __sort_global_kernel;

template <typename... _Name>
class __sort_copy_back_kernel;

template <typename... _Name>
class __scan_single_wg_kernel;

template <typename... _Name>
class __scan_single_wg_dynamic_kernel;

template <typename... Name>
class __scan_copy_single_wg_kernel;

//------------------------------------------------------------------------
// parallel_for - async pattern
//------------------------------------------------------------------------

// Use the trick with incomplete type and partial specialization to deduce the kernel name
// as the parameter pack that can be empty (for unnamed kernels) or contain exactly one
// type (for explicitly specified name by the user)
template <typename _KernelName>
struct __parallel_for_submitter;

template <typename... _Name>
struct __parallel_for_submitter<__internal::__optional_kernel_name<_Name...>>
{
    template <typename _ExecutionPolicy, typename _Fp, typename _Index, typename... _Ranges>
    auto
    operator()(_ExecutionPolicy&& __exec, _Fp __brick, _Index __count, _Ranges&&... __rngs) const
    {
        assert(oneapi::dpl::__ranges::__get_first_range_size(__rngs...) > 0);
        _PRINT_INFO_IN_DEBUG_MODE(__exec);
        auto __event = __exec.queue().submit([&__rngs..., &__brick, __count](sycl::handler& __cgh) {
            //get an access to data under SYCL buffer:
            oneapi::dpl::__ranges::__require_access(__cgh, __rngs...);

            __cgh.parallel_for<_Name...>(sycl::range</*dim=*/1>(__count), [=](sycl::item</*dim=*/1> __item_id) {
                auto __idx = __item_id.get_linear_id();
                __brick(__idx, __rngs...);
            });
        });
        return __future(__event);
    }
};

//General version of parallel_for, one additional parameter - __count of iterations of loop __cgh.parallel_for,
//for some algorithms happens that size of processing range is n, but amount of iterations is n/2.
template <typename _ExecutionPolicy, typename _Fp, typename _Index, typename... _Ranges>
auto
__parallel_for(oneapi::dpl::__internal::__device_backend_tag, _ExecutionPolicy&& __exec, _Fp __brick, _Index __count,
               _Ranges&&... __rngs)
{
    using _CustomName = oneapi::dpl::__internal::__policy_kernel_name<_ExecutionPolicy>;
    using _ForKernel = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<_CustomName>;

    return __parallel_for_submitter<_ForKernel>()(::std::forward<_ExecutionPolicy>(__exec), __brick, __count,
                                                  ::std::forward<_Ranges>(__rngs)...);
}

//------------------------------------------------------------------------
// parallel_transform_scan - async pattern
//------------------------------------------------------------------------
template <typename _GlobalScan, typename _Range2, typename _Range1, typename _Accessor, typename _Size>
struct __global_scan_caller
{
    __global_scan_caller(const _GlobalScan& __global_scan, const _Range2& __rng2, const _Range1& __rng1,
                         const _Accessor& __wg_sums_acc, _Size __n, ::std::size_t __size_per_wg)
        : __m_global_scan(__global_scan), __m_rng2(__rng2), __m_rng1(__rng1), __m_wg_sums_acc(__wg_sums_acc),
          __m_n(__n), __m_size_per_wg(__size_per_wg)
    {
    }

    void operator()(sycl::item<1> __item) const
    {
        __m_global_scan(__item, __m_rng2, __m_rng1, __m_wg_sums_acc, __m_n, __m_size_per_wg);
    }

  private:
    _GlobalScan __m_global_scan;
    _Range2 __m_rng2;
    _Range1 __m_rng1;
    _Accessor __m_wg_sums_acc;
    _Size __m_n;
    ::std::size_t __m_size_per_wg;
};

// Please see the comment for __parallel_for_submitter for optional kernel name explanation
template <typename _CustomName, typename _PropagateScanName>
struct __parallel_scan_submitter;

// Even if this class submits three kernel optional name is allowed to be only for one of them
// because for two others we have to provide the name to get the reliable work group size
template <typename _CustomName, typename... _PropagateScanName>
struct __parallel_scan_submitter<_CustomName, __internal::__optional_kernel_name<_PropagateScanName...>>
{
    template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _BinaryOperation,
              typename _InitType, typename _LocalScan, typename _GroupScan, typename _GlobalScan>
    auto
    operator()(_ExecutionPolicy&& __exec, _Range1&& __rng1, _Range2&& __rng2, _BinaryOperation __binary_op,
               _InitType __init, _LocalScan __local_scan, _GroupScan __group_scan, _GlobalScan __global_scan) const
    {
        using _Type = typename _InitType::__value_type;
        using _LocalScanKernel = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_generator<
            __scan_local_kernel, _CustomName, _Range1, _Range2, _Type, _LocalScan, _GroupScan, _GlobalScan>;
        using _GroupScanKernel = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_generator<
            __scan_group_kernel, _CustomName, _Range1, _Range2, _Type, _LocalScan, _GroupScan, _GlobalScan>;
        auto __n = __rng1.size();
        assert(__n > 0);

        auto __max_cu = oneapi::dpl::__internal::__max_compute_units(__exec);
        // get the work group size adjusted to the local memory limit
        // TODO: find a way to generalize getting of reliable work-group sizes
        ::std::size_t __wgroup_size = oneapi::dpl::__internal::__slm_adjusted_work_group_size(__exec, sizeof(_Type));

#if _ONEDPL_COMPILE_KERNEL
        //Actually there is one kernel_bundle for the all kernels of the pattern.
        auto __kernels = __internal::__kernel_compiler<_LocalScanKernel, _GroupScanKernel>::__compile(__exec);
        auto __kernel_1 = __kernels[0];
        auto __kernel_2 = __kernels[1];
        auto __wgroup_size_kernel_1 = oneapi::dpl::__internal::__kernel_work_group_size(__exec, __kernel_1);
        auto __wgroup_size_kernel_2 = oneapi::dpl::__internal::__kernel_work_group_size(__exec, __kernel_2);
        __wgroup_size = ::std::min({__wgroup_size, __wgroup_size_kernel_1, __wgroup_size_kernel_2});
#endif

        // Practically this is the better value that was found
        constexpr decltype(__wgroup_size) __iters_per_witem = 16;
        auto __size_per_wg = __iters_per_witem * __wgroup_size;
        auto __n_groups = (__n - 1) / __size_per_wg + 1;
        // Storage for the results of scan for each workgroup
        sycl::buffer<_Type> __wg_sums(__n_groups);

        _PRINT_INFO_IN_DEBUG_MODE(__exec, __wgroup_size, __max_cu);

        // 1. Local scan on each workgroup
        auto __submit_event = __exec.queue().submit([&](sycl::handler& __cgh) {
            oneapi::dpl::__ranges::__require_access(__cgh, __rng1, __rng2); //get an access to data under SYCL buffer
            auto __wg_sums_acc = __wg_sums.template get_access<access_mode::discard_write>(__cgh);
            __dpl_sycl::__local_accessor<_Type> __local_acc(__wgroup_size, __cgh);
#if _ONEDPL_COMPILE_KERNEL && _ONEDPL_KERNEL_BUNDLE_PRESENT
            __cgh.use_kernel_bundle(__kernel_1.get_kernel_bundle());
#endif
            __cgh.parallel_for<_LocalScanKernel>(
#if _ONEDPL_COMPILE_KERNEL && !_ONEDPL_KERNEL_BUNDLE_PRESENT
                __kernel_1,
#endif
                sycl::nd_range<1>(__n_groups * __wgroup_size, __wgroup_size), [=](sycl::nd_item<1> __item) {
                    __local_scan(__item, __n, __local_acc, __rng1, __rng2, __wg_sums_acc, __size_per_wg, __wgroup_size,
                                 __iters_per_witem, __init);
                });
        });
        // 2. Scan for the entire group of values scanned from each workgroup (runs on a single workgroup)
        if (__n_groups > 1)
        {
            auto __iters_per_single_wg = (__n_groups - 1) / __wgroup_size + 1;
            __submit_event = __exec.queue().submit([&](sycl::handler& __cgh) {
                __cgh.depends_on(__submit_event);
                auto __wg_sums_acc = __wg_sums.template get_access<access_mode::read_write>(__cgh);
                __dpl_sycl::__local_accessor<_Type> __local_acc(__wgroup_size, __cgh);
#if _ONEDPL_COMPILE_KERNEL && _ONEDPL_KERNEL_BUNDLE_PRESENT
                __cgh.use_kernel_bundle(__kernel_2.get_kernel_bundle());
#endif
                __cgh.parallel_for<_GroupScanKernel>(
#if _ONEDPL_COMPILE_KERNEL && !_ONEDPL_KERNEL_BUNDLE_PRESENT
                    __kernel_2,
#endif
                    // TODO: try to balance work between several workgroups instead of one
                    sycl::nd_range<1>(__wgroup_size, __wgroup_size), [=](sycl::nd_item<1> __item) {
                        __group_scan(__item, __n_groups, __local_acc, __wg_sums_acc, __wg_sums_acc,
                                     /*dummy*/ __wg_sums_acc, __n_groups, __wgroup_size, __iters_per_single_wg);
                    });
            });
        }

        // 3. Final scan for whole range
        auto __final_event = __exec.queue().submit([&](sycl::handler& __cgh) {
            __cgh.depends_on(__submit_event);
            oneapi::dpl::__ranges::__require_access(__cgh, __rng1, __rng2); //get an access to data under SYCL buffer
            auto __wg_sums_acc = __wg_sums.template get_access<access_mode::read>(__cgh);
            __cgh.parallel_for<_PropagateScanName...>(
                sycl::range<1>(__n_groups * __size_per_wg),
                __global_scan_caller<_GlobalScan, ::std::decay_t<_Range2>, ::std::decay_t<_Range1>,
                                     decltype(__wg_sums_acc), decltype(__n)>(__global_scan, __rng2, __rng1,
                                                                             __wg_sums_acc, __n, __size_per_wg));
        });

        return __future(__final_event, sycl::buffer(__wg_sums, sycl::id<1>(__n_groups - 1), sycl::range<1>(1)));
    }
};

template <typename _ValueType, bool _Inclusive, typename _Group, typename _Begin, typename _End, typename _OutIt,
          typename _BinaryOperation>
void
__scan_work_group(const _Group& __group, _Begin __begin, _End __end, _OutIt __out_it, _BinaryOperation __bin_op,
                  unseq_backend::__no_init_value<_ValueType>)
{
    if constexpr (_Inclusive)
        __dpl_sycl::__joint_inclusive_scan(__group, __begin, __end, __out_it, __bin_op);
    else
        __dpl_sycl::__joint_exclusive_scan(__group, __begin, __end, __out_it, __bin_op);
}

template <typename _ValueType, bool _Inclusive, typename _Group, typename _Begin, typename _End, typename _OutIt,
          typename _BinaryOperation>
void
__scan_work_group(const _Group& __group, _Begin __begin, _End __end, _OutIt __out_it, _BinaryOperation __bin_op,
                  unseq_backend::__init_value<_ValueType> __init)
{
    if constexpr (_Inclusive)
        __dpl_sycl::__joint_inclusive_scan(__group, __begin, __end, __out_it, __bin_op, __init.__value);
    else
        __dpl_sycl::__joint_exclusive_scan(__group, __begin, __end, __out_it, __init.__value, __bin_op);
}

template <bool _Inclusive, typename _KernelName>
struct __parallel_transform_scan_dynamic_single_group_submitter;

template <bool _Inclusive, typename... _ScanKernelName>
struct __parallel_transform_scan_dynamic_single_group_submitter<_Inclusive,
                                                                __internal::__optional_kernel_name<_ScanKernelName...>>
{
    template <typename _Policy, typename _InRng, typename _OutRng, typename _InitType, typename _BinaryOperation,
              typename _UnaryOp>
    auto
    operator()(const _Policy& __policy, _InRng&& __in_rng, _OutRng&& __out_rng, ::std::size_t __n, _InitType __init,
               _BinaryOperation __bin_op, _UnaryOp __unary_op, ::std::uint16_t __wg_size)
    {
        using _ValueType = typename _InitType::__value_type;

        const ::std::uint16_t __elems_per_item = oneapi::dpl::__internal::__dpl_ceiling_div(__n, __wg_size);
        const ::std::uint16_t __elems_per_wg = __elems_per_item * __wg_size;

        auto __event = __policy.queue().submit([&](sycl::handler& __hdl) {
            oneapi::dpl::__ranges::__require_access(__hdl, __in_rng, __out_rng);

            auto __lacc = __dpl_sycl::__local_accessor<_ValueType>(sycl::range<1>{__elems_per_wg}, __hdl);
            __hdl.parallel_for<_ScanKernelName...>(
                sycl::nd_range<1>(__wg_size, __wg_size), [=](sycl::nd_item<1> __self_item) {
                    const auto& __group = __self_item.get_group();
                    // This kernel is only launched for sizes less than 2^16
                    const ::std::uint16_t __item_id = __self_item.get_local_linear_id();

                    for (::std::uint16_t __idx = __item_id; __idx < __n; __idx += __wg_size)
                    {
                        __lacc[__idx] = __unary_op(__in_rng[__idx]);
                    }

                    auto __ptr = __dpl_sycl::__get_accessor_ptr(__lacc);
                    __scan_work_group<_ValueType, _Inclusive>(__group, __ptr, __ptr + __n, __ptr, __bin_op, __init);

                    for (::std::uint16_t __idx = __item_id; __idx < __n; __idx += __wg_size)
                    {
                        __out_rng[__idx] = __lacc[__idx];
                    }

                    const ::std::uint16_t __residual = __n % __wg_size;
                    const ::std::uint16_t __residual_start = __n - __residual;
                    if (__residual > 0 && __item_id < __residual)
                    {
                        auto __idx = __residual_start + __item_id;
                        __out_rng[__idx] = __lacc[__idx];
                    }
                });
        });
        return __future(__event);
    }
};

template <bool _Inclusive, ::std::uint16_t _ElemsPerItem, ::std::uint16_t _WGSize, bool _IsFullGroup,
          typename _KernelName>
struct __parallel_transform_scan_static_single_group_submitter;

template <bool _Inclusive, ::std::uint16_t _ElemsPerItem, ::std::uint16_t _WGSize, bool _IsFullGroup,
          typename... _ScanKernelName>
struct __parallel_transform_scan_static_single_group_submitter<_Inclusive, _ElemsPerItem, _WGSize, _IsFullGroup,
                                                               __internal::__optional_kernel_name<_ScanKernelName...>>
{
    template <typename _Policy, typename _InRng, typename _OutRng, typename _InitType, typename _BinaryOperation,
              typename _UnaryOp>
    auto
    operator()(const _Policy& __policy, _InRng&& __in_rng, _OutRng&& __out_rng, ::std::size_t __n, _InitType __init,
               _BinaryOperation __bin_op, _UnaryOp __unary_op)
    {
        using _ValueType = typename _InitType::__value_type;

        constexpr ::uint32_t __elems_per_wg = _ElemsPerItem * _WGSize;

        auto __event = __policy.queue().submit([&](sycl::handler& __hdl) {
            oneapi::dpl::__ranges::__require_access(__hdl, __in_rng, __out_rng);

            auto __lacc = __dpl_sycl::__local_accessor<_ValueType>(sycl::range<1>{__elems_per_wg}, __hdl);

            __hdl.parallel_for<_ScanKernelName...>(
                sycl::nd_range<1>(_WGSize, _WGSize), [=](sycl::nd_item<1> __self_item) {
                    const auto& __group = __self_item.get_group();
                    const auto& __subgroup = __self_item.get_sub_group();
                    // This kernel is only launched for sizes less than 2^16
                    const ::std::uint16_t __item_id = __self_item.get_local_linear_id();
                    const ::std::uint16_t __subgroup_id = __subgroup.get_group_id();
                    const ::std::uint16_t __subgroup_size = __subgroup.get_local_linear_range();

#if _ONEDPL_SYCL_SUB_GROUP_LOAD_STORE_PRESENT
                    constexpr bool __can_use_subgroup_load_store =
                        _IsFullGroup && oneapi::dpl::__internal::__range_has_raw_ptr_iterator_v<::std::decay_t<_InRng>>;
#else
                    constexpr bool __can_use_subgroup_load_store = false;
#endif

                    auto __lacc_ptr = __dpl_sycl::__get_accessor_ptr(__lacc);
                    if constexpr (__can_use_subgroup_load_store)
                    {
                        _ONEDPL_PRAGMA_UNROLL
                        for (::std::uint16_t __i = 0; __i < _ElemsPerItem; ++__i)
                        {
                            auto __idx = __i * _WGSize + __subgroup_id * __subgroup_size;
                            auto __val = __unary_op(__subgroup.load(__in_rng.begin() + __idx));
                            __subgroup.store(__lacc_ptr + __idx, __val);
                        }
                    }
                    else
                    {
                        for (::std::uint16_t __idx = __item_id; __idx < __n; __idx += _WGSize)
                        {
                            __lacc[__idx] = __unary_op(__in_rng[__idx]);
                        }
                    }

                    __scan_work_group<_ValueType, _Inclusive>(__group, __lacc_ptr, __lacc_ptr + __n,
                                                              __lacc_ptr, __bin_op, __init);

                    if constexpr (__can_use_subgroup_load_store)
                    {
                        _ONEDPL_PRAGMA_UNROLL
                        for (::std::uint16_t __i = 0; __i < _ElemsPerItem; ++__i)
                        {
                            auto __idx = __i * _WGSize + __subgroup_id * __subgroup_size;
                            auto __val = __subgroup.load(__lacc_ptr + __idx);
                            __subgroup.store(__out_rng.begin() + __idx, __val);
                        }
                    }
                    else
                    {
                        for (::std::uint16_t __idx = __item_id; __idx < __n; __idx += _WGSize)
                        {
                            __out_rng[__idx] = __lacc[__idx];
                        }

                        const ::std::uint16_t __residual = __n % _WGSize;
                        const ::std::uint16_t __residual_start = __n - __residual;
                        if (__item_id < __residual)
                        {
                            auto __idx = __residual_start + __item_id;
                            __out_rng[__idx] = __lacc[__idx];
                        }
                    }
                });
        });
        return __future(__event);
    }
};

template <typename _Size, ::std::uint16_t _ElemsPerItem, ::std::uint16_t _WGSize, bool _IsFullGroup,
          typename _KernelName>
struct __parallel_copy_if_static_single_group_submitter;

template <typename _Size, ::std::uint16_t _ElemsPerItem, ::std::uint16_t _WGSize, bool _IsFullGroup,
          typename... _ScanKernelName>
struct __parallel_copy_if_static_single_group_submitter<_Size, _ElemsPerItem, _WGSize, _IsFullGroup,
                                                        __internal::__optional_kernel_name<_ScanKernelName...>>
{
    template <typename _Policy, typename _InRng, typename _OutRng, typename _InitType, typename _BinaryOperation,
              typename _UnaryOp>
    auto
    operator()(const _Policy& __policy, _InRng&& __in_rng, _OutRng&& __out_rng, ::std::size_t __n, _InitType __init,
               _BinaryOperation __bin_op, _UnaryOp __unary_op)
    {
        using _ValueType = ::std::uint16_t;

        // This type is used as a workaround for when an internal tuple is assigned to ::std::tuple, such as
        // with zip_iterator
        using __tuple_type = typename ::oneapi::dpl::__internal::__get_tuple_type<
            typename ::std::decay_t<decltype(__in_rng[0])>, typename ::std::decay_t<decltype(__out_rng[0])>>::__type;

        constexpr ::std::uint32_t __elems_per_wg = _ElemsPerItem * _WGSize;

        sycl::buffer<_Size> __res(sycl::range<1>(1));

        auto __event = __policy.queue().submit([&](sycl::handler& __hdl) {
            oneapi::dpl::__ranges::__require_access(__hdl, __in_rng, __out_rng);

            // Local memory is split into two parts. The first half stores the result of applying the
            // predicate on each element of the input range. The second half stores the index of the output
            // range to copy elements of the input range.
            auto __lacc = __dpl_sycl::__local_accessor<_ValueType>(sycl::range<1>{__elems_per_wg * 2}, __hdl);
            auto __res_acc = __res.template get_access<access_mode::write>(__hdl);

            __hdl.parallel_for<_ScanKernelName...>(
                sycl::nd_range<1>(_WGSize, _WGSize), [=](sycl::nd_item<1> __self_item) {
                    const auto& __group = __self_item.get_group();
                    const auto& __subgroup = __self_item.get_sub_group();
                    // This kernel is only launched for sizes less than 2^16
                    const ::std::uint16_t __item_id = __self_item.get_local_linear_id();
                    const ::std::uint16_t __subgroup_id = __subgroup.get_group_id();
                    const ::std::uint16_t __subgroup_size = __subgroup.get_local_linear_range();

#if _ONEDPL_SYCL_SUB_GROUP_LOAD_STORE_PRESENT
                    constexpr bool __can_use_subgroup_load_store =
                        _IsFullGroup && oneapi::dpl::__internal::__range_has_raw_ptr_iterator_v<::std::decay_t<_InRng>>;
#else
                    constexpr bool __can_use_subgroup_load_store = false;
#endif
                    auto __lacc_ptr = __dpl_sycl::__get_accessor_ptr(__lacc);
                    if constexpr (__can_use_subgroup_load_store)
                    {
                        _ONEDPL_PRAGMA_UNROLL
                        for (::std::uint16_t __i = 0; __i < _ElemsPerItem; ++__i)
                        {
                            auto __idx = __i * _WGSize + __subgroup_id * __subgroup_size;
                            uint16_t __val = __unary_op(__subgroup.load(__in_rng.begin() + __idx));
                            __subgroup.store(__lacc_ptr + __idx, __val);
                        }
                    }
                    else
                    {
                        for (::std::uint16_t __idx = __item_id; __idx < __n; __idx += _WGSize)
                        {
                            __lacc[__idx] = __unary_op(__in_rng[__idx]);
                        }
                    }

                    __scan_work_group<_ValueType, /* _Inclusive */ false>(
                        __group, __lacc_ptr, __lacc_ptr + __elems_per_wg, __lacc_ptr + __elems_per_wg, __bin_op,
                         __init);

                    for (::std::uint16_t __idx = __item_id; __idx < __n; __idx += _WGSize)
                    {
                        if (__lacc[__idx])
                            __out_rng[__lacc[__idx + __elems_per_wg]] = static_cast<__tuple_type>(__in_rng[__idx]);
                    }

                    const ::std::uint16_t __residual = __n % _WGSize;
                    const ::std::uint16_t __residual_start = __n - __residual;
                    if (__item_id < __residual)
                    {
                        auto __idx = __residual_start + __item_id;
                        if (__lacc[__idx])
                            __out_rng[__lacc[__idx + __elems_per_wg]] = static_cast<__tuple_type>(__in_rng[__idx]);
                    }

                    if (__item_id == 0)
                    {
                        // Add predicate of last element to account for the scan's exclusivity
                        __res_acc[0] = __lacc[__elems_per_wg + __n - 1] + __lacc[__n - 1];
                    }
                });
        });
        return __future(__event, __res);
    }
};

template <typename _ExecutionPolicy, typename _InRng, typename _OutRng, typename _UnaryOperation, typename _InitType,
          typename _BinaryOperation, typename _Inclusive>
auto
__parallel_transform_scan_single_group(oneapi::dpl::__internal::__device_backend_tag, _ExecutionPolicy&& __exec,
                                       _InRng&& __in_rng, _OutRng&& __out_rng, ::std::size_t __n,
                                       _UnaryOperation __unary_op, _InitType __init, _BinaryOperation __binary_op,
                                       _Inclusive)
{
    using _CustomName = oneapi::dpl::__internal::__policy_kernel_name<_ExecutionPolicy>;

    ::std::size_t __max_wg_size = oneapi::dpl::__internal::__max_work_group_size(__exec);

    // Specialization for devices that have a max work-group size of 1024
    constexpr ::std::uint16_t __targeted_wg_size = 1024;

    if (__max_wg_size >= __targeted_wg_size)
    {
        auto __single_group_scan_f = [&](auto __size_constant) {
            constexpr ::std::uint16_t __size = decltype(__size_constant)::value;
            constexpr ::std::uint16_t __wg_size = ::std::min(__size, __targeted_wg_size);
            constexpr ::std::uint16_t __num_elems_per_item =
                oneapi::dpl::__internal::__dpl_ceiling_div(__size, __wg_size);
            const bool __is_full_group = __n == __wg_size;

            if (__is_full_group)
                return __parallel_transform_scan_static_single_group_submitter<
                    _Inclusive::value, __num_elems_per_item, __wg_size,
                    /* _IsFullGroup= */ true,
                    oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<__scan_single_wg_kernel<
                        ::std::integral_constant<::std::uint16_t, __wg_size>,
                        ::std::integral_constant<::std::uint16_t, __num_elems_per_item>, _BinaryOperation,
                        /* _IsFullGroup= */ std::true_type, _Inclusive, _CustomName>>>()(
                    ::std::forward<_ExecutionPolicy>(__exec), std::forward<_InRng>(__in_rng),
                    std::forward<_OutRng>(__out_rng), __n, __init, __binary_op, __unary_op);
            else
                return __parallel_transform_scan_static_single_group_submitter<
                    _Inclusive::value, __num_elems_per_item, __wg_size,
                    /* _IsFullGroup= */ false,
                    oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<__scan_single_wg_kernel<
                        ::std::integral_constant<::std::uint16_t, __wg_size>,
                        ::std::integral_constant<::std::uint16_t, __num_elems_per_item>, _BinaryOperation,
                        /* _IsFullGroup= */ ::std::false_type, _Inclusive, _CustomName>>>()(
                    ::std::forward<_ExecutionPolicy>(__exec), std::forward<_InRng>(__in_rng),
                    std::forward<_OutRng>(__out_rng), __n, __init, __binary_op, __unary_op);
        };
        if (__n <= 16)
            return __single_group_scan_f(std::integral_constant<::std::uint16_t, 16>{});
        else if (__n <= 32)
            return __single_group_scan_f(std::integral_constant<::std::uint16_t, 32>{});
        else if (__n <= 64)
            return __single_group_scan_f(std::integral_constant<::std::uint16_t, 64>{});
        else if (__n <= 128)
            return __single_group_scan_f(std::integral_constant<::std::uint16_t, 128>{});
        else if (__n <= 256)
            return __single_group_scan_f(std::integral_constant<::std::uint16_t, 256>{});
        else if (__n <= 512)
            return __single_group_scan_f(std::integral_constant<::std::uint16_t, 512>{});
        else if (__n <= 1024)
            return __single_group_scan_f(std::integral_constant<::std::uint16_t, 1024>{});
        else if (__n <= 2048)
            return __single_group_scan_f(std::integral_constant<::std::uint16_t, 2048>{});
        else if (__n <= 4096)
            return __single_group_scan_f(std::integral_constant<::std::uint16_t, 4096>{});
        else if (__n <= 8192)
            return __single_group_scan_f(std::integral_constant<::std::uint16_t, 8192>{});
        else
            return __single_group_scan_f(std::integral_constant<::std::uint16_t, 16384>{});
    }
    else
    {
        using _DynamicGroupScanKernel = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
            __par_backend_hetero::__scan_single_wg_dynamic_kernel<_BinaryOperation, _CustomName>>;

        return __parallel_transform_scan_dynamic_single_group_submitter<_Inclusive::value, _DynamicGroupScanKernel>()(
            ::std::forward<_ExecutionPolicy>(__exec), std::forward<_InRng>(__in_rng), std::forward<_OutRng>(__out_rng),
            __n, __init, __binary_op, __unary_op, __max_wg_size);
    }
}

template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _BinaryOperation, typename _InitType,
          typename _LocalScan, typename _GroupScan, typename _GlobalScan>
auto
__parallel_transform_scan_base(oneapi::dpl::__internal::__device_backend_tag, _ExecutionPolicy&& __exec,
                               _Range1&& __in_rng, _Range2&& __out_rng, _BinaryOperation __binary_op, _InitType __init,
                               _LocalScan __local_scan, _GroupScan __group_scan, _GlobalScan __global_scan)
{
    using _CustomName = oneapi::dpl::__internal::__policy_kernel_name<_ExecutionPolicy>;

    using _PropagateKernel =
        oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<__scan_propagate_kernel<_CustomName>>;

    return __parallel_scan_submitter<_CustomName, _PropagateKernel>()(
        ::std::forward<_ExecutionPolicy>(__exec), ::std::forward<_Range1>(__in_rng), ::std::forward<_Range2>(__out_rng),
        __binary_op, __init, __local_scan, __group_scan, __global_scan);
}

template <typename _Type>
bool
__group_scan_fits_in_slm(const sycl::queue& __queue, ::std::size_t __n, ::std::size_t __n_uniform)
{
    constexpr int __single_group_upper_limit = 16384;

    // Pessimistically only use half of the memory to take into account memory used by compiled kernel
    const ::std::size_t __max_slm_size =
        __queue.get_device().template get_info<sycl::info::device::local_mem_size>() / 2;
    const auto __req_slm_size = sizeof(_Type) * __n_uniform;

    return (__n <= __single_group_upper_limit && __max_slm_size >= __req_slm_size);
}

template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _UnaryOperation, typename _InitType,
          typename _BinaryOperation, typename _Inclusive>
auto
__parallel_transform_scan(oneapi::dpl::__internal::__device_backend_tag __backend_tag, _ExecutionPolicy&& __exec,
                          _Range1&& __in_rng, _Range2&& __out_rng, ::std::size_t __n, _UnaryOperation __unary_op,
                          _InitType __init, _BinaryOperation __binary_op, _Inclusive)
{
    using _Type = typename _InitType::__value_type;

    // Next power of 2 greater than or equal to __n
    auto __n_uniform = __n;
    if ((__n_uniform & (__n_uniform - 1)) != 0)
        __n_uniform = oneapi::dpl::__internal::__dpl_bit_floor(__n) << 1;

    constexpr bool __can_use_group_scan = unseq_backend::__has_known_identity<_BinaryOperation, _Type>::value;
    if constexpr (__can_use_group_scan)
    {
        if (__group_scan_fits_in_slm<_Type>(__exec.queue(), __n, __n_uniform))
        {
            return __parallel_transform_scan_single_group(
                __backend_tag, std::forward<_ExecutionPolicy>(__exec), ::std::forward<_Range1>(__in_rng),
                ::std::forward<_Range2>(__out_rng), __n, __unary_op, __init, __binary_op, _Inclusive{});
        }
    }

    // Either we can't use group scan or this input is too big for one workgroup
    using _Assigner = unseq_backend::__scan_assigner;
    using _NoAssign = unseq_backend::__scan_no_assign;
    using _UnaryFunctor = unseq_backend::walk_n<_ExecutionPolicy, _UnaryOperation>;
    using _NoOpFunctor = unseq_backend::walk_n<_ExecutionPolicy, oneapi::dpl::__internal::__no_op>;

    _Assigner __assign_op;
    _NoAssign __no_assign_op;
    _NoOpFunctor __get_data_op;

    return __future(
        __parallel_transform_scan_base(
            __backend_tag, ::std::forward<_ExecutionPolicy>(__exec), ::std::forward<_Range1>(__in_rng),
            ::std::forward<_Range2>(__out_rng), __binary_op, __init,
            // local scan
            unseq_backend::__scan<_Inclusive, _ExecutionPolicy, _BinaryOperation, _UnaryFunctor, _Assigner, _Assigner,
                                  _NoOpFunctor, _InitType>{__binary_op, _UnaryFunctor{__unary_op}, __assign_op,
                                                           __assign_op, __get_data_op},
            // scan between groups
            unseq_backend::__scan</*inclusive=*/::std::true_type, _ExecutionPolicy, _BinaryOperation, _NoOpFunctor,
                                  _NoAssign, _Assigner, _NoOpFunctor, unseq_backend::__no_init_value<_Type>>{
                __binary_op, _NoOpFunctor{}, __no_assign_op, __assign_op, __get_data_op},
            // global scan
            unseq_backend::__global_scan_functor<_Inclusive, _BinaryOperation, _InitType>{__binary_op, __init})
            .event());
}

template <typename _SizeType>
struct __invoke_single_group_copy_if
{
    // Specialization for devices that have a max work-group size of at least 1024
    static constexpr ::std::uint16_t __targeted_wg_size = 1024;

    template <::std::uint16_t _Size, typename _ExecutionPolicy, typename _InRng, typename _OutRng, typename _Pred>
    auto
    operator()(_ExecutionPolicy&& __exec, ::std::size_t __n, _InRng&& __in_rng, _OutRng&& __out_rng, _Pred&& __pred)
    {
        constexpr ::std::uint16_t __wg_size = ::std::min(_Size, __targeted_wg_size);
        constexpr ::std::uint16_t __num_elems_per_item = ::oneapi::dpl::__internal::__dpl_ceiling_div(_Size, __wg_size);
        const bool __is_full_group = __n == __wg_size;

        using _CustomName = oneapi::dpl::__internal::__policy_kernel_name<_ExecutionPolicy>;
        using _InitType = unseq_backend::__no_init_value<::std::uint16_t>;
        using _ReduceOp = ::std::plus<::std::uint16_t>;
        if (__is_full_group)
            return __par_backend_hetero::__parallel_copy_if_static_single_group_submitter<
                _SizeType, __num_elems_per_item, __wg_size, true,
                   oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
                    __scan_copy_single_wg_kernel<::std::integral_constant<::std::uint16_t, __wg_size>,
                                                ::std::integral_constant<::std::uint16_t, __num_elems_per_item>,
                                                /* _IsFullGroup= */ std::true_type, _CustomName>>
                >()(
                __exec, ::std::forward<_InRng>(__in_rng), ::std::forward<_OutRng>(__out_rng), __n, _InitType{},
                _ReduceOp{}, ::std::forward<_Pred>(__pred));
        else
            return __par_backend_hetero::__parallel_copy_if_static_single_group_submitter<
                _SizeType, __num_elems_per_item, __wg_size, false,
                   oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
                    __scan_copy_single_wg_kernel<::std::integral_constant<::std::uint16_t, __wg_size>,
                                                ::std::integral_constant<::std::uint16_t, __num_elems_per_item>,
                                                /* _IsFullGroup= */ std::false_type, _CustomName>>
                >()(
                __exec, ::std::forward<_InRng>(__in_rng), ::std::forward<_OutRng>(__out_rng), __n, _InitType{},
                _ReduceOp{}, ::std::forward<_Pred>(__pred));
    }
};

template <typename _ExecutionPolicy, typename _InRng, typename _OutRng, typename _Size, typename _CreateMaskOp,
          typename _CopyByMaskOp>
auto
__parallel_scan_copy(oneapi::dpl::__internal::__device_backend_tag __backend_tag, _ExecutionPolicy&& __exec,
                     _InRng&& __in_rng, _OutRng&& __out_rng, _Size __n, _CreateMaskOp __create_mask_op,
                     _CopyByMaskOp __copy_by_mask_op)
{
    using _ReduceOp = ::std::plus<_Size>;
    using _Assigner = unseq_backend::__scan_assigner;
    using _NoAssign = unseq_backend::__scan_no_assign;
    using _MaskAssigner = unseq_backend::__mask_assigner<1>;
    using _DataAcc = unseq_backend::walk_n<_ExecutionPolicy, oneapi::dpl::__internal::__no_op>;
    using _InitType = unseq_backend::__no_init_value<_Size>;

    _Assigner __assign_op;
    _ReduceOp __reduce_op;
    _DataAcc __get_data_op;
    _MaskAssigner __add_mask_op;

    // temporary buffer to store boolean mask
    oneapi::dpl::__par_backend_hetero::__buffer<_ExecutionPolicy, int32_t> __mask_buf(__exec, __n);

    return __parallel_transform_scan_base(
        __backend_tag, ::std::forward<_ExecutionPolicy>(__exec),
        oneapi::dpl::__ranges::make_zip_view(
            ::std::forward<_InRng>(__in_rng),
            oneapi::dpl::__ranges::all_view<int32_t, __par_backend_hetero::access_mode::read_write>(
                __mask_buf.get_buffer())),
        ::std::forward<_OutRng>(__out_rng), __reduce_op, _InitType{},
        // local scan
        unseq_backend::__scan</*inclusive*/ ::std::true_type, _ExecutionPolicy, _ReduceOp, _DataAcc, _Assigner,
                              _MaskAssigner, _CreateMaskOp, _InitType>{__reduce_op, __get_data_op, __assign_op,
                                                                       __add_mask_op, __create_mask_op},
        // scan between groups
        unseq_backend::__scan</*inclusive*/ ::std::true_type, _ExecutionPolicy, _ReduceOp, _DataAcc, _NoAssign,
                              _Assigner, _DataAcc, _InitType>{__reduce_op, __get_data_op, _NoAssign{}, __assign_op,
                                                              __get_data_op},
        // global scan
        __copy_by_mask_op);
}

template <typename _ExecutionPolicy, typename _InRng, typename _OutRng, typename _Size, typename _Pred>
auto
__parallel_copy_if(oneapi::dpl::__internal::__device_backend_tag __backend_tag, _ExecutionPolicy&& __exec,
                   _InRng&& __in_rng, _OutRng&& __out_rng, _Size __n, _Pred __pred)
{
    using _SingleGroupInvoker = __invoke_single_group_copy_if<_Size>;

    // Next power of 2 greater than or equal to __n
    auto __n_uniform = ::oneapi::dpl::__internal::__dpl_bit_ceil(static_cast<::std::make_unsigned_t<_Size>>(__n));

    // Pessimistically only use half of the memory to take into account memory used by compiled kernel
    const ::std::size_t __max_slm_size =
        __exec.queue().get_device().template get_info<sycl::info::device::local_mem_size>() / 2;

    // The kernel stores n integers for the predicate and another n integers for the offsets
    const auto __req_slm_size = sizeof(::std::uint16_t) * __n_uniform * 2;

    constexpr ::std::uint16_t __single_group_upper_limit = 16384;

    ::std::size_t __max_wg_size = oneapi::dpl::__internal::__max_work_group_size(__exec);

    if (__n <= __single_group_upper_limit && __max_slm_size >= __req_slm_size &&
        __max_wg_size >= _SingleGroupInvoker::__targeted_wg_size)
    {
        using _SizeBreakpoints =
            ::std::integer_sequence<::std::uint16_t, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384>;

        return __par_backend_hetero::__static_monotonic_dispatcher<_SizeBreakpoints>::__dispatch(
            _SingleGroupInvoker{}, __n, ::std::forward<_ExecutionPolicy>(__exec), __n, ::std::forward<_InRng>(__in_rng),
            ::std::forward<_OutRng>(__out_rng), __pred);
    }
    else
    {
        using _ReduceOp = ::std::plus<_Size>;
        using CreateOp = unseq_backend::__create_mask<_Pred, _Size>;
        using CopyOp = unseq_backend::__copy_by_mask<_ReduceOp, oneapi::dpl::__internal::__pstl_assign,
                                                     /*inclusive*/ ::std::true_type, 1>;

        return __parallel_scan_copy(__backend_tag, ::std::forward<_ExecutionPolicy>(__exec),
                                    ::std::forward<_InRng>(__in_rng), ::std::forward<_OutRng>(__out_rng), __n,
                                    CreateOp{__pred}, CopyOp{});
    }
}

//------------------------------------------------------------------------
// find_or tags
//------------------------------------------------------------------------

// Tag for __parallel_find_or to find the first element that satisfies predicate
template <typename _RangeType>
struct __parallel_find_forward_tag
{
// FPGA devices don't support 64-bit atomics
#if _ONEDPL_FPGA_DEVICE
    using _AtomicType = uint32_t;
#else
    using _AtomicType = oneapi::dpl::__internal::__difference_t<_RangeType>;
#endif
    using _Compare = oneapi::dpl::__internal::__pstl_less;

    // The template parameter is intended to unify __init_value in tags.
    template <typename _DiffType>
    constexpr static _AtomicType
    __init_value(_DiffType __val)
    {
        return __val;
    }
};

// Tag for __parallel_find_or to find the last element that satisfies predicate
template <typename _RangeType>
struct __parallel_find_backward_tag
{
// FPGA devices don't support 64-bit atomics
#if _ONEDPL_FPGA_DEVICE
    using _AtomicType = int32_t;
#else
    using _AtomicType = oneapi::dpl::__internal::__difference_t<_RangeType>;
#endif
    using _Compare = oneapi::dpl::__internal::__pstl_greater;

    template <typename _DiffType>
    constexpr static _AtomicType __init_value(_DiffType)
    {
        return _AtomicType{-1};
    }
};

// Tag for __parallel_find_or for or-semantic
struct __parallel_or_tag
{
    class __atomic_compare
    {
      public:
        template <typename _LocalAtomic, typename _GlobalAtomic>
        bool
        operator()(const _LocalAtomic& __found_local, const _GlobalAtomic& __found) const
        {
            return __found_local == 1 && __found == 0;
        }
    };

    using _AtomicType = int32_t;
    using _Compare = __atomic_compare;

    // The template parameter is intended to unify __init_value in tags.
    template <typename _DiffType>
    constexpr static _AtomicType __init_value(_DiffType)
    {
        return 0;
    }
};

//------------------------------------------------------------------------
// early_exit (find_or)
//------------------------------------------------------------------------

template <typename _ExecutionPolicy, typename _Pred>
struct __early_exit_find_or
{
    _Pred __pred;

    template <typename _NDItemId, typename _IterSize, typename _WgSize, typename _LocalAtomic, typename _Compare,
              typename _BrickTag, typename... _Ranges>
    void
    operator()(const _NDItemId __item_id, const _IterSize __n_iter, const _WgSize __wg_size, _Compare __comp,
               _LocalAtomic& __found_local, _BrickTag, _Ranges&&... __rngs) const
    {
        using __par_backend_hetero::__parallel_or_tag;
        using _OrTagType = ::std::is_same<_BrickTag, __par_backend_hetero::__parallel_or_tag>;
        using _BackwardTagType = ::std::is_same<typename _BrickTag::_Compare, oneapi::dpl::__internal::__pstl_greater>;

        auto __n = oneapi::dpl::__ranges::__get_first_range_size(__rngs...);

        ::std::size_t __shift = 16;
        ::std::size_t __local_idx = __item_id.get_local_id(0);
        ::std::size_t __group_idx = __item_id.get_group(0);

        // each work_item processes N_ELEMENTS with step SHIFT
        ::std::size_t __leader = (__local_idx / __shift) * __shift;
        ::std::size_t __init_index = __group_idx * __wg_size * __n_iter + __leader * __n_iter + __local_idx % __shift;

        // if our "line" is out of work group size, reduce the line to the number of the rest elements
        if (__wg_size - __leader < __shift)
            __shift = __wg_size - __leader;
        for (_IterSize __i = 0; __i < __n_iter; ++__i)
        {
            //in case of find-semantic __shifted_idx must be the same type as the atomic for a correct comparison
            using _ShiftedIdxType = ::std::conditional_t<_OrTagType::value, decltype(__init_index + __i * __shift),
                                                         decltype(__found_local.load())>;

            _IterSize __current_iter = __i;
            if constexpr (_BackwardTagType::value)
                __current_iter = __n_iter - 1 - __i;

            _ShiftedIdxType __shifted_idx = __init_index + __current_iter * __shift;
            // TODO:[Performance] the issue with atomic load (in comparison with __shifted_idx for early exit)
            // should be investigated later, with other HW
            if (__shifted_idx < __n && __pred(__shifted_idx, __rngs...))
            {
                if constexpr (_OrTagType::value)
                    __found_local.store(1);
                else
                {
                    for (auto __old = __found_local.load(); __comp(__shifted_idx, __old); __old = __found_local.load())
                    {
                        __found_local.compare_exchange_strong(__old, __shifted_idx);
                    }
                }
            }
        }
    }
};

//------------------------------------------------------------------------
// parallel_find_or - sync pattern
//------------------------------------------------------------------------

// Base pattern for __parallel_or and __parallel_find. The execution depends on tag type _BrickTag.
template <typename _ExecutionPolicy, typename _Brick, typename _BrickTag, typename... _Ranges>
::std::conditional_t<
    ::std::is_same_v<_BrickTag, __parallel_or_tag>, bool,
    oneapi::dpl::__internal::__difference_t<typename oneapi::dpl::__ranges::__get_first_range_type<_Ranges...>::type>>
__parallel_find_or(oneapi::dpl::__internal::__device_backend_tag, _ExecutionPolicy&& __exec, _Brick __f,
                   _BrickTag __brick_tag, _Ranges&&... __rngs)
{
    using _CustomName = oneapi::dpl::__internal::__policy_kernel_name<_ExecutionPolicy>;
    using _AtomicType = typename _BrickTag::_AtomicType;
    using _FindOrKernel =
        oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_generator<__find_or_kernel, _CustomName, _Brick,
                                                                               _BrickTag, _Ranges...>;

    constexpr bool __or_tag_check = ::std::is_same_v<_BrickTag, __parallel_or_tag>;
    auto __rng_n = oneapi::dpl::__ranges::__get_first_range_size(__rngs...);
    assert(__rng_n > 0);

    // TODO: find a way to generalize getting of reliable work-group size
    auto __wgroup_size = oneapi::dpl::__internal::__max_work_group_size(__exec);
#if _ONEDPL_COMPILE_KERNEL
    auto __kernel = __internal::__kernel_compiler<_FindOrKernel>::__compile(__exec);
    __wgroup_size = ::std::min(__wgroup_size, oneapi::dpl::__internal::__kernel_work_group_size(__exec, __kernel));
#endif
    auto __max_cu = oneapi::dpl::__internal::__max_compute_units(__exec);

    auto __n_groups = (__rng_n - 1) / __wgroup_size + 1;
    // TODO: try to change __n_groups with another formula for more perfect load balancing
    __n_groups = ::std::min(__n_groups, decltype(__n_groups)(__max_cu));

    auto __n_iter = (__rng_n - 1) / (__n_groups * __wgroup_size) + 1;

    _PRINT_INFO_IN_DEBUG_MODE(__exec, __wgroup_size, __max_cu);

    _AtomicType __init_value = _BrickTag::__init_value(__rng_n);
    auto __result = __init_value;

    auto __pred = oneapi::dpl::__par_backend_hetero::__early_exit_find_or<_ExecutionPolicy, _Brick>{__f};

    // scope is to copy data back to __result after destruction of temporary sycl:buffer
    {
        auto __temp = sycl::buffer<_AtomicType, 1>(&__result, 1); // temporary storage for global atomic

        // main parallel_for
        __exec.queue().submit([&](sycl::handler& __cgh) {
            oneapi::dpl::__ranges::__require_access(__cgh, __rngs...);
            auto __temp_acc = __temp.template get_access<access_mode::read_write>(__cgh);

            // create local accessor to connect atomic with
            __dpl_sycl::__local_accessor<_AtomicType> __temp_local(1, __cgh);
#if _ONEDPL_COMPILE_KERNEL && _ONEDPL_KERNEL_BUNDLE_PRESENT
            __cgh.use_kernel_bundle(__kernel.get_kernel_bundle());
#endif
            __cgh.parallel_for<_FindOrKernel>(
#if _ONEDPL_COMPILE_KERNEL && !_ONEDPL_KERNEL_BUNDLE_PRESENT
                __kernel,
#endif
                sycl::nd_range</*dim=*/1>(sycl::range</*dim=*/1>(__n_groups * __wgroup_size),
                                          sycl::range</*dim=*/1>(__wgroup_size)),
                [=](sycl::nd_item</*dim=*/1> __item_id) {
                    auto __local_idx = __item_id.get_local_id(0);

                    __dpl_sycl::__atomic_ref<_AtomicType, sycl::access::address_space::global_space> __found(
                        *__dpl_sycl::__get_accessor_ptr(__temp_acc));
                    __dpl_sycl::__atomic_ref<_AtomicType, sycl::access::address_space::local_space> __found_local(
                        *__dpl_sycl::__get_accessor_ptr(__temp_local));

                    // 1. Set initial value to local atomic
                    if (__local_idx == 0)
                        __found_local.store(__init_value);
                    __dpl_sycl::__group_barrier(__item_id);

                    // 2. Find any element that satisfies pred and set local atomic value to global atomic
                    constexpr auto __comp = typename _BrickTag::_Compare{};
                    __pred(__item_id, __n_iter, __wgroup_size, __comp, __found_local, __brick_tag, __rngs...);
                    __dpl_sycl::__group_barrier(__item_id);

                    // Set local atomic value to global atomic
                    if (__local_idx == 0 && __comp(__found_local.load(), __found.load()))
                    {
                        if constexpr (__or_tag_check)
                            __found.store(1);
                        else
                        {
                            for (auto __old = __found.load(); __comp(__found_local.load(), __old);
                                 __old = __found.load())
                            {
                                __found.compare_exchange_strong(__old, __found_local.load());
                            }
                        }
                    }
                });
        });
        //The end of the scope  -  a point of synchronization (on temporary sycl buffer destruction)
    }

    if constexpr (__or_tag_check)
        return __result;
    else
        return __result != __init_value ? __result : __rng_n;
}

//------------------------------------------------------------------------
// parallel_or - sync pattern
//------------------------------------------------------------------------

template <typename Name>
class __or_policy_wrapper
{
};

template <typename _ExecutionPolicy, typename _Iterator1, typename _Iterator2, typename _Brick>
bool
__parallel_or(oneapi::dpl::__internal::__device_backend_tag __backend_tag, _ExecutionPolicy&& __exec,
              _Iterator1 __first, _Iterator1 __last, _Iterator2 __s_first, _Iterator2 __s_last, _Brick __f)
{
    auto __keep = oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read, _Iterator1>();
    auto __buf = __keep(__first, __last);
    auto __s_keep = oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read, _Iterator2>();
    auto __s_buf = __s_keep(__s_first, __s_last);

    return oneapi::dpl::__par_backend_hetero::__parallel_find_or(
        __backend_tag,
        __par_backend_hetero::make_wrapped_policy<__or_policy_wrapper>(::std::forward<_ExecutionPolicy>(__exec)), __f,
        __parallel_or_tag{}, __buf.all_view(), __s_buf.all_view());
}

// Special overload for single sequence cases.
// TODO: check if similar pattern may apply to other algorithms. If so, these overloads should be moved out of
// backend code.
template <typename _ExecutionPolicy, typename _Iterator, typename _Brick>
bool
__parallel_or(oneapi::dpl::__internal::__device_backend_tag __backend_tag, _ExecutionPolicy&& __exec, _Iterator __first,
              _Iterator __last, _Brick __f)
{
    auto __keep = oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read, _Iterator>();
    auto __buf = __keep(__first, __last);

    return oneapi::dpl::__par_backend_hetero::__parallel_find_or(
        __backend_tag,
        __par_backend_hetero::make_wrapped_policy<__or_policy_wrapper>(::std::forward<_ExecutionPolicy>(__exec)), __f,
        __parallel_or_tag{}, __buf.all_view());
}

//------------------------------------------------------------------------
// parallel_find - sync pattern
//-----------------------------------------------------------------------

template <typename Name>
class __find_policy_wrapper
{
};

template <typename _ExecutionPolicy, typename _Iterator1, typename _Iterator2, typename _Brick, typename _IsFirst>
_Iterator1
__parallel_find(oneapi::dpl::__internal::__device_backend_tag __backend_tag, _ExecutionPolicy&& __exec,
                _Iterator1 __first, _Iterator1 __last, _Iterator2 __s_first, _Iterator2 __s_last, _Brick __f, _IsFirst)
{
    auto __keep = oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read, _Iterator1>();
    auto __buf = __keep(__first, __last);
    auto __s_keep = oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read, _Iterator2>();
    auto __s_buf = __s_keep(__s_first, __s_last);

    using _TagType = ::std::conditional_t<_IsFirst::value, __parallel_find_forward_tag<decltype(__buf.all_view())>,
                                          __parallel_find_backward_tag<decltype(__buf.all_view())>>;
    return __first + oneapi::dpl::__par_backend_hetero::__parallel_find_or(
                         __backend_tag,
                         __par_backend_hetero::make_wrapped_policy<__find_policy_wrapper>(
                             ::std::forward<_ExecutionPolicy>(__exec)),
                         __f, _TagType{}, __buf.all_view(), __s_buf.all_view());
}

// Special overload for single sequence cases.
// TODO: check if similar pattern may apply to other algorithms. If so, these overloads should be moved out of
// backend code.
template <typename _ExecutionPolicy, typename _Iterator, typename _Brick, typename _IsFirst>
_Iterator
__parallel_find(oneapi::dpl::__internal::__device_backend_tag __backend_tag, _ExecutionPolicy&& __exec,
                _Iterator __first, _Iterator __last, _Brick __f, _IsFirst)
{
    auto __keep = oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read, _Iterator>();
    auto __buf = __keep(__first, __last);

    using _TagType = ::std::conditional_t<_IsFirst::value, __parallel_find_forward_tag<decltype(__buf.all_view())>,
                                          __parallel_find_backward_tag<decltype(__buf.all_view())>>;
    return __first + oneapi::dpl::__par_backend_hetero::__parallel_find_or(
                         __backend_tag,
                         __par_backend_hetero::make_wrapped_policy<__find_policy_wrapper>(
                             ::std::forward<_ExecutionPolicy>(__exec)),
                         __f, _TagType{}, __buf.all_view());
}

//------------------------------------------------------------------------
// parallel_merge - async pattern
//-----------------------------------------------------------------------

// Partial merge implementation with O(log(k)) per routine complexity.
// Note: the routine assumes that the 2nd sequence goes after the first one, meaning that end_1 == start_2.
//
// The picture below shows how the merge is performed:
//
// input:
//    start_1     part_end_1   end_1  start_2     part_end_2   end_2
//      |_____________|_________|       |_____________|_________|
//      |______p1_____|___p2____|       |_____p3______|___p4____|
//
// Usual merge is performed on p1 and p3, the result is written to the beginning of the buffer.
// p2 and p4 are just copied to the then of the buffer as pictured below:
//
//    start_3
//      |_____________________________ __________________
//      |______sorted p1 and p3_______|____p2___|___p4___|
//
// Only first k elements from sorted p1 and p3 are guaranteed to be less than(or according to __comp) elements
// from p2 and p4. And these k elements are the only ones we care about.
template <typename _Ksize>
struct __partial_merge_kernel
{
    const _Ksize __k;
    template <typename _Idx, typename _Acc1, typename _Size1, typename _Acc2, typename _Size2, typename _Acc3,
              typename _Size3, typename _Compare>
    void
    operator()(_Idx __global_idx, const _Acc1& __in_acc1, _Size1 __start_1, _Size1 __end_1, const _Acc2& __in_acc2,
               _Size2 __start_2, _Size2 __end_2, const _Acc3& __out_acc, _Size3 __out_shift, _Compare __comp) const
    {
        const auto __part_end_1 = sycl::min(__start_1 + __k, __end_1);
        const auto __part_end_2 = sycl::min(__start_2 + __k, __end_2);

        // Handle elements from p1
        if (__global_idx >= __start_1 && __global_idx < __part_end_1)
        {
            const auto __shift =
                /* index inside p1 */ __global_idx - __start_1 +
                /* relative position in p3 */
                oneapi::dpl::__internal::__pstl_lower_bound(__in_acc2, __start_2, __part_end_2, __in_acc1[__global_idx],
                                                            __comp) -
                __start_2;
            __out_acc[__out_shift + __shift] = __in_acc1[__global_idx];
        }
        // Handle elements from p2
        else if (__global_idx >= __part_end_1 && __global_idx < __end_1)
        {
            const auto __shift =
                /* index inside p2 */ (__global_idx - __part_end_1) +
                /* size of p1 + size of p3 */ (__part_end_1 - __start_1) + (__part_end_2 - __start_2);
            __out_acc[__out_shift + __shift] = __in_acc1[__global_idx];
        }
        // Handle elements from p3
        else if (__global_idx >= __start_2 && __global_idx < __part_end_2)
        {
            const auto __shift =
                /* index inside p3 */ __global_idx - __start_2 +
                /* relative position in p1 */
                oneapi::dpl::__internal::__pstl_upper_bound(__in_acc1, __start_1, __part_end_1, __in_acc2[__global_idx],
                                                            __comp) -
                __start_1;
            __out_acc[__out_shift + __shift] = __in_acc2[__global_idx];
        }
        // Handle elements from p4
        else if (__global_idx >= __part_end_2 && __global_idx < __end_2)
        {
            const auto __shift =
                /* index inside p4 + size of p3 */ __global_idx - __start_2 +
                /* size of p1, p2 */ __end_1 - __start_1;
            __out_acc[__out_shift + __shift] = __in_acc2[__global_idx];
        }
    }
};

//Searching for an intersection of a merge matrix (n1, n2) diagonal with the Merge Path to define sub-ranges
//to serial merge. For example, a merge matrix for [0,1,1,2,3] and [0,0,2,3] is shown below:
//     0   1  1  2   3
//    ------------------
//   |--->
// 0 | 0 | 1  1  1   1
//   |   |
// 0 | 0 | 1  1  1   1
//   |   ---------->
// 2 | 0   0  0  0 | 1
//   |             ---->
// 3 | 0   0  0  0   0 |
template <typename _Rng1, typename _Rng2, typename _Index, typename _Index1, typename _Index2, typename _Compare>
auto
__find_start_point(const _Rng1& __rng1, const _Rng2& __rng2, _Index __i_elem, _Index1 __n1, _Index2 __n2,
                   _Compare __comp)
{
    _Index1 __start1 = 0;
    _Index2 __start2 = 0;
    if (__i_elem < __n2) //a condition to specify upper or lower part of the merge matrix to be processed
    {
        auto __q = __i_elem;                            //diagonal index
        auto __n_diag = ::std::min<_Index2>(__q, __n1); //diagonal size

        //searching for the first '1', a lower bound for a diagonal [0, 0,..., 0, 1, 1,.... 1, 1]
        oneapi::dpl::counting_iterator<_Index> __diag_it(0);
        auto __res = ::std::lower_bound(__diag_it, __diag_it + __n_diag, 1/*value to find*/,
            [&__rng2, &__rng1, __q, __comp](const auto& __i_diag, const auto& __value) mutable
            {
                auto __zero_or_one = __comp(__rng2[__q - __i_diag - 1], __rng1[__i_diag]);
                return __zero_or_one < __value;
            });
        __start1 = *__res;
        __start2 = __q - *__res;
    }
    else
    {
        auto __q = __i_elem - __n2;                            //diagonal index
        auto __n_diag = ::std::min<_Index1>(__n1 - __q, __n2); //diagonal size

        //searching for the first '1', a lower bound for a diagonal [0, 0,..., 0, 1, 1,.... 1, 1]
        oneapi::dpl::counting_iterator<_Index> __diag_it(0);
        auto __res = ::std::lower_bound(__diag_it, __diag_it + __n_diag, 1/*value to find*/,
            [&__rng2, &__rng1, __n2, __q, __comp](const auto& __i_diag, const auto& __value) mutable
            {
                auto __zero_or_one = __comp(__rng2[__n2 - __i_diag - 1], __rng1[__q + __i_diag]);
                return __zero_or_one < __value;
            });

        __start1 = __q + *__res;
        __start2 = __n2 - *__res;
    }
    return std::make_pair(__start1, __start2);
}

// Do serial merge of the data from rng1 (starting from start1) and rng2 (starting from start2) and writing
// to rng3 (starting from start3) in 'chunk' steps, but do not exceed the total size of the sequences (n1 and n2)
template <typename _Rng1, typename _Rng2, typename _Rng3, typename _Index1, typename _Index2, typename _Index3,
          typename _Size1, typename _Size2, typename _Compare>
void
__serial_merge(const _Rng1& __rng1, const _Rng2& __rng2, _Rng3& __rng3, _Index1 __start1, _Index2 __start2,
               _Index3 __start3, ::std::uint8_t __chunk, _Size1 __n1, _Size2 __n2, _Compare __comp)
{
    if (__start1 >= __n1)
    {
        //copying a residual of the second seq
        const auto __n = ::std::min<_Index2>(__n2 - __start2, __chunk);
        for (::std::uint8_t __i = 0; __i < __n; ++__i)
            __rng3[__start3 + __i] = __rng2[__start2 + __i];
    }
    else if (__start2 >= __n2)
    {
        //copying a residual of the first seq
        const auto __n = ::std::min<_Index1>(__n1 - __start1, __chunk);
        for (::std::uint8_t __i = 0; __i < __n; ++__i)
            __rng3[__start3 + __i] = __rng1[__start1 + __i];
    }
    else
    {
        ::std::uint8_t __n = __chunk;
        for (::std::uint8_t __i = 0; __i < __n && __start1 < __n1 && __start2 < __n2; ++__i)
        {
            const auto& __val1 = __rng1[__start1];
            const auto& __val2 = __rng2[__start2];
            if (__comp(__val2, __val1))
            {
                __rng3[__start3 + __i] = __val2;
                  if (++__start2 == __n2)
                  {
                      //copying a residual of the first seq
                      for (++__i; __i < __n && __start1 < __n1; ++__i, ++__start1)
                          __rng3[__start3 + __i] = __rng1[__start1];
                  }
            }
            else
            {
                __rng3[__start3 + __i] = __val1;
                if (++__start1 == __n1)
                {
                    //copying a residual of the second seq
                    for (++__i; __i < __n && __start2 < __n2; ++__i, ++__start2)
                        __rng3[__start3 + __i] = __rng2[__start2];
                }
            }
        }
    }
}

// Please see the comment for __parallel_for_submitter for optional kernel name explanation
template <typename _IdType, typename _Name>
struct __parallel_merge_submitter;

template <typename _IdType, typename... _Name>
struct __parallel_merge_submitter<_IdType, __internal::__optional_kernel_name<_Name...>>
{
    template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _Range3, typename _Compare>
    auto
    operator()(_ExecutionPolicy&& __exec, _Range1&& __rng1, _Range2&& __rng2, _Range3&& __rng3, _Compare __comp) const
    {
        using _Size = oneapi::dpl::__internal::__difference_t<_Range3>;

        auto __n1 = __rng1.size();
        auto __n2 = __rng2.size();

        assert(__n1 > 0 || __n2 > 0);

        _PRINT_INFO_IN_DEBUG_MODE(__exec);

        const ::std::uint8_t __chunk = __exec.queue().get_device().is_cpu() ? 128 : 4;
        const _Size __n = __n1 + __n2;
        const _Size __steps = oneapi::dpl::__internal::__dpl_ceiling_div(__n, __chunk);

        auto __event = __exec.queue().submit([&](sycl::handler& __cgh) {
            oneapi::dpl::__ranges::__require_access(__cgh, __rng1, __rng2, __rng3);
            __cgh.parallel_for<_Name...>(sycl::range</*dim=*/1>(__steps), [=](sycl::item</*dim=*/1> __item_id) {

                const _IdType __i_elem = __item_id.get_linear_id() * __chunk;
                const auto __start = __find_start_point(__rng1, __rng2, __i_elem, __n1, __n2, __comp);
                __serial_merge(__rng1, __rng2, __rng3, __start.first, __start.second, __i_elem, __chunk, __n1, __n2,
                               __comp);
            });
        });
        return __future(__event);
    }
};

template <typename... _Name>
class __merge_kernel_name;

template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _Range3, typename _Compare>
auto
__parallel_merge(oneapi::dpl::__internal::__device_backend_tag, _ExecutionPolicy&& __exec, _Range1&& __rng1,
                 _Range2&& __rng2, _Range3&& __rng3, _Compare __comp)
{
    using _CustomName = oneapi::dpl::__internal::__policy_kernel_name<_ExecutionPolicy>;

    const auto __n = __rng1.size() + __rng2.size();
    if (__n <= std::numeric_limits<::std::uint32_t>::max())
    {
        using _wi_index_type = ::std::uint32_t;
        using _MergeKernel = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
            __merge_kernel_name<_CustomName, _wi_index_type>>;
        return __parallel_merge_submitter<_wi_index_type, _MergeKernel>()(
            ::std::forward<_ExecutionPolicy>(__exec), ::std::forward<_Range1>(__rng1), ::std::forward<_Range2>(__rng2),
            ::std::forward<_Range3>(__rng3), __comp);
    }
    else
    {
        using _wi_index_type = ::std::uint64_t;
        using _MergeKernel = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
            __merge_kernel_name<_CustomName, _wi_index_type>>;
        return __parallel_merge_submitter<_wi_index_type, _MergeKernel>()(
            ::std::forward<_ExecutionPolicy>(__exec), ::std::forward<_Range1>(__rng1), ::std::forward<_Range2>(__rng2),
            ::std::forward<_Range3>(__rng3), __comp);
    }
}

//-----------------------------------------------------------------------
// parallel_sort: general implementation
//-----------------------------------------------------------------------
struct __leaf_sort_kernel
{
    template <typename _Acc, typename _Size1, typename _Compare>
    void
    operator()(const _Acc& __acc, const _Size1 __start, const _Size1 __end, _Compare __comp) const
    {
        for (_Size1 i = __start; i < __end; ++i)
        {
            for (_Size1 j = __start + 1; j < __start + __end - i; ++j)
            {
                // forwarding references allow binding of internal tuple of references with rvalue
                auto&& __first_item = __acc[j - 1];
                auto&& __second_item = __acc[j];
                if (__comp(__second_item, __first_item))
                {
                    using ::std::swap;
                    swap(__first_item, __second_item);
                }
            }
        }
    }
};

// Please see the comment for __parallel_for_submitter for optional kernel name explanation
template <typename _IdType, typename _LeafSortName, typename _GlobalSortName, typename _CopyBackName>
struct __parallel_sort_submitter;

template <typename _IdType, typename... _LeafSortName, typename... _GlobalSortName, typename... _CopyBackName>
struct __parallel_sort_submitter<_IdType, __internal::__optional_kernel_name<_LeafSortName...>,
                                 __internal::__optional_kernel_name<_GlobalSortName...>,
                                 __internal::__optional_kernel_name<_CopyBackName...>>
{
    template <typename _BackendTag, typename _ExecutionPolicy, typename _Range, typename _Compare>
    auto
    operator()(_BackendTag, _ExecutionPolicy&& __exec, _Range&& __rng, _Compare __comp) const
    {
        using _Tp = oneapi::dpl::__internal::__value_t<_Range>;
        using _Size = oneapi::dpl::__internal::__difference_t<_Range>;

        const ::std::size_t __n = __rng.size();
        assert(__n > 1);

        const bool __is_cpu = __exec.queue().get_device().is_cpu();
        const ::std::uint32_t __leaf = __is_cpu ? 16 : 4;
        _Size __steps = oneapi::dpl::__internal::__dpl_ceiling_div(__n, __leaf);

        // 1. Perform sorting of the leaves of the merge sort tree
        sycl::event __event1 = __exec.queue().submit([&](sycl::handler& __cgh) {
            oneapi::dpl::__ranges::__require_access(__cgh, __rng);
            __cgh.parallel_for<_LeafSortName...>(sycl::range</*dim=*/1>(__steps), [=](sycl::item</*dim=*/1> __item_id)
            {
                const _IdType __i_elem = __item_id.get_linear_id() * __leaf;
                __leaf_sort_kernel()(__rng, __i_elem, std::min<_IdType>(__i_elem + __leaf, __n), __comp);
            });
        });

        // 2. Merge sorting
        oneapi::dpl::__par_backend_hetero::__buffer<_ExecutionPolicy, _Tp> __temp_buf(__exec, __n);
        auto __temp = __temp_buf.get_buffer();
        bool __data_in_temp = false;
        _IdType __n_sorted = __leaf;
        const ::std::uint32_t __chunk = __is_cpu ? 32 : 4;
        __steps = oneapi::dpl::__internal::__dpl_ceiling_div(__n, __chunk);

        const ::std::size_t __n_power2 = oneapi::dpl::__internal::__dpl_bit_ceil(__n);
        const ::std::int64_t __n_iter = ::std::log2(__n_power2) - ::std::log2(__leaf);
        for (::std::int64_t __i = 0; __i < __n_iter; ++__i)
        {
            __event1 = __exec.queue().submit([&, __n_sorted, __data_in_temp](sycl::handler& __cgh) {
                __cgh.depends_on(__event1);

                oneapi::dpl::__ranges::__require_access(__cgh, __rng);
                sycl::accessor __dst(__temp, __cgh, sycl::read_write, sycl::no_init);

                __cgh.parallel_for<_GlobalSortName...>(sycl::range</*dim=*/1>(__steps), [=](sycl::item</*dim=*/1> __item_id)
                    {
                        const _IdType __i_elem = __item_id.get_linear_id() * __chunk;
                        const auto __i_elem_local = __i_elem % (__n_sorted * 2);

                        const auto __offset = ::std::min<_IdType>((__i_elem / (__n_sorted * 2)) * (__n_sorted * 2), __n);
                        const auto __n1 = ::std::min<_IdType>(__offset + __n_sorted, __n) - __offset;
                        const auto __n2 = ::std::min<_IdType>(__offset + __n1 + __n_sorted, __n) - (__offset + __n1);

                        if (__data_in_temp)
                        {
                            const auto& __rng1 = oneapi::dpl::__ranges::drop_view_simple(__dst, __offset);
                            const auto& __rng2 = oneapi::dpl::__ranges::drop_view_simple(__dst, __offset + __n1);

                            const auto start = __find_start_point(__rng1, __rng2, __i_elem_local, __n1, __n2, __comp);
                            __serial_merge(__rng1, __rng2, __rng/*__rng3*/, start.first, start.second, __i_elem, __chunk, __n1, __n2, __comp);
                        }
                        else
                        {
                            const auto& __rng1 = oneapi::dpl::__ranges::drop_view_simple(__rng, __offset);
                            const auto& __rng2 = oneapi::dpl::__ranges::drop_view_simple(__rng, __offset + __n1);

                            const auto start = __find_start_point(__rng1, __rng2, __i_elem_local, __n1, __n2, __comp);
                            __serial_merge(__rng1, __rng2, __dst/*__rng3*/, start.first, start.second, __i_elem, __chunk, __n1, __n2, __comp);
                        }
                    });
                });
            __n_sorted *= 2;
            __data_in_temp = !__data_in_temp;
        }

        // 3. If the data remained in the temporary buffer then copy it back
        if (__data_in_temp)
        {
            __event1 = __exec.queue().submit([&](sycl::handler& __cgh) {
                __cgh.depends_on(__event1);
                oneapi::dpl::__ranges::__require_access(__cgh, __rng);
                auto __temp_acc = __temp.template get_access<access_mode::read>(__cgh);
                // We cannot use __cgh.copy here because of zip_iterator usage
                __cgh.parallel_for<_CopyBackName...>(sycl::range</*dim=*/1>(__n), [=](sycl::item</*dim=*/1> __item_id) {
                    const _IdType __idx = __item_id.get_linear_id();
                    __rng[__idx] = __temp_acc[__idx];
                });
            });
        }

        return __future(__event1);
    }
};

template <typename _ExecutionPolicy, typename _Range, typename _Compare>
auto
__parallel_sort_impl(oneapi::dpl::__internal::__device_backend_tag, _ExecutionPolicy&& __exec, _Range&& __rng,
                     _Compare __comp)
{
    using _CustomName = oneapi::dpl::__internal::__policy_kernel_name<_ExecutionPolicy>;

    const auto __n = __rng.size();
    if (__n <= std::numeric_limits<::std::uint32_t>::max())
    {
        using _wi_index_type = ::std::uint32_t;
        using _LeafSortKernel =
            oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<__sort_leaf_kernel<_CustomName, _wi_index_type>>;
        using _GlobalSortKernel =
            oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<__sort_global_kernel<_CustomName, _wi_index_type>>;
        using _CopyBackKernel =
            oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<__sort_copy_back_kernel<_CustomName, _wi_index_type>>;
        return __parallel_sort_submitter<_wi_index_type, _LeafSortKernel, _GlobalSortKernel, _CopyBackKernel>()(
            oneapi::dpl::__internal::__device_backend_tag{}, ::std::forward<_ExecutionPolicy>(__exec),
            ::std::forward<_Range>(__rng), __comp);
    }
    else
    {
        using _wi_index_type = ::std::uint64_t;
        using _LeafSortKernel =
            oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<__sort_leaf_kernel<_CustomName, _wi_index_type>>;
        using _GlobalSortKernel =
            oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<__sort_global_kernel<_CustomName, _wi_index_type>>;
        using _CopyBackKernel =
            oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<__sort_copy_back_kernel<_CustomName, _wi_index_type>>;
        return __parallel_sort_submitter<_wi_index_type, _LeafSortKernel, _GlobalSortKernel, _CopyBackKernel>()(
            oneapi::dpl::__internal::__device_backend_tag{}, ::std::forward<_ExecutionPolicy>(__exec),
            ::std::forward<_Range>(__rng), __comp);
    }
}

// Please see the comment for __parallel_for_submitter for optional kernel name explanation
template <typename _GlobalSortName, typename _CopyBackName>
struct __parallel_partial_sort_submitter;

template <typename... _GlobalSortName, typename... _CopyBackName>
struct __parallel_partial_sort_submitter<__internal::__optional_kernel_name<_GlobalSortName...>,
                                         __internal::__optional_kernel_name<_CopyBackName...>>
{
    template <typename _BackendTag, typename _ExecutionPolicy, typename _Range, typename _Merge, typename _Compare>
    auto
    operator()(_BackendTag, _ExecutionPolicy&& __exec, _Range&& __rng, _Merge __merge, _Compare __comp) const
    {
        using _Tp = oneapi::dpl::__internal::__value_t<_Range>;
        using _Size = oneapi::dpl::__internal::__difference_t<_Range>;

        _Size __n = __rng.size();
        assert(__n > 1);

        oneapi::dpl::__par_backend_hetero::__buffer<_ExecutionPolicy, _Tp> __temp_buf(__exec, __n);
        auto __temp = __temp_buf.get_buffer();
        _PRINT_INFO_IN_DEBUG_MODE(__exec);

        _Size __k = 1;
        bool __data_in_temp = false;
        sycl::event __event1;
        do
        {
            __event1 = __exec.queue().submit([&, __data_in_temp, __k](sycl::handler& __cgh) {
                __cgh.depends_on(__event1);
                oneapi::dpl::__ranges::__require_access(__cgh, __rng);
                auto __temp_acc = __temp.template get_access<access_mode::read_write>(__cgh);
                __cgh.parallel_for<_GlobalSortName...>(
                    sycl::range</*dim=*/1>(__n), [=](sycl::item</*dim=*/1> __item_id) {
                        auto __global_idx = __item_id.get_linear_id();

                        _Size __start = 2 * __k * (__global_idx / (2 * __k));
                        _Size __end_1 = sycl::min(__start + __k, __n);
                        _Size __end_2 = sycl::min(__start + 2 * __k, __n);

                        if (!__data_in_temp)
                        {
                            __merge(__global_idx, __rng, __start, __end_1, __rng, __end_1, __end_2, __temp_acc, __start,
                                    __comp);
                        }
                        else
                        {
                            __merge(__global_idx, __temp_acc, __start, __end_1, __temp_acc, __end_1, __end_2, __rng,
                                    __start, __comp);
                        }
                    });
            });
            __data_in_temp = !__data_in_temp;
            __k *= 2;
        } while (__k < __n);

        // if results are in temporary buffer then copy back those
        if (__data_in_temp)
        {
            __event1 = __exec.queue().submit([&](sycl::handler& __cgh) {
                __cgh.depends_on(__event1);
                oneapi::dpl::__ranges::__require_access(__cgh, __rng);
                auto __temp_acc = __temp.template get_access<access_mode::read>(__cgh);
                // we cannot use __cgh.copy here because of zip_iterator usage
                __cgh.parallel_for<_CopyBackName...>(sycl::range</*dim=*/1>(__n), [=](sycl::item</*dim=*/1> __item_id) {
                    __rng[__item_id.get_linear_id()] = __temp_acc[__item_id];
                });
            });
        }
        // return future and extend lifetime of temporary buffer
        return __future(__event1);
    }
};

template <typename _ExecutionPolicy, typename _Range, typename _Merge, typename _Compare>
auto
__parallel_partial_sort_impl(oneapi::dpl::__internal::__device_backend_tag, _ExecutionPolicy&& __exec, _Range&& __rng,
                             _Merge __merge, _Compare __comp)
{
    using _CustomName = oneapi::dpl::__internal::__policy_kernel_name<_ExecutionPolicy>;
    using _GlobalSortKernel =
        oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<__sort_global_kernel<_CustomName>>;
    using _CopyBackKernel =
        oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<__sort_copy_back_kernel<_CustomName>>;

    return __parallel_partial_sort_submitter<_GlobalSortKernel, _CopyBackKernel>()(
        oneapi::dpl::__internal::__device_backend_tag{}, ::std::forward<_ExecutionPolicy>(__exec),
        ::std::forward<_Range>(__rng), __merge, __comp);
}

//------------------------------------------------------------------------
// parallel_stable_sort - async pattern
//-----------------------------------------------------------------------

template <typename _T, typename _Compare>
struct __is_radix_sort_usable_for_type
{
    static constexpr bool value =
#if _USE_RADIX_SORT
        (::std::is_arithmetic_v<_T> || ::std::is_same_v<sycl::half, _T>) &&
            (__internal::__is_comp_ascending<::std::decay_t<_Compare>>::value ||
            __internal::__is_comp_descending<::std::decay_t<_Compare>>::value);
#else
        false;
#endif
};

#if _USE_RADIX_SORT
template <
    typename _ExecutionPolicy, typename _Range, typename _Compare, typename _Proj,
    ::std::enable_if_t<
        __is_radix_sort_usable_for_type<oneapi::dpl::__internal::__key_t<_Proj, _Range>, _Compare>::value, int> = 0>
auto
__parallel_stable_sort(oneapi::dpl::__internal::__device_backend_tag __backend_tag, _ExecutionPolicy&& __exec,
                       _Range&& __rng, _Compare, _Proj __proj)
{
    return __parallel_radix_sort<__internal::__is_comp_ascending<::std::decay_t<_Compare>>::value>(
        __backend_tag, ::std::forward<_ExecutionPolicy>(__exec), ::std::forward<_Range>(__rng), __proj);
}
#endif

template <
    typename _ExecutionPolicy, typename _Range, typename _Compare, typename _Proj,
    ::std::enable_if_t<
        !__is_radix_sort_usable_for_type<oneapi::dpl::__internal::__key_t<_Proj, _Range>, _Compare>::value, int> = 0>
auto
__parallel_stable_sort(oneapi::dpl::__internal::__device_backend_tag __backend_tag, _ExecutionPolicy&& __exec,
                       _Range&& __rng, _Compare __comp, _Proj __proj)
{
    auto __cmp_f = [__comp, __proj](const auto& __a, const auto& __b) mutable {
        return __comp(__proj(__a), __proj(__b));
    };
    return __parallel_sort_impl(__backend_tag, ::std::forward<_ExecutionPolicy>(__exec), ::std::forward<_Range>(__rng),
                                __cmp_f);
}

//------------------------------------------------------------------------
// parallel_partial_sort - async pattern
//-----------------------------------------------------------------------

// TODO: check if it makes sense to move these wrappers out of backend to a common place
// TODO: consider changing __partial_merge_kernel to make it compatible with
//       __full_merge_kernel in order to use __parallel_sort_impl routine
template <typename _ExecutionPolicy, typename _Iterator, typename _Compare>
auto
__parallel_partial_sort(oneapi::dpl::__internal::__device_backend_tag __backend_tag, _ExecutionPolicy&& __exec,
                        _Iterator __first, _Iterator __mid, _Iterator __last, _Compare __comp)
{
    const auto __mid_idx = __mid - __first;

    auto __keep = oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read_write, _Iterator>();
    auto __buf = __keep(__first, __last);

    return __parallel_partial_sort_impl(__backend_tag, ::std::forward<_ExecutionPolicy>(__exec), __buf.all_view(),
                                        __partial_merge_kernel<decltype(__mid_idx)>{__mid_idx}, __comp);
}
} // namespace __par_backend_hetero
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_PARALLEL_BACKEND_SYCL_H
