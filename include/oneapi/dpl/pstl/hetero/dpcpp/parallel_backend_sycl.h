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
template <typename _ExecutionPolicy, typename _Fp, typename _Index,
          oneapi::dpl::__internal::__enable_if_device_execution_policy<_ExecutionPolicy, int> = 0, typename... _Ranges>
auto
__parallel_for(_ExecutionPolicy&& __exec, _Fp __brick, _Index __count, _Ranges&&... __rngs)
{
    using _Policy = typename ::std::decay<_ExecutionPolicy>::type;
    using _CustomName = typename _Policy::kernel_name;
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
                __global_scan_caller<_GlobalScan, typename ::std::decay<_Range2>::type,
                                     typename ::std::decay<_Range1>::type, decltype(__wg_sums_acc), decltype(__n)>(
                    __global_scan, __rng2, __rng1, __wg_sums_acc, __n, __size_per_wg));
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

                    __scan_work_group<_ValueType, _Inclusive>(__group, __lacc.get_pointer(), __lacc.get_pointer() + __n,
                                                              __lacc.get_pointer(), __bin_op, __init);

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
                        _IsFullGroup && dpl::__internal::__range_has_raw_ptr_iterator<::std::decay_t<_InRng>>::value;
#else
                    constexpr bool __can_use_subgroup_load_store = false;
#endif

                    if constexpr (__can_use_subgroup_load_store)
                    {
                        _ONEDPL_PRAGMA_UNROLL
                        for (::std::uint16_t __i = 0; __i < _ElemsPerItem; ++__i)
                        {
                            auto __idx = __i * _WGSize + __subgroup_id * __subgroup_size;
                            auto __val = __unary_op(__subgroup.load(__in_rng.begin() + __idx));
                            __subgroup.store(__lacc.get_pointer() + __idx, __val);
                        }
                    }
                    else
                    {
                        _ONEDPL_PRAGMA_UNROLL
                        for (::std::uint16_t __idx = __item_id; __idx < __n; __idx += _WGSize)
                        {
                            __lacc[__idx] = __unary_op(__in_rng[__idx]);
                        }
                    }

                    __scan_work_group<_ValueType, _Inclusive>(__group, __lacc.get_pointer(), __lacc.get_pointer() + __n,
                                                              __lacc.get_pointer(), __bin_op, __init);

                    if constexpr (__can_use_subgroup_load_store)
                    {
                        _ONEDPL_PRAGMA_UNROLL
                        for (::std::uint16_t __i = 0; __i < _ElemsPerItem; ++__i)
                        {
                            auto __idx = __i * _WGSize + __subgroup_id * __subgroup_size;
                            auto __val = __subgroup.load(__lacc.get_pointer() + __idx);
                            __subgroup.store(__out_rng.begin() + __idx, __val);
                        }
                    }
                    else
                    {
                        _ONEDPL_PRAGMA_UNROLL
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
                        _IsFullGroup && dpl::__internal::__range_has_raw_ptr_iterator<::std::decay_t<_InRng>>::value;
#else
                    constexpr bool __can_use_subgroup_load_store = false;
#endif

                    if constexpr (__can_use_subgroup_load_store)
                    {
                        _ONEDPL_PRAGMA_UNROLL
                        for (::std::uint16_t __i = 0; __i < _ElemsPerItem; ++__i)
                        {
                            auto __idx = __i * _WGSize + __subgroup_id * __subgroup_size;
                            uint16_t __val = __unary_op(__subgroup.load(__in_rng.begin() + __idx));
                            __subgroup.store(__lacc.get_pointer() + __idx, __val);
                        }
                    }
                    else
                    {
                        _ONEDPL_PRAGMA_UNROLL
                        for (::std::uint16_t __idx = __item_id; __idx < __n; __idx += _WGSize)
                        {
                            __lacc[__idx] = __unary_op(__in_rng[__idx]);
                        }
                    }

                    __scan_work_group<_ValueType, /* _Inclusive */ false>(
                        __group, __lacc.get_pointer(), __lacc.get_pointer() + __elems_per_wg,
                        __lacc.get_pointer() + __elems_per_wg, __bin_op, __init);

                    _ONEDPL_PRAGMA_UNROLL
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
__parallel_transform_scan_single_group(_ExecutionPolicy&& __exec, _InRng&& __in_rng, _OutRng&& __out_rng,
                                       ::std::size_t __n, _UnaryOperation __unary_op, _InitType __init,
                                       _BinaryOperation __binary_op, _Inclusive)
{
    using _CustomName = typename std::decay_t<_ExecutionPolicy>::kernel_name;

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
                    oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
                        __scan_single_wg_kernel<::std::integral_constant<::std::uint16_t, __wg_size>,
                                                ::std::integral_constant<::std::uint16_t, __num_elems_per_item>,
                                                /* _IsFullGroup= */ std::true_type, _Inclusive, _CustomName>>>()(
                    ::std::forward<_ExecutionPolicy>(__exec), std::forward<_InRng>(__in_rng),
                    std::forward<_OutRng>(__out_rng), __n, __init, __binary_op, __unary_op);
            else
                return __parallel_transform_scan_static_single_group_submitter<
                    _Inclusive::value, __num_elems_per_item, __wg_size,
                    /* _IsFullGroup= */ false,
                    oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
                        __scan_single_wg_kernel<::std::integral_constant<::std::uint16_t, __wg_size>,
                                                ::std::integral_constant<::std::uint16_t, __num_elems_per_item>,
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
            __par_backend_hetero::__scan_single_wg_dynamic_kernel<_CustomName>>;

        return __parallel_transform_scan_dynamic_single_group_submitter<_Inclusive::value, _DynamicGroupScanKernel>()(
            ::std::forward<_ExecutionPolicy>(__exec), std::forward<_InRng>(__in_rng), std::forward<_OutRng>(__out_rng),
            __n, __init, __binary_op, __unary_op, __max_wg_size);
    }
}

template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _BinaryOperation, typename _InitType,
          typename _LocalScan, typename _GroupScan, typename _GlobalScan,
          oneapi::dpl::__internal::__enable_if_device_execution_policy<_ExecutionPolicy, int> = 0>
auto
__parallel_transform_scan_base(_ExecutionPolicy&& __exec, _Range1&& __in_rng, _Range2&& __out_rng,
                               _BinaryOperation __binary_op, _InitType __init, _LocalScan __local_scan,
                               _GroupScan __group_scan, _GlobalScan __global_scan)
{
    using _Policy = typename ::std::decay<_ExecutionPolicy>::type;
    using _CustomName = typename _Policy::kernel_name;

    using _PropagateKernel =
        oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<__scan_propagate_kernel<_CustomName>>;

    return __parallel_scan_submitter<_CustomName, _PropagateKernel>()(
        ::std::forward<_ExecutionPolicy>(__exec), ::std::forward<_Range1>(__in_rng), ::std::forward<_Range2>(__out_rng),
        __binary_op, __init, __local_scan, __group_scan, __global_scan);
}

template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _UnaryOperation, typename _InitType,
          typename _BinaryOperation, typename _Inclusive,
          oneapi::dpl::__internal::__enable_if_device_execution_policy<_ExecutionPolicy, int> = 0>
auto
__parallel_transform_scan(_ExecutionPolicy&& __exec, _Range1&& __in_rng, _Range2&& __out_rng, ::std::size_t __n,
                          _UnaryOperation __unary_op, _InitType __init, _BinaryOperation __binary_op, _Inclusive)
{
    using _Type = typename _InitType::__value_type;

    // Next power of 2 greater than or equal to __n
    auto __n_uniform = __n;
    if ((__n_uniform & (__n_uniform - 1)) != 0)
        __n_uniform = oneapi::dpl::__internal::__dpl_bit_floor(__n) << 1;

    // Pessimistically only use half of the memory to take into account memory used by compiled kernel
    const ::std::size_t __max_slm_size =
        __exec.queue().get_device().template get_info<sycl::info::device::local_mem_size>() / 2;
    const auto __req_slm_size = sizeof(_Type) * __n_uniform;

    constexpr int __single_group_upper_limit = 16384;

    constexpr bool __can_use_group_scan = unseq_backend::__has_known_identity<_BinaryOperation, _Type>::value;
    if constexpr (__can_use_group_scan)
    {
        if (__n <= __single_group_upper_limit && __max_slm_size >= __req_slm_size)
        {
            return __parallel_transform_scan_single_group(
                std::forward<_ExecutionPolicy>(__exec), ::std::forward<_Range1>(__in_rng),
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
            ::std::forward<_ExecutionPolicy>(__exec), ::std::forward<_Range1>(__in_rng),
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

        using _CustomName = typename ::std::decay_t<_ExecutionPolicy>::kernel_name;
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
          typename _CopyByMaskOp,
          oneapi::dpl::__internal::__enable_if_device_execution_policy<_ExecutionPolicy, int> = 0>
auto
__parallel_scan_copy(_ExecutionPolicy&& __exec, _InRng&& __in_rng, _OutRng&& __out_rng, _Size __n,
                     _CreateMaskOp __create_mask_op, _CopyByMaskOp __copy_by_mask_op)
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
    oneapi::dpl::__par_backend_hetero::__internal::__buffer<_ExecutionPolicy, int32_t> __mask_buf(__exec, __n);

    return __parallel_transform_scan_base(
        ::std::forward<_ExecutionPolicy>(__exec),
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

template <typename _ExecutionPolicy, typename _InRng, typename _OutRng, typename _Size, typename _Pred,
          oneapi::dpl::__internal::__enable_if_device_execution_policy<_ExecutionPolicy, int> = 0>
auto
__parallel_copy_if(_ExecutionPolicy&& __exec, _InRng&& __in_rng, _OutRng&& __out_rng, _Size __n, _Pred __pred)
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

        return __parallel_scan_copy(::std::forward<_ExecutionPolicy>(__exec), ::std::forward<_InRng>(__in_rng),
                                    ::std::forward<_OutRng>(__out_rng), __n, CreateOp{__pred}, CopyOp{});
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
            using _ShiftedIdxType =
                typename ::std::conditional<_OrTagType::value, decltype(__init_index + __i * __shift),
                                            decltype(__found_local.load())>::type;

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
oneapi::dpl::__internal::__enable_if_device_execution_policy<
    _ExecutionPolicy,
    typename ::std::conditional<::std::is_same<_BrickTag, __parallel_or_tag>::value, bool,
                                oneapi::dpl::__internal::__difference_t<
                                    typename oneapi::dpl::__ranges::__get_first_range_type<_Ranges...>::type>>::type>
__parallel_find_or(_ExecutionPolicy&& __exec, _Brick __f, _BrickTag __brick_tag, _Ranges&&... __rngs)
{
    using _Policy = typename ::std::decay<_ExecutionPolicy>::type;
    using _CustomName = typename _Policy::kernel_name;
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
                        *__temp_acc.get_pointer());
                    __dpl_sycl::__atomic_ref<_AtomicType, sycl::access::address_space::local_space> __found_local(
                        *__temp_local.get_pointer());

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
oneapi::dpl::__internal::__enable_if_device_execution_policy<_ExecutionPolicy, bool>
__parallel_or(_ExecutionPolicy&& __exec, _Iterator1 __first, _Iterator1 __last, _Iterator2 __s_first,
              _Iterator2 __s_last, _Brick __f)
{
    auto __keep = oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read, _Iterator1>();
    auto __buf = __keep(__first, __last);
    auto __s_keep = oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read, _Iterator2>();
    auto __s_buf = __s_keep(__s_first, __s_last);

    return oneapi::dpl::__par_backend_hetero::__parallel_find_or(
        __par_backend_hetero::make_wrapped_policy<__or_policy_wrapper>(::std::forward<_ExecutionPolicy>(__exec)), __f,
        __parallel_or_tag{}, __buf.all_view(), __s_buf.all_view());
}

// Special overload for single sequence cases.
// TODO: check if similar pattern may apply to other algorithms. If so, these overloads should be moved out of
// backend code.
template <typename _ExecutionPolicy, typename _Iterator, typename _Brick>
oneapi::dpl::__internal::__enable_if_device_execution_policy<_ExecutionPolicy, bool>
__parallel_or(_ExecutionPolicy&& __exec, _Iterator __first, _Iterator __last, _Brick __f)
{
    auto __keep = oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read, _Iterator>();
    auto __buf = __keep(__first, __last);

    return oneapi::dpl::__par_backend_hetero::__parallel_find_or(
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
oneapi::dpl::__internal::__enable_if_device_execution_policy<_ExecutionPolicy, _Iterator1>
__parallel_find(_ExecutionPolicy&& __exec, _Iterator1 __first, _Iterator1 __last, _Iterator2 __s_first,
                _Iterator2 __s_last, _Brick __f, _IsFirst)
{
    auto __keep = oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read, _Iterator1>();
    auto __buf = __keep(__first, __last);
    auto __s_keep = oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read, _Iterator2>();
    auto __s_buf = __s_keep(__s_first, __s_last);

    using _TagType =
        typename ::std::conditional<_IsFirst::value, __parallel_find_forward_tag<decltype(__buf.all_view())>,
                                    __parallel_find_backward_tag<decltype(__buf.all_view())>>::type;
    return __first + oneapi::dpl::__par_backend_hetero::__parallel_find_or(
                         __par_backend_hetero::make_wrapped_policy<__find_policy_wrapper>(
                             ::std::forward<_ExecutionPolicy>(__exec)),
                         __f, _TagType{}, __buf.all_view(), __s_buf.all_view());
}

// Special overload for single sequence cases.
// TODO: check if similar pattern may apply to other algorithms. If so, these overloads should be moved out of
// backend code.
template <typename _ExecutionPolicy, typename _Iterator, typename _Brick, typename _IsFirst>
oneapi::dpl::__internal::__enable_if_device_execution_policy<_ExecutionPolicy, _Iterator>
__parallel_find(_ExecutionPolicy&& __exec, _Iterator __first, _Iterator __last, _Brick __f, _IsFirst)
{
    auto __keep = oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read, _Iterator>();
    auto __buf = __keep(__first, __last);

    using _TagType =
        typename ::std::conditional<_IsFirst::value, __parallel_find_forward_tag<decltype(__buf.all_view())>,
                                    __parallel_find_backward_tag<decltype(__buf.all_view())>>::type;
    return __first + oneapi::dpl::__par_backend_hetero::__parallel_find_or(
                         __par_backend_hetero::make_wrapped_policy<__find_policy_wrapper>(
                             ::std::forward<_ExecutionPolicy>(__exec)),
                         __f, _TagType{}, __buf.all_view());
}

//------------------------------------------------------------------------
// parallel_merge - async pattern
//-----------------------------------------------------------------------
struct __full_merge_kernel
{
    // this function is needed because it calls in different parallel patterns (parallel_merge, parallel_sort)
    // and replacing with this function may affect performance for them.
    template <typename _Idx, typename _Acc1, typename _Size1, typename _Acc2, typename _Size2, typename _Acc3,
              typename _Size3, typename _Compare>
    void
    operator()(const _Idx __global_idx, const _Acc1& __in_acc1, const _Size1 __start_1, const _Size1 __end_1,
               const _Acc2& __in_acc2, const _Size2 __start_2, const _Size2 __end_2, const _Acc3& __out_acc,
               const _Size3 __out_shift, _Compare __comp, const ::std::size_t __chunk) const
    {
        // Borders of the sequences to merge within this call
        const _Size1 __local_start_1 = sycl::min(static_cast<_Size1>(__global_idx + __start_1), __end_1);
        const _Size1 __local_end_1 = sycl::min(static_cast<_Size1>(__local_start_1 + __chunk), __end_1);
        const _Size2 __local_start_2 = sycl::min(static_cast<_Size2>(__global_idx + __start_2), __end_2);
        const _Size2 __local_end_2 = sycl::min(static_cast<_Size2>(__local_start_2 + __chunk), __end_2);
        // Borders of the sequences to search an offset
        _Size1 __l_search_bound_1{};
        _Size1 __r_search_bound_1{};
        _Size2 __l_search_bound_2{};
        _Size2 __r_search_bound_2{};

        const _Size1 __local_size_1 = __local_end_1 - __local_start_1;
        const _Size2 __local_size_2 = __local_end_2 - __local_start_2;

        const auto __r_item_1 = __in_acc1[__end_1 - 1];
        const auto __l_item_2 = __in_acc2[__start_2];

        // Copy if the sequences are sorted respect to each other or merge otherwise
        if (!__comp(__l_item_2, __r_item_1))
        {
            const _Size1 __out_shift_1 = __out_shift + __local_start_1 - __start_1;
            const _Size2 __out_shift_2 = __out_shift + __end_1 - __start_1 + __local_start_2 - __start_2;
            // TODO: check performance impact via a profiler: for vs memcpy
            for (_Idx __i = 0; __i < __local_size_1; ++__i)
            {
                __out_acc[__out_shift_1 + __i] = __in_acc1[__local_start_1 + __i];
            }
            for (_Idx __i = 0; __i < __local_size_2; ++__i)
            {
                __out_acc[__out_shift_2 + __i] = __in_acc2[__local_start_2 + __i];
            }
        }
        else if (__comp(__r_item_1, __l_item_2))
        {
            const _Size1 __out_shift_1 = __out_shift + __end_2 - __start_2 + __local_start_1 - __start_1;
            const _Size2 __out_shift_2 = __out_shift + __local_start_2 - __start_2;
            for (_Idx __i = 0; __i < __local_size_1; ++__i)
            {
                __out_acc[__out_shift_1 + __i] = __in_acc1[__local_start_1 + __i];
            }
            for (_Idx __i = 0; __i < __local_size_2; ++__i)
            {
                __out_acc[__out_shift_2 + __i] = __in_acc2[__local_start_2 + __i];
            }
        }
        // Perform merging
        else
        {
            // Process 1st sequence
            if (__local_start_1 < __local_end_1)
            {
                // Reduce the range for searching within the 2nd sequence and handle bound items
                const auto __local_l_item_1 = __in_acc1[__local_start_1];
                __l_search_bound_2 = oneapi::dpl::__internal::__pstl_lower_bound(__in_acc2, __start_2, __end_2,
                                                                                 __local_l_item_1, __comp);
                const auto __l_shift_1 = __local_start_1 - __start_1;
                const auto __l_shift_2 = __l_search_bound_2 - __start_2;
                __out_acc[__out_shift + __l_shift_1 + __l_shift_2] = __local_l_item_1;
                if (__local_end_1 - __local_start_1 > 1)
                {
                    const auto __local_r_item_1 = __in_acc1[__local_end_1 - 1];
                    __r_search_bound_2 = oneapi::dpl::__internal::__pstl_lower_bound(__in_acc2, __l_search_bound_2,
                                                                                     __end_2, __local_r_item_1, __comp);
                    const auto __r_shift_1 = __local_end_1 - 1 - __start_1;
                    const auto __r_shift_2 = __r_search_bound_2 - __start_2;
                    __out_acc[__out_shift + __r_shift_1 + __r_shift_2] = __local_r_item_1;
                }

                // Handle intermediate items
                for (auto __idx = __local_start_1 + 1; __idx < __local_end_1 - 1; ++__idx)
                {
                    const auto __intermediate_item_1 = __in_acc1[__idx];
                    __l_search_bound_2 = oneapi::dpl::__internal::__pstl_lower_bound(
                        __in_acc2, __l_search_bound_2, __r_search_bound_2, __intermediate_item_1, __comp);
                    const auto __shift_1 = __idx - __start_1;
                    const auto __shift_2 = __l_search_bound_2 - __start_2;
                    __out_acc[__out_shift + __shift_1 + __shift_2] = __intermediate_item_1;
                }
            }
            // Process 2nd sequence
            if (__local_start_2 < __local_end_2)
            {
                // Reduce the range for searching within the 1st sequence and handle bound items
                const auto __local_l_item_2 = __in_acc2[__local_start_2];
                __l_search_bound_1 = oneapi::dpl::__internal::__pstl_upper_bound(__in_acc1, __start_1, __end_1,
                                                                                 __local_l_item_2, __comp);
                const auto __l_shift_1 = __l_search_bound_1 - __start_1;
                const auto __l_shift_2 = __local_start_2 - __start_2;
                __out_acc[__out_shift + __l_shift_1 + __l_shift_2] = __local_l_item_2;
                if (__local_end_2 - __local_start_2 > 1)
                {
                    const auto __local_r_item_2 = __in_acc2[__local_end_2 - 1];
                    __r_search_bound_1 = oneapi::dpl::__internal::__pstl_upper_bound(__in_acc1, __l_search_bound_1,
                                                                                     __end_1, __local_r_item_2, __comp);
                    const auto __r_shift_1 = __r_search_bound_1 - __start_1;
                    const auto __r_shift_2 = __local_end_2 - 1 - __start_2;
                    __out_acc[__out_shift + __r_shift_1 + __r_shift_2] = __local_r_item_2;
                }

                // Handle intermediate items
                for (auto __idx = __local_start_2 + 1; __idx < __local_end_2 - 1; ++__idx)
                {
                    const auto __intermediate_item_2 = __in_acc2[__idx];
                    __l_search_bound_1 = oneapi::dpl::__internal::__pstl_upper_bound(
                        __in_acc1, __l_search_bound_1, __r_search_bound_1, __intermediate_item_2, __comp);
                    const auto __shift_1 = __l_search_bound_1 - __start_1;
                    const auto __shift_2 = __idx - __start_2;
                    __out_acc[__out_shift + __shift_1 + __shift_2] = __intermediate_item_2;
                }
            }
        }
    }
};

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

// Please see the comment for __parallel_for_submitter for optional kernel name explanation
template <typename _Name>
struct __parallel_merge_submitter;

template <typename... _Name>
struct __parallel_merge_submitter<__internal::__optional_kernel_name<_Name...>>
{
    template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _Range3, typename _Compare>
    auto
    operator()(_ExecutionPolicy&& __exec, _Range1&& __rng1, _Range2&& __rng2, _Range3&& __rng3, _Compare __comp) const
    {
        auto __n = __rng1.size();
        auto __n_2 = __rng2.size();

        assert(__n > 0 || __n_2 > 0);

        _PRINT_INFO_IN_DEBUG_MODE(__exec);

        const ::std::size_t __chunk = __exec.queue().get_device().is_cpu() ? 128 : 8;
        const auto __max_n = ::std::max(__n, static_cast<decltype(__n)>(__n_2));
        const ::std::size_t __steps = ((__max_n - 1) / __chunk) + 1;

        auto __event = __exec.queue().submit([&](sycl::handler& __cgh) {
            oneapi::dpl::__ranges::__require_access(__cgh, __rng1, __rng2, __rng3);
            __cgh.parallel_for<_Name...>(sycl::range</*dim=*/1>(__steps), [=](sycl::item</*dim=*/1> __item_id) {
                __full_merge_kernel()(__item_id.get_linear_id() * __chunk, __rng1, decltype(__n)(0), __n, __rng2,
                                      decltype(__n_2)(0), __n_2, __rng3, decltype(__n)(0), __comp, __chunk);
            });
        });
        return __future(__event);
    }
};

template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _Range3, typename _Compare,
          oneapi::dpl::__internal::__enable_if_device_execution_policy<_ExecutionPolicy, int> = 0>
auto
__parallel_merge(_ExecutionPolicy&& __exec, _Range1&& __rng1, _Range2&& __rng2, _Range3&& __rng3, _Compare __comp)
{
    using _Policy = typename ::std::decay<_ExecutionPolicy>::type;
    using _CustomName = typename _Policy::kernel_name;
    using _MergeKernel = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<_CustomName>;

    return __parallel_merge_submitter<_MergeKernel>()(::std::forward<_ExecutionPolicy>(__exec),
                                                      ::std::forward<_Range1>(__rng1), ::std::forward<_Range2>(__rng2),
                                                      ::std::forward<_Range3>(__rng3), __comp);
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
template <typename _LeafSortName, typename _GlobalSortName, typename _CopyBackName>
struct __parallel_sort_submitter;

template <typename... _LeafSortName, typename... _GlobalSortName, typename... _CopyBackName>
struct __parallel_sort_submitter<__internal::__optional_kernel_name<_LeafSortName...>,
                                 __internal::__optional_kernel_name<_GlobalSortName...>,
                                 __internal::__optional_kernel_name<_CopyBackName...>>
{
    template <typename _ExecutionPolicy, typename _Range, typename _Merge, typename _Compare>
    auto
    operator()(_ExecutionPolicy&& __exec, _Range&& __rng, _Merge __merge, _Compare __comp) const
    {
        using _Policy = typename ::std::decay<_ExecutionPolicy>::type;
        using _Tp = oneapi::dpl::__internal::__value_t<_Range>;
        using _Size = oneapi::dpl::__internal::__difference_t<_Range>;

        _Size __n = __rng.size();
        assert(__n > 1);

        _PRINT_INFO_IN_DEBUG_MODE(__exec);

        // __leaf: size of a block to sort using algorithm suitable for small sequences
        // __optimal_chunk: best size of a block to merge duiring a step of the merge sort algorithm
        // The coefficients were found experimentally
        _Size __leaf = 4;
        _Size __optimal_chunk = 4;
        if (__exec.queue().get_device().is_cpu())
        {
            __leaf = 16;
            __optimal_chunk = 32;
        }
        // Assume powers of 2
        assert((__leaf & (__leaf - 1)) == 0);
        assert((__optimal_chunk & (__optimal_chunk - 1)) == 0);

        const _Size __leaf_steps = ((__n - 1) / __leaf) + 1;

        // 1. Perform sorting of the leaves of the merge sort tree
        sycl::event __event1 = __exec.queue().submit([&](sycl::handler& __cgh) {
            oneapi::dpl::__ranges::__require_access(__cgh, __rng);
            __cgh.parallel_for<_LeafSortName...>(sycl::range</*dim=*/1>(__leaf_steps),
                                                 [=](sycl::item</*dim=*/1> __item_id) {
                                                     const _Size __idx = __item_id.get_linear_id() * __leaf;
                                                     const _Size __start = __idx;
                                                     const _Size __end = sycl::min(__start + __leaf, __n);
                                                     __leaf_sort_kernel()(__rng, __start, __end, __comp);
                                                 });
        });

        _Size __sorted = __leaf;
        // Chunk size cannot be bigger than size of a sorted sequence
        _Size __chunk = ::std::min(__leaf, __optimal_chunk);

        oneapi::dpl::__par_backend_hetero::__internal::__buffer<_Policy, _Tp> __temp_buf(__exec, __n);
        auto __temp = __temp_buf.get_buffer();
        bool __data_in_temp = false;

        // 2. Perform merge sorting
        // TODO: try to presort sequences with the same approach using local memory
        while (__sorted < __n)
        {
            // Number of steps is a number of work items required during a single merge sort stage.
            // Each work item handles a pair of chunks:
            // one chunk from the first sorted sequence and one chunk from the second sorted sequence.
            // Both chunks are placed with the same offset regarding the beginning of a sorted sequence.
            // Consider the following example:
            //  * Sequence: 0 1 2 3 1 2 3 4 2 3 4 5 3 4 5 6 4 5 6 7 5
            //  * Size of a sorted sequence: 4
            //  * Size of a chunk: 2
            //  Work item id and chunks it handles:   0     1     0     1      2     3     2     3      4     5    4
            //  Sequence:                          [ 0 1 | 2 3 @ 1 2 | 3 4 ][ 2 3 | 4 5 @ 3 4 | 5 6 ][ 4 5 | 6 7 @ 5 ]
            //  Legend:
            //  * [] - border between pairs of sorted sequences which are to be merged
            //  * @  - border between each sorted sequence in a pair
            //  * || - border between chunks

            _Size __sorted_pair = 2 * __sorted;
            _Size __chunks_in_sorted = __sorted / __chunk;
            _Size __full_pairs = __n / __sorted_pair;
            _Size __incomplete_pair = __n - __sorted_pair * __full_pairs;
            _Size __first_block_in_incomplete_pair = __incomplete_pair > __sorted ? __sorted : __incomplete_pair;
            _Size __incomplete_last_chunk = __first_block_in_incomplete_pair % __chunk != 0;
            _Size __incomplete_pair_steps = __first_block_in_incomplete_pair / __chunk + __incomplete_last_chunk;
            _Size __full_pairs_steps = __full_pairs * __chunks_in_sorted;
            _Size __steps = __full_pairs_steps + __incomplete_pair_steps;

            __event1 = __exec.queue().submit([&, __data_in_temp, __sorted, __sorted_pair, __chunk, __chunks_in_sorted,
                                              __steps](sycl::handler& __cgh) {
                __cgh.depends_on(__event1);
                oneapi::dpl::__ranges::__require_access(__cgh, __rng);
                auto __temp_acc = __temp.template get_access<__par_backend_hetero::access_mode::read_write>(__cgh);
                __cgh.parallel_for<_GlobalSortName...>(
                    sycl::range</*dim=*/1>(__steps), [=](sycl::item</*dim=*/1> __item_id) {
                        const _Size __idx = __item_id.get_linear_id();
                        // Borders of the first and the second sorted sequences
                        const _Size __start_1 = sycl::min(__sorted_pair * ((__idx * __chunk) / __sorted), __n);
                        const _Size __end_1 = sycl::min(__start_1 + __sorted, __n);
                        const _Size __start_2 = __end_1;
                        const _Size __end_2 = sycl::min(__start_2 + __sorted, __n);

                        // Distance between the beginning of a sorted sequence and the beginning of a chunk
                        const _Size __offset = __chunk * (__idx % __chunks_in_sorted);

                        if (!__data_in_temp)
                        {
                            __merge(__offset, __rng, __start_1, __end_1, __rng, __start_2, __end_2, __temp_acc,
                                    __start_1, __comp, __chunk);
                        }
                        else
                        {
                            __merge(__offset, __temp_acc, __start_1, __end_1, __temp_acc, __start_2, __end_2, __rng,
                                    __start_1, __comp, __chunk);
                        }
                    });
            });
            __data_in_temp = !__data_in_temp;
            __sorted = __sorted_pair;
            if (__chunk < __optimal_chunk)
                __chunk *= 2;
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
                    __rng[__item_id.get_linear_id()] = __temp_acc[__item_id];
                });
            });
        }

        return __future(__event1, __temp);
    }
};

template <typename _ExecutionPolicy, typename _Range, typename _Merge, typename _Compare,
          oneapi::dpl::__internal::__enable_if_device_execution_policy<_ExecutionPolicy, int> = 0>
auto
__parallel_sort_impl(_ExecutionPolicy&& __exec, _Range&& __rng, _Merge __merge, _Compare __comp)
{
    using _Policy = typename ::std::decay<_ExecutionPolicy>::type;
    using _CustomName = typename _Policy::kernel_name;
    using _LeafSortKernel =
        oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<__sort_leaf_kernel<_CustomName>>;
    using _GlobalSortKernel =
        oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<__sort_global_kernel<_CustomName>>;
    using _CopyBackKernel =
        oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<__sort_copy_back_kernel<_CustomName>>;

    return __parallel_sort_submitter<_LeafSortKernel, _GlobalSortKernel, _CopyBackKernel>()(
        ::std::forward<_ExecutionPolicy>(__exec), ::std::forward<_Range>(__rng), __merge, __comp);
}

// Please see the comment for __parallel_for_submitter for optional kernel name explanation
template <typename _GlobalSortName, typename _CopyBackName>
struct __parallel_partial_sort_submitter;

template <typename... _GlobalSortName, typename... _CopyBackName>
struct __parallel_partial_sort_submitter<__internal::__optional_kernel_name<_GlobalSortName...>,
                                         __internal::__optional_kernel_name<_CopyBackName...>>
{
    template <typename _ExecutionPolicy, typename _Range, typename _Merge, typename _Compare>
    auto
    operator()(_ExecutionPolicy&& __exec, _Range&& __rng, _Merge __merge, _Compare __comp) const
    {
        using _Policy = typename ::std::decay<_ExecutionPolicy>::type;
        using _Tp = oneapi::dpl::__internal::__value_t<_Range>;
        using _Size = oneapi::dpl::__internal::__difference_t<_Range>;

        _Size __n = __rng.size();
        assert(__n > 1);

        oneapi::dpl::__par_backend_hetero::__internal::__buffer<_Policy, _Tp> __temp_buf(__exec, __n);
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
        return __future(__event1, __temp);
    }
};

template <typename _ExecutionPolicy, typename _Range, typename _Merge, typename _Compare,
          oneapi::dpl::__internal::__enable_if_device_execution_policy<_ExecutionPolicy, int> = 0>
auto
__parallel_partial_sort_impl(_ExecutionPolicy&& __exec, _Range&& __rng, _Merge __merge, _Compare __comp)
{
    using _Policy = typename ::std::decay<_ExecutionPolicy>::type;
    using _CustomName = typename _Policy::kernel_name;
    using _GlobalSortKernel =
        oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<__sort_global_kernel<_CustomName>>;
    using _CopyBackKernel =
        oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<__sort_copy_back_kernel<_CustomName>>;

    return __parallel_partial_sort_submitter<_GlobalSortKernel, _CopyBackKernel>()(
        ::std::forward<_ExecutionPolicy>(__exec), ::std::forward<_Range>(__rng), __merge, __comp);
}

//------------------------------------------------------------------------
// parallel_stable_sort - async pattern
//-----------------------------------------------------------------------

template <typename _T, typename _Compare>
struct __is_radix_sort_usable_for_type
{
    static constexpr bool value =
#if _USE_RADIX_SORT
        ::std::is_arithmetic<_T>::value && (__internal::__is_comp_ascending<__decay_t<_Compare>>::value ||
                                            __internal::__is_comp_descending<__decay_t<_Compare>>::value);
#else
        false;
#endif
};

#if _USE_RADIX_SORT
template <typename _ExecutionPolicy, typename _Range, typename _Compare, typename _Proj,
    __enable_if_t<oneapi::dpl::__internal::__is_device_execution_policy<__decay_t<_ExecutionPolicy>>::value &&
    __is_radix_sort_usable_for_type<oneapi::dpl::__internal::__key_t<_Proj, _Range>, _Compare>::value, int> = 0>
auto
__parallel_stable_sort(_ExecutionPolicy&& __exec, _Range&& __rng, _Compare, _Proj __proj)
{
    return __parallel_radix_sort<__internal::__is_comp_ascending<__decay_t<_Compare>>::value>(
        ::std::forward<_ExecutionPolicy>(__exec), ::std::forward<_Range>(__rng), __proj);
}
#endif

template <
    typename _ExecutionPolicy, typename _Range, typename _Compare, typename _Proj,
    __enable_if_t<oneapi::dpl::__internal::__is_device_execution_policy<__decay_t<_ExecutionPolicy>>::value &&
    !__is_radix_sort_usable_for_type<oneapi::dpl::__internal::__key_t<_Proj, _Range>, _Compare>::value, int> = 0>
auto
__parallel_stable_sort(_ExecutionPolicy&& __exec, _Range&& __rng, _Compare __comp, _Proj __proj)
{
    auto __cmp_f = [__comp, __proj](const auto& __a, const auto& __b) mutable {
        return __comp(__proj(__a), __proj(__b));
    };
    return __parallel_sort_impl(::std::forward<_ExecutionPolicy>(__exec), ::std::forward<_Range>(__rng),
                                // Pass special tag to choose 'full' merge subroutine at compile-time
                                __full_merge_kernel(), __cmp_f);
}

//------------------------------------------------------------------------
// parallel_partial_sort - async pattern
//-----------------------------------------------------------------------

// TODO: check if it makes sense to move these wrappers out of backend to a common place
// TODO: consider changing __partial_merge_kernel to make it compatible with
//       __full_merge_kernel in order to use __parallel_sort_impl routine
template <typename _ExecutionPolicy, typename _Iterator, typename _Compare,
          oneapi::dpl::__internal::__enable_if_device_execution_policy<_ExecutionPolicy, int> = 0>
auto
__parallel_partial_sort(_ExecutionPolicy&& __exec, _Iterator __first, _Iterator __mid, _Iterator __last,
                        _Compare __comp)
{
    const auto __mid_idx = __mid - __first;

    auto __keep = oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read_write, _Iterator>();
    auto __buf = __keep(__first, __last);

    return __parallel_partial_sort_impl(::std::forward<_ExecutionPolicy>(__exec), __buf.all_view(),
                                        __partial_merge_kernel<decltype(__mid_idx)>{__mid_idx}, __comp);
}

} // namespace __par_backend_hetero
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_PARALLEL_BACKEND_SYCL_H
