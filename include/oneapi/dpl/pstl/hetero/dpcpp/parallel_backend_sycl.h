// -*- C++ -*-
//===-- parallel_backend_sycl.h -------------------------------------------===//
//
// Copyright (C) 2019-2020 Intel Corporation
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
#ifndef _ONEDPL_parallel_backend_sycl_H
#define _ONEDPL_parallel_backend_sycl_H

#include <CL/sycl.hpp>

#include <cassert>
#include <algorithm>
#include <type_traits>
#include <iostream>

#include "../../iterator_impl.h"
#include "../../execution_impl.h"
#include "../../utils_ranges.h"

#include "parallel_backend_sycl_utils.h"
#include "execution_sycl_defs.h"
#include "sycl_iterator.h"
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
        // it doesnt have any iter mode because of two factors:
        //   - since it is a raw pointer, kernel can read/write despite of access_mode
        //   - access_mode also serves for implicit syncronization for buffers to build graph dependency
        //     and since usm have only explicit syncronization and does not provide dependency resolution mechanism
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

// function is needed to wrap kernel name into another class
template <template <typename> class _NewKernelName, typename _Policy,
          oneapi::dpl::__internal::__enable_if_device_execution_policy<_Policy, int> = 0>
auto
make_wrapped_policy(_Policy&& __policy)
    -> decltype(oneapi::dpl::execution::make_device_policy<_NewKernelName<typename __decay_t<_Policy>::kernel_name>>(
        ::std::forward<_Policy>(__policy)))
{
    return oneapi::dpl::execution::make_device_policy<_NewKernelName<typename __decay_t<_Policy>::kernel_name>>(
        ::std::forward<_Policy>(__policy));
}

#if _ONEDPL_FPGA_DEVICE
template <template <typename> class _NewKernelName, typename _Policy,
          oneapi::dpl::__internal::__enable_if_fpga_execution_policy<_Policy, int> = 0>
auto
make_wrapped_policy(_Policy&& __policy)
    -> decltype(oneapi::dpl::execution::make_fpga_policy<__decay_t<_Policy>::unroll_factor,
                                                         _NewKernelName<typename __decay_t<_Policy>::kernel_name>>(
        ::std::forward<_Policy>(__policy)))
{
    return oneapi::dpl::execution::make_fpga_policy<__decay_t<_Policy>::unroll_factor,
                                                    _NewKernelName<typename __decay_t<_Policy>::kernel_name>>(
        ::std::forward<_Policy>(__policy));
}
#endif

// set of templated classes to name kernels
template <typename _DerivedKernelName>
class __kernel_name_base
{
  public:
    template <typename _Exec>
    static sycl::kernel
    __compile_kernel(_Exec&& __exec)
    {
        sycl::program __program(__exec.queue().get_context());

        __program.build_with_kernel_type<_DerivedKernelName>();
        return __program.get_kernel<_DerivedKernelName>();
    }
};

template <typename... _Name>
class __parallel_for_kernel : public __kernel_name_base<__parallel_for_kernel<_Name...>>
{
};
template <typename... _Name>
class __parallel_reduce_kernel : public __kernel_name_base<__parallel_reduce_kernel<_Name...>>
{
};
template <typename... _Name>
class __parallel_scan_kernel_1 : public __kernel_name_base<__parallel_scan_kernel_1<_Name...>>
{
};
template <typename... _Name>
class __parallel_scan_kernel_2 : public __kernel_name_base<__parallel_scan_kernel_2<_Name...>>
{
};
template <typename... _Name>
class __parallel_scan_kernel_3 : public __kernel_name_base<__parallel_scan_kernel_3<_Name...>>
{
};
template <typename... _Name>
class __parallel_find_or_kernel_1 : public __kernel_name_base<__parallel_find_or_kernel_1<_Name...>>
{
};
template <typename... _Name>
class __parallel_merge_kernel : public __kernel_name_base<__parallel_merge_kernel<_Name...>>
{
};
template <typename... _Name>
class __parallel_sort_kernel_1 : public __kernel_name_base<__parallel_sort_kernel_1<_Name...>>
{
};
template <typename... _Name>
class __parallel_sort_kernel_2 : public __kernel_name_base<__parallel_sort_kernel_2<_Name...>>
{
};
template <typename... _Name>
class __parallel_sort_kernel_3 : public __kernel_name_base<__parallel_sort_kernel_3<_Name...>>
{
};

template <typename _ExecutionPolicy>
class __future
{
    _ExecutionPolicy __exec;

  public:
    __future(const _ExecutionPolicy& __e) : __exec(__e) {}
    void
    wait()
    {
#if !ONEDPL_ALLOW_DEFERRED_WAITING
        __exec.queue().wait_and_throw();
#endif
    }
};

//------------------------------------------------------------------------
// parallel_for - async pattern
//------------------------------------------------------------------------

//General version of parallel_for, one additional parameter - __count of iterations of loop __cgh.parallel_for,
//for some algorithms happens that size of processing range is n, but amount of iterations is n/2.
template <typename _ExecutionPolicy, typename _Fp, typename _Index, typename... _Ranges>
oneapi::dpl::__internal::__enable_if_device_execution_policy<_ExecutionPolicy, __future<_ExecutionPolicy>>
__parallel_for(_ExecutionPolicy&& __exec, _Fp __brick, _Index __count, _Ranges&&... __rngs)
{
    assert(__get_first_range(::std::forward<_Ranges>(__rngs)...).size() > 0);

    using _Policy = typename ::std::decay<_ExecutionPolicy>::type;
    using __kernel_name = typename _Policy::kernel_name;
#if __SYCL_UNNAMED_LAMBDA__
    using __kernel_name_t = __parallel_for_kernel<_Fp, __kernel_name, _Ranges...>;
#else
    using __kernel_name_t = __parallel_for_kernel<__kernel_name>;
#endif

    _PRINT_INFO_IN_DEBUG_MODE(__exec);
    __exec.queue().submit([&__rngs..., &__brick, __count](sycl::handler& __cgh) {
        //get an access to data under SYCL buffer:
        oneapi::dpl::__ranges::__require_access(__cgh, __rngs...);

        __cgh.parallel_for<__kernel_name_t>(sycl::range</*dim=*/1>(__count), [=](sycl::item</*dim=*/1> __item_id) {
            auto __idx = __item_id.get_linear_id();
            __brick(__idx, __rngs...);
        });
    });
    return __future<_ExecutionPolicy>(__exec);
}

//------------------------------------------------------------------------
// parallel_transform_reduce - sync pattern
//------------------------------------------------------------------------

template <typename _Tp, unsigned int __grainsize = 4, typename _ExecutionPolicy, typename _Up, typename _Cp,
          typename _Rp, typename... _Ranges>
oneapi::dpl::__internal::__enable_if_device_execution_policy<_ExecutionPolicy, _Tp>
__parallel_transform_reduce(_ExecutionPolicy&& __exec, _Up __u, _Cp __combine, _Rp __brick_reduce, _Ranges&&... __rngs)
{
    auto __n = __get_first_range(__rngs...).size();
    assert(__n > 0);

    using _Size = decltype(__n);
    using _Policy = typename ::std::decay<_ExecutionPolicy>::type;
    using __kernel_name = typename _Policy::kernel_name;
#if __SYCL_UNNAMED_LAMBDA__
    using __kernel_name_t = __parallel_reduce_kernel<_Up, _Cp, _Rp, __kernel_name, _Ranges...>;
#else
    using __kernel_name_t = __parallel_reduce_kernel<__kernel_name>;
#endif

    sycl::cl_uint __mcu = oneapi::dpl::__internal::__max_compute_units(__exec);
    ::std::size_t __work_group_size = oneapi::dpl::__internal::__max_work_group_size(__exec);
    // change __work_group_size according to local memory limit
    __work_group_size = oneapi::dpl::__internal::__max_local_allocation_size<_ExecutionPolicy, _Tp>(
        ::std::forward<_ExecutionPolicy>(__exec), __work_group_size);
#if _ONEDPL_COMPILE_KERNEL
    sycl::kernel __kernel = __kernel_name_t::__compile_kernel(::std::forward<_ExecutionPolicy>(__exec));
    __work_group_size = ::std::min(__work_group_size, oneapi::dpl::__internal::__kernel_work_group_size(
                                                          ::std::forward<_ExecutionPolicy>(__exec), __kernel));
#endif
    ::std::size_t __iters_per_work_item = __grainsize;
    if (__exec.queue().get_device().is_cpu())
        __iters_per_work_item = __n / (__mcu * __work_group_size);
    ::std::size_t __size_per_work_group =
        __iters_per_work_item * __work_group_size;            // number of buffer elements processed within workgroup
    _Size __n_groups = (__n - 1) / __size_per_work_group + 1; // number of work groups
    _Size __n_items = (__n - 1) / __iters_per_work_item + 1; // number of bunch of elements that the algorithm processes

    _PRINT_INFO_IN_DEBUG_MODE(__exec, __work_group_size);

    // Create temporary global buffers to store temporary values
    sycl::buffer<_Tp> __temp(sycl::range<1>(2 * __n_groups));
    // __is_first == true. Reduce over each work_group
    // __is_first == false. Reduce between work groups
    bool __is_first = true;

    // For memory utilization it's better to use one big buffer instead of two small because size of the buffer is close to a few MB
    _Size __offset_1 = 0;
    _Size __offset_2 = __n_groups;

    sycl::event __reduce_event;
    do
    {
        __reduce_event = __exec.queue().submit([&](sycl::handler& __cgh) {
            __cgh.depends_on(__reduce_event);

            oneapi::dpl::__ranges::__require_access(__cgh, __rngs...); //get an access to data under SYCL buffer
            auto __temp_acc = __temp.template get_access<access_mode::read_write>(__cgh);
            sycl::accessor<_Tp, 1, access_mode::read_write, sycl::access::target::local> __temp_local(
                sycl::range<1>(__work_group_size), __cgh);
            __cgh.parallel_for<__kernel_name_t>(
#if _ONEDPL_COMPILE_KERNEL
                __kernel,
#endif
                sycl::nd_range<1>(sycl::range<1>(__n_groups * __work_group_size), sycl::range<1>(__work_group_size)),
                [=](sycl::nd_item<1> __item_id) {
                    ::std::size_t __global_idx = __item_id.get_global_id(0);
                    ::std::size_t __local_idx = __item_id.get_local_id(0);
                    // 1. Initialization (transform part). Fill local memory
                    if (__is_first)
                    {
                        __u(__item_id, __n, __iters_per_work_item, __temp_local, __rngs...);
                    }
                    else
                    {
                        // TODO: check the approach when we use grainsize here too
                        if (__global_idx < __n_items)
                            __temp_local[__local_idx] = __temp_acc[__offset_2 + __global_idx];
                        __item_id.barrier(sycl::access::fence_space::local_space);
                    }
                    // 2. Reduce within work group using local memory
                    _Tp __result = __brick_reduce(__item_id, __global_idx, __n_items, __temp_local);
                    if (__local_idx == 0)
                    {
                        __temp_acc[__offset_1 + __item_id.get_group(0)] = __result;
                    }
                });
        });
        if (__is_first)
        {
            __is_first = false;
        }
        ::std::swap(__offset_1, __offset_2);
        __n_items = __n_groups;
        __n_groups = (__n_items - 1) / __work_group_size + 1;
    } while (__n_items > 1);
    return __temp.template get_access<access_mode::read_write>()[__offset_2];
}

//------------------------------------------------------------------------
// parallel_transform_scan - sync pattern
//------------------------------------------------------------------------
template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _BinaryOperation, typename _InitType,
          typename _LocalScan, typename _GroupScan, typename _GlobalScan>
oneapi::dpl::__internal::__enable_if_device_execution_policy<
    _ExecutionPolicy, ::std::pair<oneapi::dpl::__internal::__difference_t<_Range2>, typename _InitType::__value_type>>
__parallel_transform_scan(_ExecutionPolicy&& __exec, _Range1&& __rng1, _Range2&& __rng2, _BinaryOperation __binary_op,
                          _InitType __init, _LocalScan __local_scan, _GroupScan __group_scan, _GlobalScan __global_scan)
{
    using _Policy = typename ::std::decay<_ExecutionPolicy>::type;
    using _KernelName = typename _Policy::kernel_name;
    using _Type = typename _InitType::__value_type;

#if __SYCL_UNNAMED_LAMBDA__
    using __kernel_1_name_t = __parallel_scan_kernel_1<_Range1, _Range2, _BinaryOperation, _Type, _LocalScan,
                                                       _GroupScan, _GlobalScan, _KernelName>;
    using __kernel_2_name_t = __parallel_scan_kernel_2<_Range1, _Range2, _BinaryOperation, _Type, _LocalScan,
                                                       _GroupScan, _GlobalScan, _KernelName>;
    using __kernel_3_name_t = __parallel_scan_kernel_3<_Range1, _Range2, _BinaryOperation, _Type, _LocalScan,
                                                       _GroupScan, _GlobalScan, _KernelName>;
#else
    using __kernel_1_name_t = __parallel_scan_kernel_1<_KernelName>;
    using __kernel_2_name_t = __parallel_scan_kernel_2<_KernelName>;
    using __kernel_3_name_t = __parallel_scan_kernel_3<_KernelName>;
#endif

    auto __n = __rng1.size();
    assert(__n > 0);

    auto __mcu = oneapi::dpl::__internal::__max_compute_units(__exec);
    auto __wgroup_size = oneapi::dpl::__internal::__max_work_group_size(__exec);
    // change __wgroup_size according to local memory limit
    __wgroup_size = oneapi::dpl::__internal::__max_local_allocation_size<_ExecutionPolicy, _Type>(
        ::std::forward<_ExecutionPolicy>(__exec), __wgroup_size);

#if _ONEDPL_COMPILE_KERNEL
    auto __kernel_1 = __kernel_1_name_t::__compile_kernel(::std::forward<_ExecutionPolicy>(__exec));
    auto __kernel_2 = __kernel_2_name_t::__compile_kernel(::std::forward<_ExecutionPolicy>(__exec));
    auto __wgroup_size_kernel_1 =
        oneapi::dpl::__internal::__kernel_work_group_size(::std::forward<_ExecutionPolicy>(__exec), __kernel_1);
    auto __wgroup_size_kernel_2 =
        oneapi::dpl::__internal::__kernel_work_group_size(::std::forward<_ExecutionPolicy>(__exec), __kernel_2);
    __wgroup_size = ::std::min({__wgroup_size, __wgroup_size_kernel_1, __wgroup_size_kernel_2});
#endif

    // Practically this is the better value that was found
    auto __iters_per_witem = decltype(__wgroup_size)(16);
    auto __size_per_wg = __iters_per_witem * __wgroup_size;
    auto __n_groups = (__n - 1) / __size_per_wg + 1;
    // Storage for the results of scan for each workgroup
    sycl::buffer<_Type> __wg_sums(__n_groups);

    _PRINT_INFO_IN_DEBUG_MODE(__exec, __wgroup_size, __mcu);

    // 1. Local scan on each workgroup
    auto __submit_event = __exec.queue().submit([&](sycl::handler& __cgh) {
        oneapi::dpl::__ranges::__require_access(__cgh, __rng1, __rng2); //get an access to data under SYCL buffer
        auto __wg_sums_acc = __wg_sums.template get_access<access_mode::discard_write>(__cgh);
        sycl::accessor<_Type, 1, access_mode::discard_read_write, sycl::access::target::local> __local_acc(
            __wgroup_size, __cgh);

        __cgh.parallel_for<__kernel_1_name_t>(
#if _ONEDPL_COMPILE_KERNEL
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
            sycl::accessor<_Type, 1, access_mode::discard_read_write, sycl::access::target::local> __local_acc(
                __wgroup_size, __cgh);

            __cgh.parallel_for<__kernel_2_name_t>(
#if _ONEDPL_COMPILE_KERNEL
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
    __exec.queue().submit([&](sycl::handler& __cgh) {
        __cgh.depends_on(__submit_event);
        oneapi::dpl::__ranges::__require_access(__cgh, __rng1, __rng2); //get an access to data under SYCL buffer
        auto __wg_sums_acc = __wg_sums.template get_access<access_mode::read>(__cgh);
        __cgh.parallel_for<__kernel_3_name_t>(sycl::range<1>(__n_groups * __size_per_wg), [=](sycl::item<1> __item) {
            __global_scan(__item, __rng2, __rng1, __wg_sums_acc, __n, __size_per_wg);
        });
    });

    //point of syncronization (on host access)
    auto __last_scaned_value = __wg_sums.template get_access<access_mode::read>()[__n_groups - 1];
    return ::std::make_pair(__n, __last_scaned_value);
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

        auto __n = oneapi::dpl::__ranges::__get_first_range(__rngs...).size();
        using _Size = decltype(__n);

        auto __global_idx = __item_id.get_global_id(0);
        auto __local_idx = __item_id.get_local_id(0);
        auto __shift = (__wg_size > 8) ? 8 : __wg_size;

        // each work_item processes N_ELEMENTS with step SHIFT
        auto __shift_global = __global_idx / __wg_size;
        auto __shift_local = __local_idx / __shift;
        auto __init_index =
            __shift_global * __wg_size * __n_iter + __shift_local * __shift * __n_iter + __local_idx % __shift;

        for (_Size __i = 0; __i < __n_iter; ++__i)
        {
            //in case of find-semantic __shifted_idx must be the same type as the atomic for a correct comparison
            using _ShiftedIdxType =
                typename ::std::conditional<_OrTagType::value, decltype(__init_index + __i * __shift),
                                            decltype(__found_local.load())>::type;
            auto __current_iter = oneapi::dpl::__internal::__invoke_if_else(
                _BackwardTagType{}, [__n_iter, __i]() { return __n_iter - 1 - __i; }, [__i]() { return __i; });

            _ShiftedIdxType __shifted_idx = __init_index + __current_iter * __shift;
            // TODO:[Performance] the issue with atomic load (in comparison with __shifted_idx for erly exit)
            // should be investigated later, with other HW
            if (__shifted_idx < __n && __pred(__shifted_idx, __rngs...))
            {
                oneapi::dpl::__internal::__invoke_if_else(
                    _OrTagType{}, [&__found_local]() { __found_local.store(1); },
                    [&__found_local, &__comp, &__shifted_idx]() {
                        for (auto __old = __found_local.load(); __comp(__shifted_idx, __old);
                             __old = __found_local.load())
                        {
                            __found_local.compare_exchange_strong(__old, __shifted_idx);
                        }
                    });
                return;
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
    using __kernel_name = typename _Policy::kernel_name;
    using _AtomicType = typename _BrickTag::_AtomicType;

#if __SYCL_UNNAMED_LAMBDA__
    using __kernel_1_name_t = __parallel_find_or_kernel_1<_Ranges..., _Brick, __kernel_name>;
#else
    using __kernel_1_name_t = __parallel_find_or_kernel_1<__kernel_name>;
#endif

    auto __or_tag_check = ::std::is_same<_BrickTag, __parallel_or_tag>{};
    auto __rng_n = oneapi::dpl::__ranges::__get_first_range(::std::forward<_Ranges>(__rngs)...).size();
    assert(__rng_n > 0);

    auto __wgroup_size = oneapi::dpl::__internal::__max_work_group_size(::std::forward<_ExecutionPolicy>(__exec));
#if _ONEDPL_COMPILE_KERNEL
    auto __kernel = __kernel_1_name_t::__compile_kernel(::std::forward<_ExecutionPolicy>(__exec));
    __wgroup_size = ::std::min(__wgroup_size, oneapi::dpl::__internal::__kernel_work_group_size(
                                                  ::std::forward<_ExecutionPolicy>(__exec), __kernel));
#endif

    auto __mcu = oneapi::dpl::__internal::__max_compute_units(__exec);

    auto __n_groups = (__rng_n - 1) / __wgroup_size + 1;
    // TODO: try to change __n_groups with another formula for more perfect load balancing
    __n_groups = ::std::min(__n_groups, decltype(__n_groups)(__mcu));

    auto __n_iter = (__rng_n - 1) / (__n_groups * __wgroup_size) + 1;

    _PRINT_INFO_IN_DEBUG_MODE(__exec, __wgroup_size, __mcu);

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
            sycl::accessor<_AtomicType, 1, access_mode::read_write, sycl::access::target::local> __temp_local(1, __cgh);
            __cgh.parallel_for<__kernel_1_name_t>(
#if _ONEDPL_COMPILE_KERNEL
                __kernel,
#endif
                sycl::nd_range</*dim=*/1>(sycl::range</*dim=*/1>(__n_groups * __wgroup_size),
                                          sycl::range</*dim=*/1>(__wgroup_size)),
                [=](sycl::nd_item</*dim=*/1> __item_id) {
                    auto __local_idx = __item_id.get_local_id(0);

                    // connect global atomic with global memory
                    sycl::atomic<_AtomicType> __found(__temp_acc.get_pointer());
                    // connect local atomic with local memory
                    sycl::atomic<_AtomicType, sycl::access::address_space::local_space> __found_local(
                        __temp_local.get_pointer());

                    // 1. Set value from global atomic to local atomic
                    if (__local_idx == 0)
                    {
                        __found_local.store(__found.load());
                    }
                    __item_id.barrier(sycl::access::fence_space::local_space);

                    // 2. find any element that satisfies pred and Set local atomic value to global atomic
                    constexpr auto __comp = typename _BrickTag::_Compare{};
                    __pred(__item_id, __n_iter, __wgroup_size, __comp, __found_local, __brick_tag, __rngs...);
                    __item_id.barrier(sycl::access::fence_space::local_space);

                    // Set local atomic value to global atomic
                    if (__local_idx == 0 && __comp(__found_local.load(), __found.load()))
                    {
                        oneapi::dpl::__internal::__invoke_if_else(
                            __or_tag_check, [&__found]() { __found.store(1); },
                            [&__found_local, &__found, &__comp]() {
                                for (auto __old = __found.load(); __comp(__found_local.load(), __old);
                                     __old = __found.load())
                                {
                                    __found.compare_exchange_strong(__old, __found_local.load());
                                }
                            });
                    }
                });
        });
        //The end of the scope  -  a point of syncronization (on temporary sycl buffer destruction)
    }

    return oneapi::dpl::__internal::__invoke_if_else(
        __or_tag_check, [&__result]() { return __result; },
        [&__result, &__rng_n, &__init_value]() { return __result != __init_value ? __result : __rng_n; });
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

template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _Range3, typename _Compare>
oneapi::dpl::__internal::__enable_if_device_execution_policy<_ExecutionPolicy, __future<_ExecutionPolicy>>
__parallel_merge(_ExecutionPolicy&& __exec, _Range1&& __rng1, _Range2&& __rng2, _Range3&& __rng3, _Compare __comp)
{
    using _Policy = typename ::std::decay<_ExecutionPolicy>::type;
    using __kernel_name = typename _Policy::kernel_name;
#if __SYCL_UNNAMED_LAMBDA__
    using __kernel_1_name_t = __parallel_merge_kernel<_Range1, _Range2, _Range3, _Compare, __kernel_name>;
#else
    using __kernel_1_name_t = __parallel_merge_kernel<__kernel_name>;
#endif
    auto __n = __rng1.size();
    auto __n_2 = __rng2.size();

    assert(__n > 0 || __n_2 > 0);

    _PRINT_INFO_IN_DEBUG_MODE(__exec);

    const ::std::size_t __chunk = __exec.queue().get_device().is_cpu() ? 128 : 8;
    const auto __max_n = ::std::max(__n, static_cast<decltype(__n)>(__n_2));
    const ::std::size_t __steps = ((__max_n - 1) / __chunk) + 1;

    __exec.queue().submit([&](sycl::handler& __cgh) {
        oneapi::dpl::__ranges::__require_access(__cgh, __rng1, __rng2, __rng3);
        __cgh.parallel_for<__kernel_1_name_t>(sycl::range</*dim=*/1>(__steps), [=](sycl::item</*dim=*/1> __item_id) {
            __full_merge_kernel()(__item_id.get_linear_id() * __chunk, __rng1, decltype(__n)(0), __n, __rng2,
                                  decltype(__n_2)(0), __n_2, __rng3, decltype(__n)(0), __comp, __chunk);
        });
    });
    return __future<_ExecutionPolicy>(__exec);
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

template <typename T>
class t_printer;

template <typename _ExecutionPolicy, typename _Range, typename _Merge, typename _Compare>
oneapi::dpl::__internal::__enable_if_device_execution_policy<_ExecutionPolicy, void>
__parallel_sort_impl(_ExecutionPolicy&& __exec, _Range&& __rng, _Merge __merge, _Compare __comp)
{
    using _Policy = typename ::std::decay<_ExecutionPolicy>::type;
    using __kernel_name = typename _Policy::kernel_name;
#if __SYCL_UNNAMED_LAMBDA__
    using __kernel_1_name_t = __parallel_sort_kernel_1<_Range, _Merge, _Compare, __kernel_name>;
    using __kernel_2_name_t = __parallel_sort_kernel_2<_Range, _Merge, _Compare, __kernel_name>;
    using __kernel_3_name_t = __parallel_sort_kernel_3<_Range, _Merge, _Compare, __kernel_name>;
#else
    using __kernel_1_name_t = __parallel_sort_kernel_1<__kernel_name>;
    using __kernel_2_name_t = __parallel_sort_kernel_2<__kernel_name>;
    using __kernel_3_name_t = __parallel_sort_kernel_3<__kernel_name>;
#endif

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
        __cgh.parallel_for<__kernel_1_name_t>(sycl::range</*dim=*/1>(__leaf_steps),
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

        __event1 = __exec.queue().submit([&](sycl::handler& __cgh) {
            __cgh.depends_on(__event1);
            oneapi::dpl::__ranges::__require_access(__cgh, __rng);
            auto __temp_acc = __temp.template get_access<__par_backend_hetero::access_mode::read_write>(__cgh);
            __cgh.parallel_for<__kernel_2_name_t>(
                sycl::range</*dim=*/1>(__steps), [=](sycl::item</*dim=*/1> __item_id) {
                    const _Size __idx = __item_id.get_linear_id();
                    // Borders of the first and the second sorted sequences
                    const _Size __start_1 = sycl::min(__sorted_pair * ((__idx * __chunk) / __sorted), __n);
                    const _Size __end_1 = sycl::min(__start_1 + __sorted, __n);
                    const _Size __start_2 = __end_1;
                    const _Size __end_2 = sycl::min(__start_2 + __sorted, __n);

                    // Distance between the beginning of a sorted sequence and the begining of a chunk
                    const _Size __offset = __chunk * (__idx % __chunks_in_sorted);

                    if (!__data_in_temp)
                    {
                        __merge(__offset, __rng, __start_1, __end_1, __rng, __start_2, __end_2, __temp_acc, __start_1,
                                __comp, __chunk);
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
        __exec.queue().submit([&](sycl::handler& __cgh) {
            __cgh.depends_on(__event1);
            oneapi::dpl::__ranges::__require_access(__cgh, __rng);
            auto __temp_acc = __temp.template get_access<access_mode::read>(__cgh);
            // We cannot use __cgh.copy here because of zip_iterator usage
            __cgh.parallel_for<__kernel_3_name_t>(sycl::range</*dim=*/1>(__n), [=](sycl::item</*dim=*/1> __item_id) {
                __rng[__item_id.get_linear_id()] = __temp_acc[__item_id];
            });
        });
    }
}

template <typename _ExecutionPolicy, typename _Range, typename _Merge, typename _Compare>
oneapi::dpl::__internal::__enable_if_device_execution_policy<_ExecutionPolicy, void>
__parallel_partial_sort_impl(_ExecutionPolicy&& __exec, _Range&& __rng, _Merge __merge, _Compare __comp)
{
    using _Policy = typename ::std::decay<_ExecutionPolicy>::type;
    using __kernel_name = typename _Policy::kernel_name;
#if __SYCL_UNNAMED_LAMBDA__
    using __kernel_1_name_t = __parallel_sort_kernel_1<_Range, _Merge, _Compare, __kernel_name>;
    using __kernel_2_name_t = __parallel_sort_kernel_2<_Range, _Merge, _Compare, __kernel_name>;
#else
    using __kernel_1_name_t = __parallel_sort_kernel_1<__kernel_name>;
    using __kernel_2_name_t = __parallel_sort_kernel_2<__kernel_name>;
#endif

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
        __event1 = __exec.queue().submit([&](sycl::handler& __cgh) {
            __cgh.depends_on(__event1);
            oneapi::dpl::__ranges::__require_access(__cgh, __rng);
            auto __temp_acc = __temp.template get_access<access_mode::read_write>(__cgh);
            __cgh.parallel_for<__kernel_1_name_t>(sycl::range</*dim=*/1>(__n), [=](sycl::item</*dim=*/1> __item_id) {
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
                    __merge(__global_idx, __temp_acc, __start, __end_1, __temp_acc, __end_1, __end_2, __rng, __start,
                            __comp);
                }
            });
        });
        __data_in_temp = !__data_in_temp;
        __k *= 2;
    } while (__k < __n);

    // if results are in temporary buffer then copy back those
    if (__data_in_temp)
    {
        __exec.queue().submit([&](sycl::handler& __cgh) {
            __cgh.depends_on(__event1);
            oneapi::dpl::__ranges::__require_access(__cgh, __rng);
            auto __temp_acc = __temp.template get_access<access_mode::read>(__cgh);
            // we cannot use __cgh.copy here because of zip_iterator usage
            __cgh.parallel_for<__kernel_2_name_t>(sycl::range</*dim=*/1>(__n), [=](sycl::item</*dim=*/1> __item_id) {
                __rng[__item_id.get_linear_id()] = __temp_acc[__item_id];
            });
        });
    }
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
template <typename _ExecutionPolicy, typename _Iterator, typename _Compare>
__enable_if_t<oneapi::dpl::__internal::__is_device_execution_policy<__decay_t<_ExecutionPolicy>>::value &&
                  __is_radix_sort_usable_for_type<__value_t<_Iterator>, _Compare>::value,
              __future<_ExecutionPolicy>>
__parallel_stable_sort(_ExecutionPolicy&& __exec, _Iterator __first, _Iterator __last, _Compare __comp)
{
    auto __keep = oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read_write, _Iterator>();
    auto __buf = __keep(__first, __last);

    __parallel_radix_sort<__internal::__is_comp_ascending<__decay_t<_Compare>>::value>(__exec, __buf.all_view());
    return __future<_ExecutionPolicy>(__exec);
}
#endif

template <typename _ExecutionPolicy, typename _Iterator, typename _Compare>
__enable_if_t<oneapi::dpl::__internal::__is_device_execution_policy<__decay_t<_ExecutionPolicy>>::value &&
                  !__is_radix_sort_usable_for_type<__value_t<_Iterator>, _Compare>::value,
              __future<_ExecutionPolicy>>
__parallel_stable_sort(_ExecutionPolicy&& __exec, _Iterator __first, _Iterator __last, _Compare __comp)
{
    auto __keep = oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read_write, _Iterator>();
    auto __buf = __keep(__first, __last);

    __parallel_sort_impl(::std::forward<_ExecutionPolicy>(__exec), __buf.all_view(),
                         // Pass special tag to choose 'full' merge subroutine at compile-time
                         __full_merge_kernel(), __comp);
    return __future<_ExecutionPolicy>(__exec);
}

//------------------------------------------------------------------------
// parallel_partial_sort - async pattern
//-----------------------------------------------------------------------

// TODO: check if it makes sense to move these wrappers out of backend to a common place
// TODO: consider changing __partial_merge_kernel to make it compatible with
//       __full_merge_kernel in order to use __parallel_sort_impl routine
template <typename _ExecutionPolicy, typename _Iterator, typename _Compare>
oneapi::dpl::__internal::__enable_if_device_execution_policy<_ExecutionPolicy, __future<_ExecutionPolicy>>
__parallel_partial_sort(_ExecutionPolicy&& __exec, _Iterator __first, _Iterator __mid, _Iterator __last,
                        _Compare __comp)
{
    const auto __mid_idx = __mid - __first;

    auto __keep = oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read_write, _Iterator>();
    auto __buf = __keep(__first, __last);

    __parallel_partial_sort_impl(::std::forward<_ExecutionPolicy>(__exec), __buf.all_view(),
                                 __partial_merge_kernel<decltype(__mid_idx)>{__mid_idx}, __comp);
    return __future<_ExecutionPolicy>(__exec);
}

} // namespace __par_backend_hetero
} // namespace dpl
} // namespace oneapi

#endif /* _ONEDPL_parallel_backend_sycl_H */
