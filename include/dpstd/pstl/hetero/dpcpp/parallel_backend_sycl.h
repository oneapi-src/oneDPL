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

//!!! NOTE: This file should be included under the macro _PSTL_BACKEND_SYCL
#ifndef _PSTL_parallel_backend_sycl_H
#define _PSTL_parallel_backend_sycl_H

#include <CL/sycl.hpp>

#include <cassert>
#include <algorithm>
#include <type_traits>
#include <iostream>

#include "parallel_backend_sycl_utils.h"
#include "../../execution_impl.h"
#include "execution_sycl_defs.h"
#include "sycl_iterator.h"
#include "../../iterator_impl.h"

#if _USE_SUB_GROUPS
#    include "parallel_backend_sycl_radix_sort.h"
#endif

namespace dpstd
{
namespace __par_backend_hetero
{

namespace sycl = cl::sycl;

//-----------------------------------------------------------------------------
//- iter_mode_resolver
//-----------------------------------------------------------------------------

using iter_mode_t = cl::sycl::access::mode;

// iter_mode_resolver resolves the situations when
// the access mode provided by a user differs (inMode) from
// the access mode required by an algorithm (outMode).
// In general case iter_mode_resolver accepts the only situations
// when inMode == outMode,
// whereas the template specializations describe cases with specific
// inMode and outMode and the preferred access mode between the two.
template <iter_mode_t inMode, iter_mode_t outMode>
struct iter_mode_resolver
{
    static_assert(inMode == outMode, "Access mode provided by user conflicts with the one required by the algorithm");
    static constexpr iter_mode_t value = inMode;
};

template <>
struct iter_mode_resolver<read, read_write>
{
    static constexpr iter_mode_t value = read;
};

template <>
struct iter_mode_resolver<write, read_write>
{
    static constexpr iter_mode_t value = write;
};

template <>
struct iter_mode_resolver<read_write, read>
{
    //TODO: warn user that the access mode is changed
    static constexpr iter_mode_t value = read;
};

template <>
struct iter_mode_resolver<read_write, write>
{
    //TODO: warn user that the access mode is changed
    static constexpr iter_mode_t value = write;
};

template <>
struct iter_mode_resolver<discard_write, write>
{
    static constexpr iter_mode_t value = discard_write;
};

template <>
struct iter_mode_resolver<discard_read_write, write>
{
    //TODO: warn user that the access mode is changed
    static constexpr iter_mode_t value = write;
};

template <>
struct iter_mode_resolver<discard_read_write, read_write>
{
    static constexpr iter_mode_t value = discard_read_write;
};

//-----------------------------------------------------------------------------
//- iter_mode
//-----------------------------------------------------------------------------

// create iterator with different access mode
template <cl::sycl::access::mode outMode>
struct iter_mode
{
    // for common heterogeneous iterator
    template <template <cl::sycl::access::mode, typename...> class Iter, cl::sycl::access::mode inMode,
              typename... Types>
    Iter<iter_mode_resolver<inMode, outMode>::value, Types...>
    operator()(const Iter<inMode, Types...>& it)
    {
        constexpr iter_mode_t preferredMode = iter_mode_resolver<inMode, outMode>::value;
        if (inMode == preferredMode)
            return it;
        return Iter<preferredMode, Types...>(it);
    }
    // for ounting_iterator
    template <typename T>
    dpstd::counting_iterator<T>
    operator()(const dpstd::counting_iterator<T>& it)
    {
        return it;
    }
    // for zip_iterator
    template <typename... Iters>
    auto
    operator()(const dpstd::zip_iterator<Iters...>& it) -> decltype(map_zip(*this, it.base()))
    {
        return map_zip(*this, it.base());
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

template <cl::sycl::access::mode outMode, typename _Iterator>
auto
make_iter_mode(const _Iterator& __it) -> decltype(iter_mode<outMode>()(__it))
{
    return iter_mode<outMode>()(__it);
}

// function is needed to wrap kernel name into another class
template <template <typename> class _NewKernelName, typename _Policy,
          dpstd::__internal::__enable_if_device_execution_policy<_Policy, int> = 0>
auto
make_wrapped_policy(_Policy&& __policy)
    -> decltype(dpstd::execution::make_device_policy<_NewKernelName<typename std::decay<_Policy>::type::kernel_name>>(
        std::forward<_Policy>(__policy)))
{
    return dpstd::execution::make_device_policy<_NewKernelName<typename std::decay<_Policy>::type::kernel_name>>(
        std::forward<_Policy>(__policy));
}

#if _PSTL_FPGA_DEVICE
template <template <typename> class _NewKernelName, typename _Policy,
          dpstd::__internal::__enable_if_fpga_execution_policy<_Policy, int> = 0>
auto
make_wrapped_policy(_Policy&& __policy) -> decltype(
    dpstd::execution::make_fpga_policy<_NewKernelName<typename std::decay<_Policy>::type::kernel_name>,
                                       std::decay<_Policy>::type::unroll_factor>(std::forward<_Policy>(__policy)))
{
    return dpstd::execution::make_fpga_policy<_NewKernelName<typename std::decay<_Policy>::type::kernel_name>,
                                              std::decay<_Policy>::type::unroll_factor>(
        std::forward<_Policy>(__policy));
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
class __parallel_reduce_kernel_1 : public __kernel_name_base<__parallel_reduce_kernel_1<_Name...>>
{
};
template <typename... _Name>
class __parallel_reduce_kernel_2 : public __kernel_name_base<__parallel_reduce_kernel_2<_Name...>>
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

//------------------------------------------------------------------------
// parallel_for_ext
//------------------------------------------------------------------------
//Extended version of parallel_for, one additional parameter was added to control
//size of __target__buffer. For some algorithms happens that size of
//processing range is n, but amount of iterations is n/2.

template <typename _ExecutionPolicy, typename _Iterator, typename _Fp>
dpstd::__internal::__enable_if_device_execution_policy<_ExecutionPolicy, void>
__parallel_for_ext(_ExecutionPolicy&& __exec, _Iterator __first, _Iterator __mid, _Iterator __last, _Fp __brick)
{
    if (__first == __last || __first == __mid)
        return;

    auto __target_buffer = __internal::get_buffer()(__first, __last); // hides sycl::buffer
    // inferring number of dimension:
    //loop over target_buffer.get_count() / target_buffer.get_range().get(i);
    //e.g.dimension1 = target_buffer.get_range().get(0);
    // SYCL queue is inside execuiton policy. It is instantiated either implicitly or provided by user.
    using _Policy = typename std::decay<_ExecutionPolicy>::type;
    using __kernel_name = typename _Policy::kernel_name;
#if __SYCL_UNNAMED_LAMBDA__
    using __kernel_name_t = __parallel_for_kernel<_Iterator, _Fp, __kernel_name>;
#else
    using __kernel_name_t = __parallel_for_kernel<__kernel_name>;
#endif

    _PRINT_INFO_IN_DEBUG_MODE(__exec);
    //In reverse we need (__mid - __first) swaps
    auto __n = __mid - __first;
    __exec.queue().submit([&__target_buffer, &__brick, __n](sycl::handler& __cgh) {
        auto __acc = __internal::get_access<_Iterator>(__cgh)(__target_buffer);

        __cgh.parallel_for<__kernel_name_t>(sycl::range</*dim=*/1>(__n), [=](sycl::item</*dim=*/1> __item_id) mutable {
            auto __idx = __item_id.get_linear_id();
            __brick(__idx, __acc);
        });
    });
}

//------------------------------------------------------------------------
// parallel_for
//------------------------------------------------------------------------

template <typename _ExecutionPolicy, typename _Iterator, typename _Fp>
dpstd::__internal::__enable_if_device_execution_policy<_ExecutionPolicy, void>
__parallel_for(_ExecutionPolicy&& __exec, _Iterator __first, _Iterator __last, _Fp __brick)
{
    __parallel_for_ext(std::forward<_ExecutionPolicy>(__exec), __first, __last, __last, __brick);
}

//------------------------------------------------------------------------
// parallel_transform_reduce
//------------------------------------------------------------------------

template <typename _Tp, typename _ExecutionPolicy, typename _Iterator, typename _Up, typename _Cp, typename _Rp,
          typename... _PolicyParams>
dpstd::__internal::__enable_if_device_execution_policy<_ExecutionPolicy, _Tp>
__parallel_transform_reduce(_ExecutionPolicy&& __exec, _Iterator __first, _Iterator __last, _Up __u, _Cp __combine,
                            _Rp __brick_reduce)
{
    using _Policy = typename std::decay<_ExecutionPolicy>::type;
    using __kernel_name = typename _Policy::kernel_name;
#if __SYCL_UNNAMED_LAMBDA__
    using __kernel_1_name_t = __parallel_reduce_kernel_1<_Iterator, _Up, _Cp, _Rp, __kernel_name>;
    using __kernel_2_name_t = __parallel_reduce_kernel_2<_Iterator, _Up, _Cp, _Rp, __kernel_name>;
#else
    using __kernel_1_name_t = __parallel_reduce_kernel_1<__kernel_name>;
    using __kernel_2_name_t = __parallel_reduce_kernel_2<__kernel_name>;
#endif

    auto __target_buffer = __internal::get_buffer()(__first, __last); // hides sycl::buffer
    auto __wgroup_size = dpstd::__internal::__max_work_group_size(__exec);
#if _PSTL_COMPILE_KERNEL
    auto __kernel = __kernel_1_name_t::__compile_kernel(std::forward<_ExecutionPolicy>(__exec));
    __wgroup_size = std::min(
        __wgroup_size, dpstd::__internal::__kernel_work_group_size(std::forward<_ExecutionPolicy>(__exec), __kernel));
#endif
    auto __mcu = dpstd::__internal::__max_compute_units(std::forward<_ExecutionPolicy>(__exec));
    auto __n = __last - __first;
    auto __n_groups = (__n - 1) / __wgroup_size + 1;
    __n_groups = std::min(decltype(__n_groups)(__mcu), __n_groups);
    // TODO: try to change __n_groups with another formula for more perfect load balancing

    _PRINT_INFO_IN_DEBUG_MODE(__exec, __wgroup_size, __mcu);

    // 0. Create temporary global buffer to store temporary value
    auto __temp = sycl::buffer<_Tp, 1>(sycl::range<1>(__n_groups));
    // 1. Reduce over each work_group
    auto __local_reduce_event =
        __exec.queue().submit([&__target_buffer, &__temp, &__brick_reduce, &__u, __n, __n_groups,
#if _PSTL_COMPILE_KERNEL
                               &__kernel,
#endif
                               __wgroup_size](sycl::handler& __cgh) {
            auto __acc = __internal::get_access<_Iterator>(__cgh)(__target_buffer);
            auto __temp_acc = __temp.template get_access<discard_write>(__cgh);
            // Create temporary local buffer
            // TODO: add check for local_memory size
            sycl::accessor<_Tp, 1, read_write, sycl::access::target::local> __temp_local(sycl::range<1>(__wgroup_size),
                                                                                         __cgh);
            __cgh.parallel_for<__kernel_1_name_t>(
#if _PSTL_COMPILE_KERNEL
                __kernel,
#endif
                sycl::nd_range</*dim=*/1>(sycl::range</*dim=*/1>(__n_groups * __wgroup_size),
                                          sycl::range</*dim=*/1>(__wgroup_size)),
                [=](sycl::nd_item</*dim=*/1> __item_id) mutable {
                    auto __global_idx = __item_id.get_global_id(0);
                    // 1. Initialization (transform part). Fill local memory
                    __u(__item_id, __global_idx, __acc, __n, __temp_local);
                    // 2. Reduce within work group
                    auto __res = __brick_reduce(__item_id, __global_idx, __n, __temp_local);
                    if (__item_id.get_local_id(0) == 0)
                    {
                        // TODO: replace get_group() with atomic
                        __temp_acc[__item_id.get_group(0)] = __res;
                    }
                });
        });
    // TODO: think about replacing the section 2 with the code below. So section 2 is not needed
    // do{
    // __n = __n_groups;
    // __n_groups = (__n_groups - 1) / __wgroup_size + 1;
    // ...
    // }
    // while(__n_groups!=1);

    // 2. global reduction
    auto __reduce_event = __local_reduce_event;
    if (__n_groups > 1)
    {
        auto __k = decltype(__n_groups)(1);
        do
        {
            __reduce_event =
                __exec.queue().submit([&__reduce_event, &__temp, &__combine, __k, __n_groups](sycl::handler& __cgh) {
                    __cgh.depends_on(__reduce_event);
                    auto __temp_acc = __temp.template get_access<read_write>(__cgh);
                    __cgh.parallel_for<__kernel_2_name_t>(
                        sycl::range</*dim=*/1>(__n_groups), [=](sycl::item</*dim=*/1> __item_id) mutable {
                            auto __global_idx = __item_id.get_linear_id();
                            if (__global_idx % (2 * __k) == 0 && __global_idx + __k < __n_groups)
                            {
                                __temp_acc[__global_idx] =
                                    __combine(__temp_acc[__global_idx], __temp_acc[__global_idx + __k]);
                            }
                        });
                });
            __k *= 2;
        } while (__k < __n_groups);
    }

    return __temp.template get_access<read>()[0];
}

//------------------------------------------------------------------------
// parallel_transform_scan
//------------------------------------------------------------------------

// returns iterator that points on the last partial sum
// and the last partial sum
template <typename _ExecutionPolicy, typename _Iterator1, typename _Iterator2, typename _BinaryOperation, typename _Tp,
          typename _Transform, typename _Reduce, typename _Scan>
dpstd::__internal::__enable_if_device_execution_policy<_ExecutionPolicy, std::pair<_Iterator2, _Tp>>
__parallel_transform_scan(_ExecutionPolicy&& __exec, _Iterator1 __first, _Iterator1 __last, _Iterator2 __result,
                          _BinaryOperation __binary_op, _Tp __init, _Transform __brick_transform,
                          _Reduce __brick_reduce, _Scan __brick_scan)
{
    using _Policy = typename std::decay<_ExecutionPolicy>::type;
    using __kernel_name = typename _Policy::kernel_name;
#if __SYCL_UNNAMED_LAMBDA__
    using __kernel_1_name_t = __parallel_scan_kernel_1<_Iterator1, _Iterator2, _BinaryOperation, _Tp, _Transform,
                                                       _Reduce, _Scan, __kernel_name>;
    using __kernel_2_name_t = __parallel_scan_kernel_2<_Iterator1, _Iterator2, _BinaryOperation, _Tp, _Transform,
                                                       _Reduce, _Scan, __kernel_name>;
#else
    using __kernel_1_name_t = __parallel_scan_kernel_1<__kernel_name>;
    using __kernel_2_name_t = __parallel_scan_kernel_2<__kernel_name>;
#endif

    if (__first == __last)
        return std::make_pair(__result, __init);

    // __result_buffer is declared before __target_buffer so it is copied back to host last.
    // This is needed for in-place scan to produce the correct result.
    auto __result_buffer = __internal::get_buffer()(__result, __result + (__last - __first));
    auto __target_buffer = __internal::get_buffer()(__first, __last);
    auto __wgroup_size = dpstd::__internal::__max_work_group_size(__exec);
    auto __mcu = dpstd::__internal::__max_compute_units(__exec);
    auto __n = __last - __first;
#if _PSTL_COMPILE_KERNEL
    auto __kernel = __kernel_2_name_t::__compile_kernel(std::forward<_ExecutionPolicy>(__exec));
    __wgroup_size = std::min(
        __wgroup_size, dpstd::__internal::__kernel_work_group_size(std::forward<_ExecutionPolicy>(__exec), __kernel));
#endif

    auto __n_groups = (__n - 1) / __wgroup_size + 1;
    __n_groups = std::min(decltype(__n_groups)(__mcu), __n_groups);
    // TODO: try to change __n_groups with another formula for more perfect load balancing
    _PRINT_INFO_IN_DEBUG_MODE(__exec, __wgroup_size, __mcu);

    // TODO: try to replace with int8_t
    using _AtomicType = int32_t;
    // 0. Create temporary global buffer to store temporary value
    auto __local_sums = sycl::buffer<_Tp, /*dim=*/1>(__n_groups);          // temporary storage for global atomic
    auto __ready_flags = sycl::buffer<_AtomicType, /*dim=*/1>(__n_groups); // temporary storage for global atomic

    // 1. Initialize temp buffer
    auto __init_event = __exec.queue().submit([&](sycl::handler& __cgh) {
        auto __ready_flags_acc = __ready_flags.template get_access<discard_write>(__cgh);

        __cgh.parallel_for<__kernel_1_name_t>(sycl::range</*dim=*/1>(__n_groups), [=](sycl::item</*dim=*/1> __item_id) {
            __ready_flags_acc[__item_id] = 0;
        });
    });
    uint32_t __for_dynamic_id = 0;
    auto __dynamic_id_buf =
        sycl::buffer<uint32_t, /*dim=*/1>(&__for_dynamic_id, 1); // temporary storage for group_id atomic
    // Main parallel_for
    __exec.queue().submit([&](sycl::handler& __cgh) {
        __cgh.depends_on(__init_event);
        auto __acc = __internal::get_access<_Iterator1>(__cgh)(__target_buffer);
        auto __dynamic_id_acc = __dynamic_id_buf.template get_access<read_write>(__cgh);
        auto __local_sums_acc = __local_sums.template get_access<read_write>(__cgh);
        auto __result_acc = __internal::get_access<_Iterator2>(__cgh)(__result_buffer);
        auto __ready_flags_acc = __ready_flags.template get_access<read_write>(__cgh);

        // create local accessors
        sycl::accessor<uint32_t, 1, read_write, sycl::access::target::local> __group_id_local(1, __cgh);
        sycl::accessor<_Tp, 1, read_write, sycl::access::target::local> __transform_local(__wgroup_size, __cgh);
        sycl::accessor<_Tp, 1, read_write, sycl::access::target::local> __reduce_local_mem(__wgroup_size, __cgh);
        __cgh.parallel_for<__kernel_2_name_t>(
#if _PSTL_COMPILE_KERNEL
            __kernel,
#endif
            sycl::nd_range</*dim=*/1>(sycl::range</*dim=*/1>(__n_groups * __wgroup_size),
                                      sycl::range</*dim=*/1>(__wgroup_size)),
            [=](sycl::nd_item</*dim=*/1> __item_id) mutable {
                auto __local_idx = __item_id.get_local_id(0);
                auto __group_size = __item_id.get_local_range().size();
                // dynamic group_id
                if (__local_idx == 0)
                {
                    // add 1 to __dynamic_id_acc atomically
                    __group_id_local[0] = sycl::atomic<uint32_t>(__dynamic_id_acc.get_pointer()).fetch_add(uint32_t(1));
                }
                __item_id.barrier(sycl::access::fence_space::local_space);
                auto __group_id = __group_id_local[0];
                auto __global_idx = (__group_id * __group_size) + __local_idx;

                // 2. Initialization (transform part). Fill local memory
                __brick_transform(__item_id, __global_idx, __acc, __n, __transform_local);
                // copy to another memory to save the state
                __reduce_local_mem[__local_idx] = __transform_local[__local_idx];
                __item_id.barrier(sycl::access::fence_space::local_space);

                // TODO: think about the model Scan-Add. It will help us to get rid of 2 reduce calls
                // and __reduce_local_mem won't be needed
                // 3. local reduce
                auto __local_reduce = __brick_reduce(__item_id, __global_idx, __n, __reduce_local_mem);
                if (__group_id == 0 && __local_idx == 0)
                {
                    // the next 2 lines might be swapped
                    __local_sums_acc[0] = __binary_op(__init, __local_reduce);
                    sycl::atomic<_AtomicType>(__ready_flags_acc.get_pointer()).store(1);
                }
                __item_id.barrier(sycl::access::fence_space::local_space);

                // 4. get reduced value from the previous work group
                _Tp __new_init = __init;
                if (__group_id != 0 && __local_idx == 0)
                {
                    _AtomicType __temp;
                    // wait for updating atomic from the previous work group
                    while (
                        (__temp = sycl::atomic<_AtomicType>(__ready_flags_acc.get_pointer() + __group_id - 1).load()) ==
                        0)
                    {
                    }
                    auto __new_res = __binary_op(__local_sums_acc[__group_id - 1], __local_reduce);
                    // the next 2 lines might be swapped
                    __local_sums_acc[__group_id] = __new_res;
                    sycl::atomic<_AtomicType>(__ready_flags_acc.get_pointer() + __group_id).store(1);
                    __new_init = __local_sums_acc[__group_id - 1];
                }
                __item_id.barrier(sycl::access::fence_space::local_space);

                // 5. local scan and putting down to __result
                __brick_scan(__item_id, __global_idx, __n, __transform_local, __acc, __result_acc, __new_init);
            });
    });
    auto __last_reduced_value = __local_sums.template get_access<read>()[__n_groups - 1];
    return std::make_pair(__result + __n, __last_reduced_value);
}

//------------------------------------------------------------------------
// find_or tags
//------------------------------------------------------------------------

// Tag for __parallel_find_or to find the first element that satisfies predicate
template <typename _IteratorType>
struct __parallel_find_forward_tag
{
// FPGA devices don't support 64-bit atomics
#if _PSTL_FPGA_DEVICE
    using _AtomicType = uint32_t;
#else
    using _AtomicType = typename std::iterator_traits<_IteratorType>::difference_type;
#endif
    using _Compare = dpstd::__internal::__pstl_less;

    // The template parameter is intended to unify __init_value in tags.
    constexpr static _AtomicType
    __init_value(_IteratorType __first, _IteratorType __last)
    {
        return __last - __first;
    }
};

// Tag for __parallel_find_or to find the last element that satisfies predicate
template <typename _IteratorType>
struct __parallel_find_backward_tag
{
// FPGA devices don't support 64-bit atomics
#if _PSTL_FPGA_DEVICE
    using _AtomicType = int32_t;
#else
    using _AtomicType = typename std::iterator_traits<_IteratorType>::difference_type;
#endif
    using _Compare = dpstd::__internal::__pstl_greater;

    constexpr static _AtomicType __init_value(_IteratorType, _IteratorType) { return _AtomicType{-1}; }
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
    template <typename _IteratorType>
    constexpr static _AtomicType
    __init_value(_IteratorType __first, _IteratorType __last)
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

    template <typename _NDItemId, typename _IterSize, typename _WgSize, typename _Acc, typename _Size, typename _Acc2,
              typename _Size2, typename _LocalAtomic, typename _Compare, typename _BrickTag>
    void
    operator()(const _NDItemId __item_id, const _IterSize __n_iter, const _WgSize __wg_size, _Acc& __acc,
               const _Size __n, _Acc2& __s_acc, const _Size2 __s_n, _Compare __comp, _LocalAtomic& __found_local,
               _BrickTag __brick_tag)
    {
        using __par_backend_hetero::__parallel_or_tag;
        using _OrTagType = std::is_same<_BrickTag, __par_backend_hetero::__parallel_or_tag>;

        auto __global_idx = __item_id.get_global_id(0);
        auto __local_idx = __item_id.get_local_id(0);
        auto __global_range_size = __item_id.get_global_range().size();

        constexpr auto __shift = 8;

        // each work_item processes N_ELEMENTS with step SHIFT
        auto __shift_global = __global_idx / __wg_size;
        auto __shift_local = __local_idx / __shift;
        auto __init_index =
            __shift_global * __wg_size * __n_iter + __shift_local * __shift * __n_iter + __local_idx % __shift;

        for (_Size __i = 0; __i < __n_iter; ++__i)
        {

            //in case of find-semantic __shifted_idx must be the same type as the atomic for a correct comparison
            using _ShiftedIdxType = typename std::conditional<_OrTagType::value, decltype(__init_index + __i * __shift),
                                                              decltype(__found_local.load())>::type;

            _ShiftedIdxType __shifted_idx = __init_index + __i * __shift;
            // TODO:[Performance] the issue with atomic load (in comparison with __shifted_idx for erly exit)
            // should be investigated later, with other HW

            if (__shifted_idx < __n && __pred(__shifted_idx, __acc, __n, __s_acc, __s_n))
            {
                dpstd::__internal::__invoke_if_else(_OrTagType{}, [&__found_local]() { __found_local.store(1); },
                                                    [&__found_local, &__comp, &__shifted_idx]() {
                                                        for (auto __old = __found_local.load();
                                                             __comp(__shifted_idx, __old); __old = __found_local.load())
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
// parallel_find_or
//------------------------------------------------------------------------

// Base pattern for __parallel_or and __parallel_find. The execution depends on tag type _BrickTag.
template <typename _ExecutionPolicy, typename _Iterator1, typename _Iterator2, typename _Brick, typename _BrickTag>
dpstd::__internal::__enable_if_device_execution_policy<
    _ExecutionPolicy,
    typename std::conditional<std::is_same<_BrickTag, __parallel_or_tag>::value, bool, _Iterator1>::type>
__parallel_find_or(_ExecutionPolicy&& __exec, _Iterator1 __first, _Iterator1 __last, _Iterator2 __s_first,
                   _Iterator2 __s_last, _Brick __f, _BrickTag __brick_tag)
{
    using _Policy = typename std::decay<_ExecutionPolicy>::type;
    using __kernel_name = typename _Policy::kernel_name;
#if __SYCL_UNNAMED_LAMBDA__
    using __kernel_1_name_t = __parallel_find_or_kernel_1<_Iterator1, _Iterator2, _Brick, __kernel_name>;
#else
    using __kernel_1_name_t = __parallel_find_or_kernel_1<__kernel_name>;
#endif

    auto __or_tag_check = std::is_same<_BrickTag, __parallel_or_tag>{};
    // if sequence is empty number of groups couldn't be counted correctly and buffer couldn't be created from range
    if (__first == __last)
        return dpstd::__internal::__invoke_if_else(__or_tag_check, []() { return false; },
                                                   [&__first]() { return __first; });
    auto __target_buffer = __internal::get_buffer()(__first, __last); // hides sycl::buffer
    auto __n = __last - __first;

    auto __s_target_buffer = __internal::get_buffer()(__s_first, __s_last);
    auto __s_n = __s_last - __s_first;
    auto __wgroup_size = dpstd::__internal::__max_work_group_size(std::forward<_ExecutionPolicy>(__exec));
#if _PSTL_COMPILE_KERNEL
    auto __kernel = __kernel_1_name_t::__compile_kernel(std::forward<_ExecutionPolicy>(__exec));
    __wgroup_size = std::min(
        __wgroup_size, dpstd::__internal::__kernel_work_group_size(std::forward<_ExecutionPolicy>(__exec), __kernel));
#endif

    auto __mcu = dpstd::__internal::__max_compute_units(__exec);

    auto __n_groups = (__n - 1) / __wgroup_size + 1;
    // TODO: try to change __n_groups with another formula for more perfect load balancing
    __n_groups = std::min(__n_groups, decltype(__n_groups)(__mcu));

    auto __n_iter = (__n - 1) / (__n_groups * __wgroup_size) + 1;

    _PRINT_INFO_IN_DEBUG_MODE(__exec, __wgroup_size, __mcu);

    using _AtomicType = typename _BrickTag::_AtomicType;
    _AtomicType __init_value = _BrickTag::__init_value(__first, __last);
    auto __result = __init_value;

    auto __pred = dpstd::__par_backend_hetero::__early_exit_find_or<_ExecutionPolicy, _Brick>{__f};

    // scope is to copy data back to __result after destruction of temporary sycl:buffer
    {
        auto __temp = sycl::buffer<_AtomicType, 1>(&__result, 1); // temporary storage for global atomic

        // main parallel_for
        __exec.queue().submit([&](sycl::handler& __cgh) {
            auto __acc = __internal::get_access<_Iterator1>(__cgh)(__target_buffer);
            auto __s_acc = __internal::get_access<_Iterator2>(__cgh)(__s_target_buffer);
            auto __temp_acc = __temp.template get_access<read_write>(__cgh);

            // create local accessor to connect atomic with
            sycl::accessor<_AtomicType, 1, read_write, sycl::access::target::local> __temp_local(1, __cgh);
            __cgh.parallel_for<__kernel_1_name_t>(
#if _PSTL_COMPILE_KERNEL
                __kernel,
#endif
                sycl::nd_range</*dim=*/1>(sycl::range</*dim=*/1>(__n_groups * __wgroup_size),
                                          sycl::range</*dim=*/1>(__wgroup_size)),
                [=](sycl::nd_item</*dim=*/1> __item_id) mutable {
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
                    __pred(__item_id, __n_iter, __wgroup_size, __acc, __n, __s_acc, __s_n, __comp, __found_local,
                           __brick_tag);
                    __item_id.barrier(sycl::access::fence_space::local_space);

                    // Set local atomic value to global atomic
                    if (__local_idx == 0 && __comp(__found_local.load(), __found.load()))
                    {
                        dpstd::__internal::__invoke_if_else(
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
    }

    return dpstd::__internal::__invoke_if_else(__or_tag_check, [&__result]() { return __result; },
                                               [&__result, &__first, &__last, &__init_value]() {
                                                   return __result != __init_value ? __first + __result : __last;
                                               });
}

//------------------------------------------------------------------------
// parallel_or
//------------------------------------------------------------------------

template <typename Name>
class __or_policy_wrapper
{
};

template <typename _ExecutionPolicy, typename _Iterator1, typename _Iterator2, typename _Brick>
dpstd::__internal::__enable_if_device_execution_policy<_ExecutionPolicy, bool>
__parallel_or(_ExecutionPolicy&& __exec, _Iterator1 __first, _Iterator1 __last, _Iterator2 __s_first,
              _Iterator2 __s_last, _Brick __f)
{
    return dpstd::__par_backend_hetero::__parallel_find_or(
        __par_backend_hetero::make_wrapped_policy<__or_policy_wrapper>(std::forward<_ExecutionPolicy>(__exec)), __first,
        __last, __s_first, __s_last, __f, __parallel_or_tag{});
}

// Special overload for single sequence cases.
// TODO: check if similar pattern may apply to other algorithms. If so, these overloads should be moved out of
// backend code.
template <typename _ExecutionPolicy, typename _Iterator, typename _Brick>
dpstd::__internal::__enable_if_device_execution_policy<_ExecutionPolicy, bool>
__parallel_or(_ExecutionPolicy&& __exec, _Iterator __first, _Iterator __last, _Brick __f)
{
    return dpstd::__par_backend_hetero::__parallel_or(std::forward<_ExecutionPolicy>(__exec), __first, __last,
                                                      /* dummy */ __first, __first, __f);
}

//------------------------------------------------------------------------
// parallel_find
//-----------------------------------------------------------------------

template <typename Name>
class __find_policy_wrapper
{
};

template <typename _ExecutionPolicy, typename _Iterator1, typename _Iterator2, typename _Brick, typename _IsFirst>
dpstd::__internal::__enable_if_device_execution_policy<_ExecutionPolicy, _Iterator1>
__parallel_find(_ExecutionPolicy&& __exec, _Iterator1 __first, _Iterator1 __last, _Iterator2 __s_first,
                _Iterator2 __s_last, _Brick __f, _IsFirst __is_first)
{
    using _TagType = typename std::conditional<_IsFirst::value, __parallel_find_forward_tag<_Iterator1>,
                                               __parallel_find_backward_tag<_Iterator1>>::type;
    return dpstd::__par_backend_hetero::__parallel_find_or(
        __par_backend_hetero::make_wrapped_policy<__find_policy_wrapper>(std::forward<_ExecutionPolicy>(__exec)),
        __first, __last, __s_first, __s_last, __f, _TagType{});
}

// Special overload for single sequence cases.
// TODO: check if similar pattern may apply to other algorithms. If so, these overloads should be moved out of
// backend code.
template <typename _ExecutionPolicy, typename _Iterator, typename _Brick, typename _IsFirst>
dpstd::__internal::__enable_if_device_execution_policy<_ExecutionPolicy, _Iterator>
__parallel_find(_ExecutionPolicy&& __exec, _Iterator __first, _Iterator __last, _Brick __f, _IsFirst __is_first)
{
    return dpstd::__par_backend_hetero::__parallel_find(std::forward<_ExecutionPolicy>(__exec), __first, __last,
                                                        /*dummy*/ __first, __first, __f, __is_first);
}

//------------------------------------------------------------------------
// parallel_merge
//-----------------------------------------------------------------------

struct __full_merge_kernel
{
    // this function is needed because it calls in different parallel patterns (parallel_merge, parallel_sort)
    template <typename _Idx, typename _Acc1, typename _Size1, typename _Acc2, typename _Size2, typename _Acc3,
              typename _Size3, typename _Compare>
    void
    operator()(_Idx __global_idx, const _Acc1& __in_acc1, _Size1 __start_1, _Size1 __end_1, const _Acc2& __in_acc2,
               _Size2 __start_2, _Size2 __end_2, const _Acc3& __out_acc, _Size3 __out_shift, _Compare __comp) const
    {
        // process 1st sequence
        if (__global_idx >= __start_1 && __global_idx < __end_1)
        {
            auto __shift_1 = __global_idx - __start_1;
            auto __shift_2 =
                dpstd::__internal::__pstl_lower_bound(__in_acc2, __start_2, __end_2, __in_acc1[__global_idx], __comp);
            __shift_2 -= __start_2;
            __out_acc[__out_shift + __shift_1 + __shift_2] = __in_acc1[__global_idx];
        }
        // process 2nd sequence
        if (__global_idx >= __start_2 && __global_idx < __end_2)
        {
            auto __shift_1 =
                dpstd::__internal::__pstl_upper_bound(__in_acc1, __start_1, __end_1, __in_acc2[__global_idx], __comp);
            __shift_1 -= __start_1;
            auto __shift_2 = __global_idx - __start_2;
            __out_acc[__out_shift + __shift_1 + __shift_2] = __in_acc2[__global_idx];
        }
    }
};

struct __full_merge_bunch_kernel
{
    // this function is needed because it calls in different parallel patterns (parallel_merge, parallel_sort)
    // and replacing with this function may affect performance for them.
    template <typename _Idx, typename _Acc1, typename _Size1, typename _Acc2, typename _Size2, typename _Acc3,
              typename _Size3, typename _Compare>
    void
    operator()(_Idx __global_idx, const _Acc1& __in_acc1, _Size1 __start_1, _Size1 __end_1, const _Acc2& __in_acc2,
               _Size2 __start_2, _Size2 __end_2, const _Acc3& __out_acc, _Size3 __out_shift, _Compare __comp,
               int __bunch_size) const
    {
        // a number at the end of a variable name imply a sequence is being processed
        auto __local_start_1 = __global_idx;
        auto __local_end_1 = __global_idx + __bunch_size;
        auto __local_start_2 = __global_idx;
        auto __local_end_2 = __global_idx + __bunch_size;
        auto __l_search_bound_1 = _Size1{};
        auto __r_search_bound_1 = _Size1{};
        auto __l_search_bound_2 = _Size2{};
        auto __r_search_bound_2 = _Size2{};

        // adjust bounds for processing
        if (__local_start_1 < __start_1)
            __local_start_1 = __start_1;
        if (__local_end_1 >= __end_1)
            __local_end_1 = __end_1;
        if (__local_start_2 < __start_2)
            __local_start_2 = __start_2;
        if (__local_end_2 >= __end_2)
            __local_end_2 = __end_2;

        // process 1st sequence
        if (__local_start_1 < __local_end_1)
        {
            // reduce a range of searching within the 2nd sequence and handle bound items
            const auto __leftmost_item_1 = __in_acc1[__local_start_1];
            __l_search_bound_2 =
                dpstd::__internal::__pstl_lower_bound(__in_acc2, __start_2, __end_2, __leftmost_item_1, __comp);
            const auto __l_shift_1 = __local_start_1 - __start_1;
            const auto __l_shift_2 = __l_search_bound_2 - __start_2;
            __out_acc[__out_shift + __l_shift_1 + __l_shift_2] = __leftmost_item_1;
            if (__local_end_1 - __local_start_1 > 1)
            {
                const auto __rightmost_item_1 = __in_acc1[__local_end_1 - 1];
                __r_search_bound_2 = dpstd::__internal::__pstl_lower_bound(__in_acc2, __l_search_bound_2, __end_2,
                                                                           __rightmost_item_1, __comp);
                const auto __r_shift_1 = __local_end_1 - 1 - __start_1;
                const auto __r_shift_2 = __r_search_bound_2 - __start_2;
                __out_acc[__out_shift + __r_shift_1 + __r_shift_2] = __rightmost_item_1;
            }

            // handle intermediate items
            for (auto __idx = __local_start_1 + 1; __idx < __local_end_1 - 1; ++__idx)
            {
                const auto __intermediate_item_1 = __in_acc1[__idx];
                __l_search_bound_2 = dpstd::__internal::__pstl_lower_bound(
                    __in_acc2, __l_search_bound_2, __r_search_bound_2, __intermediate_item_1, __comp);
                const auto __shift_1 = __idx - __start_1;
                const auto __shift_2 = __l_search_bound_2 - __start_2;
                __out_acc[__out_shift + __shift_1 + __shift_2] = __intermediate_item_1;
            }
        }
        // process 2nd sequence
        if (__local_start_2 < __local_end_2)
        {
            // reduce a range of searching within the 1st sequence and handle bound items
            const auto __leftmost_item_2 = __in_acc2[__local_start_2];
            __l_search_bound_1 =
                dpstd::__internal::__pstl_upper_bound(__in_acc1, __start_1, __end_1, __leftmost_item_2, __comp);
            const auto __l_shift_1 = __l_search_bound_1 - __start_1;
            const auto __l_shift_2 = __local_start_2 - __start_2;
            __out_acc[__out_shift + __l_shift_1 + __l_shift_2] = __leftmost_item_2;
            if (__local_end_2 - __local_start_2 > 1)
            {
                const auto __rightmost_item_2 = __in_acc2[__local_end_2 - 1];
                __r_search_bound_1 = dpstd::__internal::__pstl_upper_bound(__in_acc1, __l_search_bound_1, __end_1,
                                                                           __rightmost_item_2, __comp);
                const auto __r_shift_1 = __r_search_bound_1 - __start_1;
                const auto __r_shift_2 = __local_end_2 - 1 - __start_2;
                __out_acc[__out_shift + __r_shift_1 + __r_shift_2] = __rightmost_item_2;
            }

            // handle intermediate items
            for (auto __idx = __local_start_2 + 1; __idx < __local_end_2 - 1; ++__idx)
            {
                const auto __intermediate_item_2 = __in_acc2[__idx];
                __l_search_bound_1 = dpstd::__internal::__pstl_upper_bound(
                    __in_acc1, __l_search_bound_1, __r_search_bound_1, __intermediate_item_2, __comp);
                const auto __shift_1 = __l_search_bound_1 - __start_1;
                const auto __shift_2 = __idx - __start_2;
                __out_acc[__out_shift + __shift_1 + __shift_2] = __intermediate_item_2;
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
        const auto __part_end_1 = cl::sycl::min(__start_1 + __k, __end_1);
        const auto __part_end_2 = cl::sycl::min(__start_2 + __k, __end_2);

        // Handle elements from p1
        if (__global_idx >= __start_1 && __global_idx < __part_end_1)
        {
            const auto __shift =
                /* index inside p1 */ __global_idx - __start_1 +
                /* relative position in p3 */
                dpstd::__internal::__pstl_lower_bound(__in_acc2, __start_2, __part_end_2, __in_acc1[__global_idx],
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
                dpstd::__internal::__pstl_upper_bound(__in_acc1, __start_1, __part_end_1, __in_acc2[__global_idx],
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

template <typename _ExecutionPolicy, typename _Iterator1, typename _Iterator2, typename _Iterator3, typename _Compare>
dpstd::__internal::__enable_if_device_execution_policy<_ExecutionPolicy, _Iterator3>
__parallel_merge(_ExecutionPolicy&& __exec, _Iterator1 __first1, _Iterator1 __last1, _Iterator2 __first2,
                 _Iterator2 __last2, _Iterator3 __d_first, _Compare __comp)
{
    using _Policy = typename std::decay<_ExecutionPolicy>::type;
    using __kernel_name = typename _Policy::kernel_name;
#if __SYCL_UNNAMED_LAMBDA__
    using __kernel_1_name_t = __parallel_merge_kernel<_Iterator1, _Iterator2, _Iterator3, _Compare, __kernel_name>;
#else
    using __kernel_1_name_t = __parallel_merge_kernel<__kernel_name>;
#endif
    auto __n = __last1 - __first1;
    auto __n_2 = __last2 - __first2;
    if (__n == 0 && __n_2 == 0)
    {
        return __d_first;
    }
    auto __in_buffer1 = __internal::get_buffer()(__first1, __last1);
    auto __in_buffer2 = __internal::get_buffer()(__first2, __last2);
    auto __out_buffer = __internal::get_buffer()(__d_first, __d_first + __n + __n_2);
    _PRINT_INFO_IN_DEBUG_MODE(__exec);

    const int __bunch_size = __exec.queue().get_device().is_cpu() ? 128 : 8;
    const auto __max_n = std::max(__n, static_cast<decltype(__n)>(__n_2));
    const int __aux_step = (__max_n % __bunch_size) != 0;

    __exec.queue().submit([&](sycl::handler& __cgh) {
        auto __in_acc1 = __internal::get_access<_Iterator1>(__cgh)(__in_buffer1);
        auto __in_acc2 = __internal::get_access<_Iterator2>(__cgh)(__in_buffer2);
        auto __out_acc = __internal::get_access<_Iterator3>(__cgh)(__out_buffer);
        __cgh.parallel_for<__kernel_1_name_t>(
            sycl::range</*dim=*/1>(__max_n / __bunch_size + __aux_step), [=](sycl::item</*dim=*/1> __item_id) mutable {
                __full_merge_bunch_kernel()(__item_id.get_linear_id() * __bunch_size, __in_acc1, decltype(__n)(0), __n,
                                            __in_acc2, decltype(__n_2)(0), __n_2, __out_acc, decltype(__n)(0), __comp,
                                            __bunch_size);
            });
    });
    return __d_first + __n + __n_2;
}

//-----------------------------------------------------------------------
// parallel_sort: general implementation
//-----------------------------------------------------------------------

// Common sort routine used by full and partial sorting
// TODO: think about cut off
template <typename _ExecutionPolicy, typename _Iterator, typename _Merge, typename _Compare>
dpstd::__internal::__enable_if_device_execution_policy<_ExecutionPolicy, void>
__parallel_sort_impl(_ExecutionPolicy&& __exec, _Iterator __first, _Iterator __last, _Merge __merge, _Compare __comp)
{
    using _Policy = typename std::decay<_ExecutionPolicy>::type;
    using __kernel_name = typename _Policy::kernel_name;
#if __SYCL_UNNAMED_LAMBDA__
    using __kernel_1_name_t = __parallel_sort_kernel_1<_Iterator, _Merge, _Compare, __kernel_name>;
    using __kernel_2_name_t = __parallel_sort_kernel_2<_Iterator, _Merge, _Compare, __kernel_name>;
#else
    using __kernel_1_name_t = __parallel_sort_kernel_1<__kernel_name>;
    using __kernel_2_name_t = __parallel_sort_kernel_2<__kernel_name>;
#endif

    using _Tp = typename std::iterator_traits<_Iterator>::value_type;
    using _Size = typename std::iterator_traits<_Iterator>::difference_type;
    _Size __n = __last - __first;
    if (__n <= 1)
    {
        return;
    }
    auto __buffer = __internal::get_buffer()(__first, __last);
    dpstd::__par_backend_hetero::__internal::__buffer<_Policy, _Tp> __temp_buf(__exec, __n);
    auto __temp = __temp_buf.get_buffer();
    _PRINT_INFO_IN_DEBUG_MODE(__exec);

    _Size __k = 1;
    bool __data_in_temp = false;
    sycl::event __event1;
    do
    {
        __event1 = __exec.queue().submit([&](sycl::handler& __cgh) {
            __cgh.depends_on(__event1);
            auto __acc = __internal::get_access<_Iterator>(__cgh)(__buffer);
            auto __temp_acc = __temp.template get_access<read_write>(__cgh);
            __cgh.parallel_for<__kernel_1_name_t>(sycl::range</*dim=*/1>(__n),
                                                  [=](sycl::item</*dim=*/1> __item_id) mutable {
                                                      auto __global_idx = __item_id.get_linear_id();

                                                      _Size __start = 2 * __k * (__global_idx / (2 * __k));
                                                      _Size __end_1 = cl::sycl::min(__start + __k, __n);
                                                      _Size __end_2 = cl::sycl::min(__start + 2 * __k, __n);

                                                      if (!__data_in_temp)
                                                      {
                                                          __merge(__global_idx, __acc, __start, __end_1, __acc, __end_1,
                                                                  __end_2, __temp_acc, __start, __comp);
                                                      }
                                                      else
                                                      {
                                                          __merge(__global_idx, __temp_acc, __start, __end_1,
                                                                  __temp_acc, __end_1, __end_2, __acc, __start, __comp);
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
            auto __acc = __internal::get_access<_Iterator>(__cgh)(__buffer);
            auto __temp_acc = __temp.template get_access<read>(__cgh);
            // we cannot use __cgh.copy here because of zip_iterator usage
            __cgh.parallel_for<__kernel_2_name_t>(sycl::range</*dim=*/1>(__n),
                                                  [=](sycl::item</*dim=*/1> __item_id) mutable {
                                                      __acc[__item_id.get_linear_id()] = __temp_acc[__item_id];
                                                  });
        });
    }
}

//------------------------------------------------------------------------
// parallel_stable_sort
//-----------------------------------------------------------------------

template <typename _T, typename _Compare>
struct __is_radix_sort_usable_for_type
{
    static constexpr bool value =
#if _USE_SUB_GROUPS
        std::is_arithmetic<_T>::value && (__internal::__is_comp_ascending<__decay_t<_Compare>>::value ||
                                          __internal::__is_comp_descending<__decay_t<_Compare>>::value);
#else
        false;
#endif
};

#if _USE_SUB_GROUPS
template <typename _ExecutionPolicy, typename _Iterator, typename _Compare>
__enable_if_t<dpstd::__internal::__is_device_execution_policy<__decay_t<_ExecutionPolicy>>::value &&
                  __is_radix_sort_usable_for_type<__value_t<_Iterator>, _Compare>::value,
              void>
__parallel_stable_sort(_ExecutionPolicy&& __exec, _Iterator __first, _Iterator __last, _Compare __comp)
{
    __parallel_radix_sort<__internal::__is_comp_ascending<__decay_t<_Compare>>::value>(__exec, __first, __last);
}
#endif

template <typename _ExecutionPolicy, typename _Iterator, typename _Compare>
__enable_if_t<dpstd::__internal::__is_device_execution_policy<__decay_t<_ExecutionPolicy>>::value &&
                  !__is_radix_sort_usable_for_type<__value_t<_Iterator>, _Compare>::value,
              void>
__parallel_stable_sort(_ExecutionPolicy&& __exec, _Iterator __first, _Iterator __last, _Compare __comp)
{
    __parallel_sort_impl(std::forward<_ExecutionPolicy>(__exec), __first, __last,
                         // Pass special tag to choose 'full' merge subroutine at compile-time
                         __full_merge_kernel(), __comp);
}

//------------------------------------------------------------------------
// parallel_partial_sort
//-----------------------------------------------------------------------

// TODO: check if it makes sense to move these wrappers out of backend to a common place
template <typename _ExecutionPolicy, typename _Iterator, typename _Compare>
dpstd::__internal::__enable_if_device_execution_policy<_ExecutionPolicy, void>
__parallel_partial_sort(_ExecutionPolicy&& __exec, _Iterator __first, _Iterator __mid, _Iterator __last,
                        _Compare __comp)
{
    const auto __mid_idx = __mid - __first;
    __parallel_sort_impl(std::forward<_ExecutionPolicy>(__exec), __first, __last,
                         __partial_merge_kernel<decltype(__mid_idx)>{__mid_idx}, __comp);
}

} // namespace __par_backend_hetero
} // namespace dpstd

#endif /* _PSTL_parallel_backend_sycl_H */
