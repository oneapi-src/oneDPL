/*
 *  Copyright (c) 2020 Intel Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#ifndef async_reduce_impl_hpp
#define async_reduce_impl_hpp

//#include "oneapi/dpl/pstl/hetero/numeric_impl_hetero.h"

namespace oneapi
{

namespace dpl
{

namespace __par_backend_hetero
{

//template <typename _ExecPolicy>
using __future_with_tmps = oneapi::dpl::__internal::__future_with_tmps /*<_ExecPolicy>*/;

using __future_base = oneapi::dpl::__internal::__future_base;

//------------------------------------------------------------------------
// parallel_stable_sort - async pattern 2.0
//-----------------------------------------------------------------------

template <typename _ExecutionPolicy, typename _Iterator, typename _Merge, typename _Compare>
oneapi::dpl::__internal::__enable_if_async_execution_policy<_ExecutionPolicy, __future_with_tmps>
__parallel_sort_impl_async(_ExecutionPolicy&& __exec, _Iterator __first, _Iterator __last, _Merge __merge,
                           _Compare __comp)
{
    using _Policy = typename ::std::decay<_ExecutionPolicy>::type;
    using __kernel_name = typename _Policy::kernel_name;
#if __SYCL_UNNAMED_LAMBDA__
    using __kernel_1_name_t = __parallel_sort_kernel_1<_Iterator, _Merge, _Compare, __kernel_name>;
    using __kernel_2_name_t = __parallel_sort_kernel_2<_Iterator, _Merge, _Compare, __kernel_name>;
    using __kernel_3_name_t = __parallel_sort_kernel_3<_Iterator, _Merge, _Compare, __kernel_name>;
#else
    using __kernel_1_name_t = __parallel_sort_kernel_1<__kernel_name>;
    using __kernel_2_name_t = __parallel_sort_kernel_2<__kernel_name>;
    using __kernel_3_name_t = __parallel_sort_kernel_3<__kernel_name>;
#endif

    using _Tp = typename ::std::iterator_traits<_Iterator>::value_type;
    using _Size = typename ::std::iterator_traits<_Iterator>::difference_type;
    _Size __n = __last - __first;
    if (__n <= 1)
    {
        return __future_with_tmps(sycl::event{});
    }
    auto __buffer = __internal::get_buffer()(__first, __last);
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
        auto __acc = __internal::get_access<_Iterator>(__cgh)(__buffer);
        __cgh.parallel_for<__kernel_1_name_t>(sycl::range</*dim=*/1>(__leaf_steps),
                                              [=](sycl::item</*dim=*/1> __item_id) {
                                                  const _Size __idx = __item_id.get_linear_id() * __leaf;
                                                  const _Size __start = __idx;
                                                  const _Size __end = sycl::min(__start + __leaf, __n);
                                                  __leaf_sort_kernel()(__acc, __start, __end, __comp);
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
            auto __acc = __internal::get_access<_Iterator>(__cgh)(__buffer);
            auto __temp_acc = __temp.template get_access<access_mode::read_write>(__cgh);
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
                        __merge(__offset, __acc, __start_1, __end_1, __acc, __start_2, __end_2, __temp_acc, __start_1,
                                __comp, __chunk);
                    }
                    else
                    {
                        __merge(__offset, __temp_acc, __start_1, __end_1, __temp_acc, __start_2, __end_2, __acc,
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
            auto __acc =
                __internal::get_access<decltype(make_iter_mode<access_mode::write>(::std::declval<_Iterator>()))>(
                    __cgh)(__buffer);
            auto __temp_acc = __temp.template get_access<access_mode::read>(__cgh);
            // We cannot use __cgh.copy here because of zip_iterator usage
            __cgh.parallel_for<__kernel_3_name_t>(sycl::range</*dim=*/1>(__n), [=](sycl::item</*dim=*/1> __item_id) {
                __acc[__item_id.get_linear_id()] = __temp_acc[__item_id];
            });
        });
    }
    return __future_with_tmps(__event1, __temp);
}

template <typename _ExecutionPolicy, typename _Iterator, typename _Compare>
oneapi::dpl::__internal::__enable_if_async_execution_policy<_ExecutionPolicy, __future_with_tmps>
//__enable_if_t<oneapi::dpl::__internal::__is_async_execution_policy<__decay_t<_ExecutionPolicy>>::value &&
//                  !__is_radix_sort_usable_for_type<__value_t<_Iterator>, _Compare>::value,
//              __future<_ExecutionPolicy>>
__parallel_stable_sort_async(_ExecutionPolicy&& __exec, _Iterator __first, _Iterator __last, _Compare __comp)
{
    return  __parallel_sort_impl_async(::std::forward<_ExecutionPolicy>(__exec), __first, __last,
                               // Pass special tag to choose 'full' merge subroutine at compile-time
                               __full_merge_kernel(), __comp);
}

//------------------------------------------------------------------------
// parallel_for - async pattern 2.0
//------------------------------------------------------------------------

// General version of parallel_for, one additional parameter - __count of iterations of loop __cgh.parallel_for,
// for some algorithms happens that size of processing range is n, but amount of iterations is n/2.
template <typename _ExecutionPolicy, typename _Fp, typename _Index, typename... _Ranges>
oneapi::dpl::__internal::__enable_if_async_execution_policy<_ExecutionPolicy, __future_base>
__parallel_for_async(_ExecutionPolicy&& __exec, _Fp __brick, _Index __count, _Ranges&&... __rngs)
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
    auto __event = __exec.queue().submit([&__rngs..., &__brick, __count](sycl::handler& __cgh) {
        // get an access to data under SYCL buffer:
        oneapi::dpl::__ranges::__require_access(__cgh, __rngs...);

        __cgh.parallel_for<__kernel_name_t>(sycl::range</*dim=*/1>(__count), [=](sycl::item</*dim=*/1> __item_id) {
            auto __idx = __item_id.get_linear_id();
            __brick(__idx, __rngs...);
        });
    });
    return __future_base(__event);
}

//------------------------------------------------------------------------
// parallel_transform_reduce - async pattern 2.0
//------------------------------------------------------------------------

template <typename _Tp, typename _ExecutionPolicy, typename _Up, typename _Cp, typename _Rp, typename... _Ranges>
oneapi::dpl::__internal::__enable_if_device_execution_policy<
    _ExecutionPolicy, oneapi::dpl::__internal::__future<_Tp>>
__parallel_transform_reduce_async(_ExecutionPolicy&& __exec, _Up __u, _Cp __combine, _Rp __brick_reduce,
                                  _Ranges&&... __rngs)
{
    auto __n = __get_first_range(__rngs...).size();
    assert(__n > 0);

    using _Policy = typename ::std::decay<_ExecutionPolicy>::type;
    using __kernel_name = typename _Policy::kernel_name;
#if __SYCL_UNNAMED_LAMBDA__
    using __kernel_name_t = __parallel_reduce_kernel<_Up, _Cp, _Rp, __kernel_name, _Ranges...>;
#else
    using __kernel_name_t = __parallel_reduce_kernel<__kernel_name>;
#endif

    auto __wgroup_size = oneapi::dpl::__internal::__max_work_group_size(__exec);
    // change __wgroup_size according to local memory limit
    __wgroup_size = oneapi::dpl::__internal::__max_local_allocation_size<_ExecutionPolicy, _Tp>(
        ::std::forward<_ExecutionPolicy>(__exec), __wgroup_size);
#if _ONEDPL_COMPILE_KERNEL
    auto __kernel = __kernel_name_t::__compile_kernel(::std::forward<_ExecutionPolicy>(__exec));
    __wgroup_size = ::std::min(__wgroup_size, oneapi::dpl::__internal::__kernel_work_group_size(
                                                  ::std::forward<_ExecutionPolicy>(__exec), __kernel));
#endif
    auto __mcu = oneapi::dpl::__internal::__max_compute_units(::std::forward<_ExecutionPolicy>(__exec));
    auto __n_groups = (__n - 1) / __wgroup_size + 1;
    __n_groups = ::std::min(decltype(__n_groups)(__mcu), __n_groups);
    // TODO: try to change __n_groups with another formula for more perfect load balancing

    _PRINT_INFO_IN_DEBUG_MODE(__exec, __wgroup_size, __mcu);

    // Create temporary global buffers to store temporary values
    auto __temp_1 = sycl::buffer<_Tp>(sycl::range<1>(__n_groups));
    auto __temp_2 = sycl::buffer<_Tp>(sycl::range<1>(__n_groups));
    // __is_first == true. Reduce over each work_group
    // __is_first == false. Reduce between work groups
    bool __is_first = true;
    auto __buf_1_ptr = &__temp_1; // __buf_1_ptr is not accessed on the device when __is_first == true
    auto __buf_2_ptr = &__temp_1;
    sycl::event __reduce_event;
    do
    {
        __reduce_event = __exec.queue().submit([&](sycl::handler& __cgh) {
            __cgh.depends_on(__reduce_event);

            oneapi::dpl::__ranges::__require_access(__cgh, __rngs...); // get an access to data under SYCL buffer
            auto __temp_1_acc = __buf_1_ptr->template get_access<access_mode::read_write>(__cgh);
            auto __temp_2_acc = __buf_2_ptr->template get_access<access_mode::write>(__cgh);
            sycl::accessor<_Tp, 1, access_mode::read_write, sycl::access::target::local> __temp_local(
                sycl::range<1>(__wgroup_size), __cgh);
            __cgh.parallel_for<__kernel_name_t>(
#if _ONEDPL_COMPILE_KERNEL
                __kernel,
#endif
                sycl::nd_range<1>(sycl::range<1>(__n_groups * __wgroup_size), sycl::range<1>(__wgroup_size)),
                [=](sycl::nd_item<1> __item_id) {
                    auto __global_idx = __item_id.get_global_id(0);
                    auto __local_idx = __item_id.get_local_id(0);
                    // 1. Initialization (transform part). Fill local memory
                    if (__is_first)
                    {
                        __u(__item_id, __global_idx, __n, __temp_local, __rngs...);
                    }
                    else
                    {
                        if (__global_idx < __n)
                            __temp_local[__local_idx] = __temp_1_acc[__global_idx];
                        __item_id.barrier(sycl::access::fence_space::local_space);
                    }
                    // 2. Reduce within work group using local memory
                    auto __res = __brick_reduce(__item_id, __global_idx, __n, __temp_local);
                    if (__local_idx == 0)
                    {
                        __temp_2_acc[__item_id.get_group(0)] = __res;
                    }
                });
        });
        if (__is_first)
        {
            __buf_2_ptr = &__temp_2;
            __is_first = false;
        }
        else
        {
            ::std::swap(__buf_1_ptr, __buf_2_ptr);
        }
        __n = __n_groups;
        __n_groups = (__n - 1) / __wgroup_size + 1;
    } while (__n > 1);

    // point of syncronization (on host access)
    // return __buf_1_ptr->template get_access<access_mode::read>()[0];
    return ::oneapi::dpl::__internal::__future<_Tp>( __reduce_event, *__buf_1_ptr, *__buf_2_ptr);
}

} // namespace __par_backend_hetero

namespace __internal
{

// Async pattern overloads:

// From: oneDPL/include/oneapi/dpl/pstl/hetero/algorithm_impl_hetero.h

template <typename _ExecutionPolicy, typename _ForwardIterator, typename _Function>
oneapi::dpl::__internal::__enable_if_async_execution_policy<_ExecutionPolicy,
                                                            oneapi::dpl::__internal::__future_base>
__pattern_walk1_async(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last, _Function __f,
                      /*vector=*/::std::true_type, /*parallel=*/::std::true_type)
{
    auto __n = __last - __first;
    if (__n <= 0)
        return oneapi::dpl::__internal::__future_base(sycl::event{});

    auto __keep =
        oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read_write, _ForwardIterator>();
    auto __buf = __keep(__first, __last);

    auto __future_obj = oneapi::dpl::__par_backend_hetero::__parallel_for_async(
        ::std::forward<_ExecutionPolicy>(__exec), unseq_backend::walk_n<_ExecutionPolicy, _Function>{__f}, __n,
        __buf.all_view());
    // TODO: Pass correct return value;
    return __future_obj; // oneapi::dpl::__internal::__future<_ExecutionPolicy>(__exec);
}

// TODO: A tag _IsSync is used for provide a patterns call pipeline, where the last one should be synchronous
// Probably it should be re-designed by a pipeline approach, when a patern returns some sync obejects
// and ones are combined into a "pipeline" (probably like Range pipeline)
template <typename _IsSync = ::std::true_type,
          __par_backend_hetero::access_mode __acc_mode1 = __par_backend_hetero::access_mode::read,
          __par_backend_hetero::access_mode __acc_mode2 = __par_backend_hetero::access_mode::write,
          typename _ExecutionPolicy, typename _ForwardIterator1, typename _ForwardIterator2, typename _Function>
oneapi::dpl::__internal::__enable_if_async_execution_policy<
    _ExecutionPolicy, oneapi::dpl::__internal::__future<_ForwardIterator2>>
__pattern_walk2_async(_ExecutionPolicy&& __exec, _ForwardIterator1 __first1, _ForwardIterator1 __last1,
                      _ForwardIterator2 __first2, _Function __f, /*vector=*/::std::true_type,
                      /*parallel=*/::std::true_type)
{
    auto __n = __last1 - __first1;
    if (__n <= 0)
        return oneapi::dpl::__internal::__future<_ForwardIterator2>( sycl::event{}, __first2);

    auto __keep1 = oneapi::dpl::__ranges::__get_sycl_range<__acc_mode1, _ForwardIterator1>();
    auto __buf1 = __keep1(__first1, __last1);

    auto __keep2 = oneapi::dpl::__ranges::__get_sycl_range<__acc_mode2, _ForwardIterator2>();
    auto __buf2 = __keep2(__first2, __first2 + __n);

    auto __future_obj = oneapi::dpl::__par_backend_hetero::__parallel_for(
        ::std::forward<_ExecutionPolicy>(__exec), unseq_backend::walk_n<_ExecutionPolicy, _Function>{__f}, __n,
        __buf1.all_view(), __buf2.all_view());
    oneapi::dpl::__internal::__invoke_if(_IsSync(), [&__future_obj]() { __future_obj.wait(); });

    return oneapi::dpl::__internal::__future<_ForwardIterator2>(__future_obj, __first2 + __n);
}

template <typename _ExecutionPolicy, typename _ForwardIterator1, typename _ForwardIterator2, typename _Brick>
oneapi::dpl::__internal::__enable_if_async_execution_policy<
    _ExecutionPolicy, oneapi::dpl::__internal::__future<_ForwardIterator2>>
__pattern_walk2_brick_async(_ExecutionPolicy&& __exec, _ForwardIterator1 __first1, _ForwardIterator1 __last1,
                            _ForwardIterator2 __first2, _Brick __brick, /*parallel*/ ::std::true_type)
{
    return __pattern_walk2_async(
        __par_backend_hetero::make_wrapped_policy<__walk2_brick_wrapper>(::std::forward<_ExecutionPolicy>(__exec)),
        __first1, __last1, __first2, __brick,
        /*vector=*/::std::true_type{}, /*parallel*/ ::std::true_type{});
}

//------------------------------------------------------------------------
// transform_reduce (with unary and binary functions)
//------------------------------------------------------------------------

template <typename _ExecutionPolicy, typename _ForwardIterator, typename _Tp, typename _BinaryOperation,
          typename _UnaryOperation>
oneapi::dpl::__internal::__enable_if_async_execution_policy<
    _ExecutionPolicy,
    //_Tp>
    oneapi::dpl::__internal::__future<_Tp>>
__pattern_transform_reduce_async(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last,
                                 _Tp __init, _BinaryOperation __binary_op, _UnaryOperation __unary_op,
                                 /*vector=*/::std::true_type,
                                 /*parallel=*/::std::true_type)
{
    // TODO: Empty sequence -> RETURN VALUE!
    // if (__first == __last)
    //    return ::oneapi::dpl::__internal::__make_future_promise_pair(
    //    ::oneapi::dpl::__internal::__future<_ExecutionPolicy>{__exec}, __init );//__init;

    using _Policy = _ExecutionPolicy;
    using _Functor = unseq_backend::walk_n<_Policy, _UnaryOperation>;
    using _RepackedTp = __par_backend_hetero::__repacked_tuple_t<_Tp>;

    auto __keep = oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read, _ForwardIterator>();
    auto __buf = __keep(__first, __last);
    //_RepackedTp
    auto __res = oneapi::dpl::__par_backend_hetero::__parallel_transform_reduce_async<_RepackedTp>(
        ::std::forward<_ExecutionPolicy>(__exec),
        unseq_backend::transform_init<_Policy, _BinaryOperation, _Functor>{__binary_op,
                                                                           _Functor{__unary_op}}, // transform
        __binary_op,                                                                              // combine
        unseq_backend::reduce<_Policy, _BinaryOperation, _RepackedTp>{__binary_op},               // reduce
        __buf.all_view());
    // TODO: add binary op and initial value to future obj
    // return value holds single element buffer, event and tempories.
    return oneapi::dpl::__internal::__future<_Tp>(__res, __init, __binary_op);
    // return __res; //__binary_op(__init, _Tp{__res});
}

//------------------------------------------------------------------------
// sort
//------------------------------------------------------------------------
template <typename _ExecutionPolicy, typename _Iterator, typename _Compare>
oneapi::dpl::__internal::__enable_if_async_execution_policy<
    _ExecutionPolicy, oneapi::dpl::__internal::__future_with_tmps>
__pattern_sort_async(_ExecutionPolicy&& __exec, _Iterator __first, _Iterator __last, _Compare __comp,
                     /*vector=*/::std::true_type, /*parallel=*/::std::true_type,
                     /*is_move_constructible=*/::std::true_type)
{
    auto ret_val = __par_backend_hetero::__parallel_stable_sort_async(
        ::std::forward<_ExecutionPolicy>(__exec),
        __par_backend_hetero::make_iter_mode<__par_backend_hetero::access_mode::read_write>(__first),
        __par_backend_hetero::make_iter_mode<__par_backend_hetero::access_mode::read_write>(__last), __comp);
    return ret_val; // oneapi::dpl::__internal::__future<_ExecutionPolicy>{__exec};
}

} // namespace __internal

namespace async
{

// [alg.async.transform]
template <class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2, class _UnaryOperation>
oneapi::dpl::__internal::__enable_if_async_execution_policy<
    _ExecutionPolicy, oneapi::dpl::__internal::__future<_ForwardIterator2>>
transform(_ExecutionPolicy&& __exec, _ForwardIterator1 __first, _ForwardIterator1 __last, _ForwardIterator2 __result,
          _UnaryOperation __op)
{
    auto ret_val = oneapi::dpl::__internal::__pattern_walk2_async(
        ::std::forward<_ExecutionPolicy>(__exec), __first, __last, __result,
        oneapi::dpl::__internal::__invoke_unary_op<_UnaryOperation>{::std::move(__op)},
        oneapi::dpl::__internal::__is_vectorization_preferred<_ExecutionPolicy, _ForwardIterator1, _ForwardIterator2>(
            __exec),
        __exec.__allow_parallel());
    return ret_val; // oneapi::dpl::__internal::__future<_ExecutionPolicy,_ForwardIterator2>{__exec, ret_val};
}

// [alg.async.copy]
template <class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2>
oneapi::dpl::__internal::__enable_if_async_execution_policy<
    _ExecutionPolicy, oneapi::dpl::__internal::__future<_ForwardIterator2>>
copy(_ExecutionPolicy&& __exec, _ForwardIterator1 __first, _ForwardIterator1 __last, _ForwardIterator2 __result)
{
    auto ret_val = oneapi::dpl::__internal::__pattern_walk2_brick_async(
        ::std::forward<_ExecutionPolicy>(__exec), __first, __last, __result,
        oneapi::dpl::__internal::__brick_copy<_ExecutionPolicy>{},
        oneapi::dpl::__internal::__is_parallelization_preferred<_ExecutionPolicy, _ForwardIterator1, _ForwardIterator2>(
            __exec));
    return ret_val; // oneapi::dpl::__internal::__future<_ExecutionPolicy,_ForwardIterator2>{__exec, ret_val};
}

// [alg.async.sort]
template <class _ExecutionPolicy, class _RandomAccessIterator, class _Compare>
oneapi::dpl::__internal::__enable_if_async_execution_policy<
    _ExecutionPolicy, oneapi::dpl::__internal::__future_with_tmps>
sort(_ExecutionPolicy&& __exec, _RandomAccessIterator __first, _RandomAccessIterator __last, _Compare __comp)
{
    // Calls oneDPL/include/oneapi/dpl/pstl/hetero/algorithm_impl_hetero.h
    typedef typename ::std::iterator_traits<_RandomAccessIterator>::value_type _InputType;
    auto ret_val = oneapi::dpl::__internal::__pattern_sort_async(
        ::std::forward<_ExecutionPolicy>(__exec), __first, __last, __comp,
        oneapi::dpl::__internal::__is_vectorization_preferred<_ExecutionPolicy, _RandomAccessIterator>(__exec),
        oneapi::dpl::__internal::__is_parallelization_preferred<_ExecutionPolicy, _RandomAccessIterator>(__exec),
        typename ::std::is_move_constructible<_InputType>::type());
    return ret_val; // oneapi::dpl::__internal::__future<_ExecutionPolicy>{__exec};
}

// [alg.async.foreach]
template <class _ExecutionPolicy, class _ForwardIterator, class _Function>
oneapi::dpl::__internal::__enable_if_async_execution_policy<_ExecutionPolicy,
                                                            oneapi::dpl::__internal::__future_base>
for_each(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last, _Function __f)
{
    auto ret_val = oneapi::dpl::__internal::__pattern_walk1_async(
        ::std::forward<_ExecutionPolicy>(__exec), __first, __last, __f,
        oneapi::dpl::__internal::__is_vectorization_preferred<_ExecutionPolicy, _ForwardIterator>(__exec),
        __exec.__allow_parallel());
    return ret_val; // oneapi::dpl::__internal::__future<_ExecutionPolicy>{__exec};
}

// [alg.async.reduce]
template <class _ExecutionPolicy, class _ForwardIterator, class _Tp, class _BinaryOperation>
oneapi::dpl::__internal::__enable_if_async_execution_policy<
    _ExecutionPolicy, oneapi::dpl::__internal::__future<_Tp>>
reduce(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last, _Tp __init,
       _BinaryOperation __binary_op)
{
    // return transform_reduce(::std::forward<_ExecutionPolicy>(__exec), __first, __last, __init, __binary_op,
    //                        oneapi::dpl::__internal::__no_op());
    typedef typename ::std::iterator_traits<_ForwardIterator>::value_type _InputType;
    auto ret_val = oneapi::dpl::__internal::__pattern_transform_reduce_async(
        ::std::forward<_ExecutionPolicy>(__exec), __first, __last, __init, ::std::plus<_InputType>(),
        oneapi::dpl::__internal::__no_op(), //::std::multiplies<_InputType>(),
        oneapi::dpl::__internal::__is_vectorization_preferred<_ExecutionPolicy, _ForwardIterator>(__exec),
        oneapi::dpl::__internal::__is_parallelization_preferred<_ExecutionPolicy, _ForwardIterator>(__exec));
    // return oneapi::dpl::__internal::__future<_ExecutionPolicy, _Tp>{__exec, ret_val};
    // return
    // oneapi::dpl::__internal::__make_future_promise_pair(::oneapi::dpl::__internal::__future<_ExecutionPolicy>{__exec},
    // ret_val);
    return ret_val;
}

#if 0
template<class _ExecutionPolicy, class _ForwardIterator, class T>
//oneapi::dpl::__internal::__enable_if_async_execution_policy<Policy,
//oneapi::dpl::__internal::__future_promise_pair<Policy, T>>
//reduce(Policy&& exec, Iter __first, Iter __last)
//template <class _ExecutionPolicy, class _ForwardIterator>
oneapi::dpl::__internal::__enable_if_async_execution_policy<_ExecutionPolicy,
    oneapi::dpl::__internal::__future_promise_pair<_ExecutionPolicy, typename ::std::iterator_traits<_ForwardIterator>::value_type>>
reduce(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last) {
    using _Tp = typename ::std::iterator_traits<_ForwardIterator>::value_type;
    return oneapi::dpl::async::reduce(std::forward<_ExecutionPolicy>(__exec), __first, __last, T{0}, std::plus<T>{});
}
#endif

} // namespace async

} // namespace dpl
} // namespace oneapi

#endif /* async_reduce_impl_hpp */
