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

#ifndef _ONEDPL_ASYNC_EXTENSION_IMPL_H
#define _ONEDPL_ASYNC_EXTENSION_IMPL_H

namespace oneapi
{
namespace dpl
{
namespace __par_backend_hetero
{

//------------------------------------------------------------------------
// parallel_transform_reduce - async pattern 2.0
//------------------------------------------------------------------------

template <typename _Tp, typename _ExecutionPolicy, typename _Up, typename _Cp, typename _Rp, typename... _Ranges>
oneapi::dpl::__internal::__enable_if_device_execution_policy<_ExecutionPolicy, oneapi::dpl::__internal::__future<_Tp>>
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
    return ::oneapi::dpl::__internal::__future<_Tp>(__reduce_event, *__buf_1_ptr, *__buf_2_ptr);
}

} // namespace __par_backend_hetero

namespace __internal
{

// Async pattern overloads:

template <typename _ExecutionPolicy, typename _ForwardIterator, typename _Function>
oneapi::dpl::__internal::__enable_if_async_execution_policy<_ExecutionPolicy,
                                                            oneapi::dpl::__par_backend_hetero::__future<void>>
__pattern_walk1_async(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last, _Function __f,
                      /*vector=*/::std::true_type, /*parallel=*/::std::true_type)
{
    auto __n = __last - __first;
    if (__n <= 0)
        return oneapi::dpl::__par_backend_hetero::__future<void>(sycl::event{});

    auto __keep =
        oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read_write, _ForwardIterator>();
    auto __buf = __keep(__first, __last);

    auto __future_obj = oneapi::dpl::__par_backend_hetero::__parallel_for /*_async*/ (
        ::std::forward<_ExecutionPolicy>(__exec), unseq_backend::walk_n<_ExecutionPolicy, _Function>{__f}, __n,
        __buf.all_view());
    return __future_obj;
}

template <typename _IsSync = ::std::false_type,
          __par_backend_hetero::access_mode __acc_mode1 = __par_backend_hetero::access_mode::read,
          __par_backend_hetero::access_mode __acc_mode2 = __par_backend_hetero::access_mode::write,
          typename _ExecutionPolicy, typename _ForwardIterator1, typename _ForwardIterator2, typename _Function>
oneapi::dpl::__internal::__enable_if_async_execution_policy<_ExecutionPolicy,
                                                            oneapi::dpl::__internal::__future<_ForwardIterator2>>
__pattern_walk2_async(_ExecutionPolicy&& __exec, _ForwardIterator1 __first1, _ForwardIterator1 __last1,
                      _ForwardIterator2 __first2, _Function __f, /*vector=*/::std::true_type,
                      /*parallel=*/::std::true_type)
{
    auto __n = __last1 - __first1;
    if (__n <= 0)
        return oneapi::dpl::__internal::__future<_ForwardIterator2>(sycl::event{}, __first2);

    auto __keep1 = oneapi::dpl::__ranges::__get_sycl_range<__acc_mode1, _ForwardIterator1>();
    auto __buf1 = __keep1(__first1, __last1);

    auto __keep2 = oneapi::dpl::__ranges::__get_sycl_range<__acc_mode2, _ForwardIterator2>();
    auto __buf2 = __keep2(__first2, __first2 + __n);

    auto __future_obj = oneapi::dpl::__par_backend_hetero::__parallel_for /*_async*/ (
        ::std::forward<_ExecutionPolicy>(__exec), unseq_backend::walk_n<_ExecutionPolicy, _Function>{__f}, __n,
        __buf1.all_view(), __buf2.all_view());
    oneapi::dpl::__internal::__invoke_if(_IsSync(), [&__future_obj]() { __future_obj.wait(); });

    return oneapi::dpl::__internal::__future<_ForwardIterator2>(__future_obj, __first2 + __n);
}

template <typename _ExecutionPolicy, typename _ForwardIterator1, typename _ForwardIterator2, typename _ForwardIterator3,
          typename _Function>
oneapi::dpl::__internal::__enable_if_async_execution_policy<_ExecutionPolicy,
                                                            oneapi::dpl::__internal::__future<_ForwardIterator3>>
__pattern_walk3_async(_ExecutionPolicy&& __exec, _ForwardIterator1 __first1, _ForwardIterator1 __last1,
                      _ForwardIterator2 __first2, _ForwardIterator3 __first3, _Function __f,
                      /*vector=*/::std::true_type,
                      /*parallel=*/::std::true_type)
{
    auto __n = __last1 - __first1;
    if (__n <= 0)
        return oneapi::dpl::__internal::__future<_ForwardIterator3>(sycl::event{}, __first3);

    auto __keep1 =
        oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read, _ForwardIterator1>();
    auto __buf1 = __keep1(__first1, __last1);
    auto __keep2 =
        oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read, _ForwardIterator2>();
    auto __buf2 = __keep2(__first2, __first2 + __n);
    auto __keep3 =
        oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::write, _ForwardIterator3>();
    auto __buf3 = __keep3(__first3, __first3 + __n);

    auto __future_obj = oneapi::dpl::__par_backend_hetero::__parallel_for /*_async*/ (
        ::std::forward<_ExecutionPolicy>(__exec), unseq_backend::walk_n<_ExecutionPolicy, _Function>{__f}, __n,
        __buf1.all_view(), __buf2.all_view(), __buf3.all_view());

    return oneapi::dpl::__internal::__future<_ForwardIterator3>(__future_obj, __first3 + __n);
}

template <typename _ExecutionPolicy, typename _ForwardIterator1, typename _ForwardIterator2, typename _Brick>
oneapi::dpl::__internal::__enable_if_async_execution_policy<_ExecutionPolicy,
                                                            oneapi::dpl::__internal::__future<_ForwardIterator2>>
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
oneapi::dpl::__internal::__enable_if_async_execution_policy<_ExecutionPolicy, oneapi::dpl::__internal::__future<_Tp>>
__pattern_transform_reduce_async(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last,
                                 _Tp __init, _BinaryOperation __binary_op, _UnaryOperation __unary_op,
                                 /*vector=*/::std::true_type,
                                 /*parallel=*/::std::true_type)
{
    if (__first == __last)
        return oneapi::dpl::__internal::__future<_Tp>(sycl::event{}, __init);

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
    return oneapi::dpl::__internal::__future<_Tp>(__res, __init, __binary_op);
}

template <typename _ExecutionPolicy, typename _ForwardIterator, typename _T>
oneapi::dpl::__internal::__enable_if_async_execution_policy<_ExecutionPolicy,
                                                            oneapi::dpl::__par_backend_hetero::__future<void>>
__pattern_fill_async(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last, const _T& __value,
                     /*vector=*/::std::true_type, /*parallel=*/::std::true_type)
{
    auto ret_val =
        __pattern_walk1_async(::std::forward<_ExecutionPolicy>(__exec),
                              __par_backend_hetero::make_iter_mode<__par_backend_hetero::access_mode::write>(__first),
                              __par_backend_hetero::make_iter_mode<__par_backend_hetero::access_mode::write>(__last),
                              fill_functor<_T>{__value}, ::std::true_type{}, ::std::true_type{});
    return ret_val;
}

namespace async
{

// [async.transform]
template <class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2, class _UnaryOperation>
oneapi::dpl::__internal::__enable_if_async_execution_policy<_ExecutionPolicy,
                                                            oneapi::dpl::__internal::__future<_ForwardIterator2>>
transform(_ExecutionPolicy&& __exec, _ForwardIterator1 __first, _ForwardIterator1 __last, _ForwardIterator2 __result,
          _UnaryOperation __op)
{
    auto ret_val = oneapi::dpl::__internal::__pattern_walk2_async(
        ::std::forward<_ExecutionPolicy>(__exec), __first, __last, __result,
        oneapi::dpl::__internal::__invoke_unary_op<_UnaryOperation>{::std::move(__op)},
        oneapi::dpl::__internal::__is_vectorization_preferred<_ExecutionPolicy, _ForwardIterator1, _ForwardIterator2>(
            __exec),
        __exec.__allow_parallel());
    return ret_val;
}

template <class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2, class _ForwardIterator,
          class _BinaryOperation>
oneapi::dpl::__internal::__enable_if_async_execution_policy<_ExecutionPolicy,
                                                            oneapi::dpl::__internal::__future<_ForwardIterator>>
transform(_ExecutionPolicy&& __exec, _ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2,
          _ForwardIterator __result, _BinaryOperation __op)
{
    auto ret_val = oneapi::dpl::__internal::__pattern_walk3_async(
        ::std::forward<_ExecutionPolicy>(__exec), __first1, __last1, __first2, __result,
        oneapi::dpl::__internal::__transform_functor<
            oneapi::dpl::__internal::__ref_or_copy<_ExecutionPolicy, _BinaryOperation>>(__op),
        oneapi::dpl::__internal::__is_vectorization_preferred<_ExecutionPolicy, _ForwardIterator1, _ForwardIterator2,
                                                              _ForwardIterator>(__exec),
        __exec.__allow_parallel());
    return ret_val;
}

// [async.copy]
template <class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2>
oneapi::dpl::__internal::__enable_if_async_execution_policy<_ExecutionPolicy,
                                                            oneapi::dpl::__internal::__future<_ForwardIterator2>>
copy(_ExecutionPolicy&& __exec, _ForwardIterator1 __first, _ForwardIterator1 __last, _ForwardIterator2 __result)
{
    auto ret_val = oneapi::dpl::__internal::__pattern_walk2_brick_async(
        ::std::forward<_ExecutionPolicy>(__exec), __first, __last, __result,
        oneapi::dpl::__internal::__brick_copy<_ExecutionPolicy>{},
        oneapi::dpl::__internal::__is_parallelization_preferred<_ExecutionPolicy, _ForwardIterator1, _ForwardIterator2>(
            __exec));
    return ret_val;
}

// [async.sort]
template <class _ExecutionPolicy, class _RandomAccessIterator, class _Compare>
oneapi::dpl::__internal::__enable_if_async_execution_policy<_ExecutionPolicy,
                                                            oneapi::dpl::__par_backend_hetero::__future<void>>
sort(_ExecutionPolicy&& __exec, _RandomAccessIterator __first, _RandomAccessIterator __last, _Compare __comp)
{
    auto ret_val = __par_backend_hetero::__parallel_stable_sort(
        ::std::forward<_ExecutionPolicy>(__exec),
        __par_backend_hetero::make_iter_mode<__par_backend_hetero::access_mode::read_write>(__first),
        __par_backend_hetero::make_iter_mode<__par_backend_hetero::access_mode::read_write>(__last), __comp);
    return ret_val;
}

// [async.for_each]
template <class _ExecutionPolicy, class _ForwardIterator, class _Function>
oneapi::dpl::__internal::__enable_if_async_execution_policy<_ExecutionPolicy,
                                                            oneapi::dpl::__par_backend_hetero::__future<void>>
for_each(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last, _Function __f)
{
    auto ret_val = oneapi::dpl::__internal::__pattern_walk1_async(
        ::std::forward<_ExecutionPolicy>(__exec), __first, __last, __f,
        oneapi::dpl::__internal::__is_vectorization_preferred<_ExecutionPolicy, _ForwardIterator>(__exec),
        __exec.__allow_parallel());
    return ret_val;
}

// [async.reduce]

template <class _ExecutionPolicy, class _ForwardIterator, class _Tp, class _BinaryOperation>
oneapi::dpl::__internal::__enable_if_async_execution_policy<_ExecutionPolicy, oneapi::dpl::__internal::__future<_Tp>>
reduce(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last, _Tp __init,
       _BinaryOperation __binary_op)
{
    typedef typename ::std::iterator_traits<_ForwardIterator>::value_type _InputType;
    auto ret_val = oneapi::dpl::__internal::__pattern_transform_reduce_async(
        ::std::forward<_ExecutionPolicy>(__exec), __first, __last, __init, ::std::plus<_InputType>(),
        oneapi::dpl::__internal::__no_op(),
        oneapi::dpl::__internal::__is_vectorization_preferred<_ExecutionPolicy, _ForwardIterator>(__exec),
        oneapi::dpl::__internal::__is_parallelization_preferred<_ExecutionPolicy, _ForwardIterator>(__exec));
    return ret_val;
}

// [async.fill]

template <class _ExecutionPolicy, class _ForwardIterator, class _Tp>
oneapi::dpl::__internal::__enable_if_async_execution_policy<_ExecutionPolicy,
                                                            oneapi::dpl::__par_backend_hetero::__future<void>>
fill(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last, const _Tp& __value)
{
    return oneapi::dpl::__internal::__pattern_fill_async(
        ::std::forward<_ExecutionPolicy>(__exec), __first, __last, __value,
        oneapi::dpl::__internal::__is_parallelization_preferred<_ExecutionPolicy, _ForwardIterator>(__exec),
        oneapi::dpl::__internal::__is_vectorization_preferred<_ExecutionPolicy, _ForwardIterator>(__exec));
}

// [async.transform_reduce]

template <class _ExecutionPolicy, class _ForwardIt1, class _ForwardIt2, class _T, class _BinaryOp1, class _BinaryOp2>
oneapi::dpl::__internal::__enable_if_async_execution_policy<_ExecutionPolicy, oneapi::dpl::__internal::__future<_T>>
transform_reduce(_ExecutionPolicy&& __exec, _ForwardIt1 __first1, _ForwardIt1 __last1, _ForwardIt2 __first2, _T __init,
                 _BinaryOp1 __binary_op1, _BinaryOp2 __binary_op2)
{
    return oneapi::dpl::__internal::__pattern_transform_reduce_async(
        ::std::forward<_ExecutionPolicy>(__exec), __first1, __last1, __first2, __init, __binary_op1, __binary_op2,
        oneapi::dpl::__internal::__is_vectorization_preferred<_ExecutionPolicy, _ForwardIt1, _ForwardIt2>(__exec),
        oneapi::dpl::__internal::__is_parallelization_preferred<_ExecutionPolicy, _ForwardIt1, _ForwardIt2>(__exec));
}

template <class _ExecutionPolicy, class _ForwardIt, class _T, class _BinaryOp, class _UnaryOp>
oneapi::dpl::__internal::__enable_if_async_execution_policy<_ExecutionPolicy, oneapi::dpl::__internal::__future<_T>>
transform_reduce(_ExecutionPolicy&& __exec, _ForwardIt __first, _ForwardIt __last, _T __init, _BinaryOp __binary_op,
                 _UnaryOp __unary_op)
{
    return oneapi::dpl::__internal::__pattern_transform_reduce_async(
        ::std::forward<_ExecutionPolicy>(__exec), __first, __last, __init, __binary_op, __unary_op,
        oneapi::dpl::__internal::__is_vectorization_preferred<_ExecutionPolicy, _ForwardIt>(__exec),
        oneapi::dpl::__internal::__is_parallelization_preferred<_ExecutionPolicy, _ForwardIt>(__exec));
}

} // namespace async

} // namespace __internal

namespace experimental
{

template <typename... _Ts>
void
wait_for_all(const _Ts&... __Events)
{
    ::std::vector<sycl::event> __wait_list = {__Events...};
    for (auto _a : __wait_list)
        _a.wait();
}

template <class _ExecutionPolicy, class InputIter, class OutputIter, class... _Events>
oneapi::dpl::__internal::__enable_if_async_execution_policy<_ExecutionPolicy,
                                                            oneapi::dpl::__internal::__future<OutputIter>>
copy_async(_ExecutionPolicy&& __exec, InputIter __input_first, InputIter __input_last, OutputIter __output_first,
           _Events&&... __dependencies)
{
    wait_for_all(__dependencies...);
    return __internal::async::copy(std::forward<_ExecutionPolicy>(__exec), __input_first, __input_last, __output_first);
}

template <class _ExecutionPolicy, class InputIter, class UnaryFunction, class... _Events>
oneapi::dpl::__internal::__enable_if_async_execution_policy<_ExecutionPolicy,
                                                            oneapi::dpl::__par_backend_hetero::__future<void>>
for_each_async(_ExecutionPolicy&& __exec, InputIter __first, InputIter __last, UnaryFunction __f,
               _Events&&... __dependencies)
{
    wait_for_all(__dependencies...);
    return __internal::async::for_each(std::forward<_ExecutionPolicy>(__exec), __first, __last, __f);
}

template <class _ExecutionPolicy, class _ForwardIterator, class _Tp, class _BinaryOperation, class... _Events>
oneapi::dpl::__internal::__enable_if_async_execution_policy<_ExecutionPolicy, oneapi::dpl::__internal::__future<_Tp>,
                                                            _Events...>
reduce_async(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last, _Tp __init,
             _BinaryOperation __binary_op, _Events&&... __dependencies)
{
    wait_for_all(__dependencies...);
    return __internal::async::reduce(std::forward<_ExecutionPolicy>(__exec), __first, __last, __init, __binary_op);
}

template <class _ExecutionPolicy, class _ForwardIt1, class _ForwardIt2, class _UnaryOperation, class... _Events>
oneapi::dpl::__internal::__enable_if_async_execution_policy<_ExecutionPolicy,
                                                            oneapi::dpl::__internal::__future<_ForwardIt2>, _Events...>
transform_async(_ExecutionPolicy&& __exec, _ForwardIt1 first1, _ForwardIt1 last1, _ForwardIt2 d_first,
                _UnaryOperation unary_op, _Events&&... __dependencies)
{
    wait_for_all(__dependencies...);
    return __internal::async::transform(std::forward<_ExecutionPolicy>(__exec), first1, last1, d_first, unary_op);
}

template <class _ExecutionPolicy, class _ForwardIt1, class _ForwardIt2, class _ForwardIt3, class _BinaryOperation,
          class... _Events>
oneapi::dpl::__internal::__enable_if_async_execution_policy_single_no_default<
    _ExecutionPolicy, oneapi::dpl::__internal::__future<_ForwardIt3>, _BinaryOperation, _Events...>
transform_async(_ExecutionPolicy&& __exec, _ForwardIt1 first1, _ForwardIt1 last1, _ForwardIt2 first2,
                _ForwardIt3 d_first, _BinaryOperation binary_op, _Events&&... __dependencies)
{
    wait_for_all(__dependencies...);
    return __internal::async::transform(std::forward<_ExecutionPolicy>(__exec), first1, last1, first2, d_first,
                                        binary_op);
}

template <class _ExecutionPolicy, class _ForwardIt1, class _ForwardIt2, class _T, class _BinaryOp1, class _BinaryOp2,
          class... _Events>
oneapi::dpl::__internal::__enable_if_async_execution_policy_double_no_default<
    _ExecutionPolicy, oneapi::dpl::__internal::__future<_T>, _BinaryOp1, _BinaryOp2, _Events...>
transform_reduce_async(_ExecutionPolicy&& __exec, _ForwardIt1 __first1, _ForwardIt1 __last1, _ForwardIt2 __first2,
                       _T __init, _BinaryOp1 __binary_op1, _BinaryOp2 __binary_op2, _Events&&... __dependencies)
{
    wait_for_all(__dependencies...);
    return __internal::async::transform_reduce(std::forward<_ExecutionPolicy>(__exec), __first1, __last1, __first2,
                                               __init, __binary_op1, __binary_op2);
}

template <class _ExecutionPolicy, class _ForwardIt1, class _ForwardIt2, class _T, class... _Events>
oneapi::dpl::__internal::__enable_if_async_execution_policy<_ExecutionPolicy, oneapi::dpl::__internal::__future<_T>,
                                                            _Events...>
transform_reduce_async(_ExecutionPolicy&& __exec, _ForwardIt1 __first1, _ForwardIt1 __last1, _ForwardIt2 __first2,
                       _T __init, _Events&&... __dependencies)
{
    wait_for_all(__dependencies...);
    return __internal::async::transform_reduce(std::forward<_ExecutionPolicy>(__exec), __first1, __last1, __first2,
                                               __init, ::std::plus<>(), ::std::multiplies<>());
}

template <class _ExecutionPolicy, class _ForwardIt, class _T, class _BinaryOp, class _UnaryOp, class... _Events>
oneapi::dpl::__internal::__enable_if_async_execution_policy_single_no_default<
    _ExecutionPolicy, oneapi::dpl::__internal::__future<_T>, _UnaryOp, _Events...>
transform_reduce_async(_ExecutionPolicy&& __exec, _ForwardIt __first, _ForwardIt __last, _T __init,
                       _BinaryOp __binary_op, _UnaryOp __unary_op, _Events&&... __dependencies)
{
    wait_for_all(__dependencies...);
    return __internal::async::transform_reduce(std::forward<_ExecutionPolicy>(__exec), __first, __last, __init,
                                               __binary_op, __unary_op);
}

template <class _ExecutionPolicy, class _RandomAccessIterator, class... _Events>
oneapi::dpl::__internal::__enable_if_async_execution_policy<
    _ExecutionPolicy, oneapi::dpl::__par_backend_hetero::__future<void>, _Events...>
sort_async(_ExecutionPolicy&& __exec, _RandomAccessIterator __first, _RandomAccessIterator __last,
           _Events&&... __dependencies)
{
    using __T = typename ::std::iterator_traits<_RandomAccessIterator>::value_type;
    wait_for_all(__dependencies...);
    return __internal::async::sort(std::forward<_ExecutionPolicy>(__exec), __first, __last, ::std::less<__T>());
}

template <class _ExecutionPolicy, class _RandomAccessIterator, class _Compare, class... _Events>
oneapi::dpl::__internal::__enable_if_async_execution_policy_single_no_default<
    _ExecutionPolicy, oneapi::dpl::__par_backend_hetero::__future<void>, _Compare, _Events...>
sort_async(_ExecutionPolicy&& __exec, _RandomAccessIterator __first, _RandomAccessIterator __last, _Compare __comp,
           _Events&&... __dependencies)
{
    wait_for_all(__dependencies...);
    return __internal::async::sort(std::forward<_ExecutionPolicy>(__exec), __first, __last, __comp);
}

template <class _ExecutionPolicy, class _ForwardIterator, class _Tp, class... _Events>
oneapi::dpl::__internal::__enable_if_async_execution_policy<
    _ExecutionPolicy, oneapi::dpl::__par_backend_hetero::__future<void>, _Events...>
fill_async(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last, const _Tp& __value,
           _Events&&... __dependencies)
{
    wait_for_all(__dependencies...);
    return __internal::async::fill(std::forward<_ExecutionPolicy>(__exec), __first, __last, __value);
}

} // namespace experimental

} // namespace dpl

} // namespace oneapi

#endif /* _ONEDPL_ASYNC_EXTENSION_IMPL_H */
