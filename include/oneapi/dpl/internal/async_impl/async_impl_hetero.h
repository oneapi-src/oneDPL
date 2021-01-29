/*
 *  Copyright (c) 2020-2021 Intel Corporation
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

#ifndef _ONEDPL_ASYNC_IMPL_HETERO_H
#define _ONEDPL_ASYNC_IMPL_HETERO_H

#if _ONEDPL_BACKEND_SYCL
#    include "async_backend_sycl.h"
#endif

namespace oneapi
{
namespace dpl
{
namespace __internal
{

template <typename _ExecutionPolicy, typename _ForwardIterator, typename _Function>
oneapi::dpl::__internal::__enable_if_async_execution_policy<_ExecutionPolicy,
                                                            oneapi::dpl::__par_backend_hetero::__future<void>>
__pattern_walk1_async(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last, _Function __f)
{
    auto __n = __last - __first;
    if (__n <= 0)
        return oneapi::dpl::__par_backend_hetero::__future<void>(sycl::event{});

    auto __keep =
        oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read_write, _ForwardIterator>();
    auto __buf = __keep(__first, __last);

    auto __future_obj = oneapi::dpl::__par_backend_hetero::__parallel_for(
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
                      _ForwardIterator2 __first2, _Function __f)
{
    auto __n = __last1 - __first1;
    if (__n <= 0)
        return oneapi::dpl::__internal::__future<_ForwardIterator2>(sycl::event{}, __first2);

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

template <typename _ExecutionPolicy, typename _ForwardIterator1, typename _ForwardIterator2, typename _ForwardIterator3,
          typename _Function>
oneapi::dpl::__internal::__enable_if_async_execution_policy<_ExecutionPolicy,
                                                            oneapi::dpl::__internal::__future<_ForwardIterator3>>
__pattern_walk3_async(_ExecutionPolicy&& __exec, _ForwardIterator1 __first1, _ForwardIterator1 __last1,
                      _ForwardIterator2 __first2, _ForwardIterator3 __first3, _Function __f)
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

    auto __future_obj = oneapi::dpl::__par_backend_hetero::__parallel_for(
        ::std::forward<_ExecutionPolicy>(__exec), unseq_backend::walk_n<_ExecutionPolicy, _Function>{__f}, __n,
        __buf1.all_view(), __buf2.all_view(), __buf3.all_view());

    return oneapi::dpl::__internal::__future<_ForwardIterator3>(__future_obj, __first3 + __n);
}

template <typename _ExecutionPolicy, typename _ForwardIterator1, typename _ForwardIterator2, typename _Brick>
oneapi::dpl::__internal::__enable_if_async_execution_policy<_ExecutionPolicy,
                                                            oneapi::dpl::__internal::__future<_ForwardIterator2>>
__pattern_walk2_brick_async(_ExecutionPolicy&& __exec, _ForwardIterator1 __first1, _ForwardIterator1 __last1,
                            _ForwardIterator2 __first2, _Brick __brick)
{
    return __pattern_walk2_async(
        __par_backend_hetero::make_wrapped_policy<__walk2_brick_wrapper>(::std::forward<_ExecutionPolicy>(__exec)),
        __first1, __last1, __first2, __brick);
}

//------------------------------------------------------------------------
// transform_reduce (version with two binary functions)
//------------------------------------------------------------------------

template <typename _ExecutionPolicy, typename _RandomAccessIterator1, typename _RandomAccessIterator2, typename _Tp,
          typename _BinaryOperation1, typename _BinaryOperation2>
oneapi::dpl::__internal::__enable_if_async_execution_policy<_ExecutionPolicy, oneapi::dpl::__internal::__future<_Tp>>
__pattern_transform_reduce_async(_ExecutionPolicy&& __exec, _RandomAccessIterator1 __first1,
                                 _RandomAccessIterator1 __last1, _RandomAccessIterator2 __first2, _Tp __init,
                                 _BinaryOperation1 __binary_op1, _BinaryOperation2 __binary_op2)
{
    if (__first1 == __last1)
        return oneapi::dpl::__internal::__future<_Tp>(sycl::event{}, __init);

    using _Policy = _ExecutionPolicy;
    using _Functor = unseq_backend::walk_n<_Policy, _BinaryOperation2>;
    using _RepackedTp = __par_backend_hetero::__repacked_tuple_t<_Tp>;

    auto __n = __last1 - __first1;
    auto __keep1 =
        oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read, _RandomAccessIterator1>();
    auto __buf1 = __keep1(__first1, __last1);
    auto __keep2 =
        oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read, _RandomAccessIterator2>();
    auto __buf2 = __keep2(__first2, __first2 + __n);

    auto __res = oneapi::dpl::__par_backend_hetero::__parallel_transform_reduce_async<_RepackedTp>(
        ::std::forward<_ExecutionPolicy>(__exec),
        unseq_backend::transform_init<_Policy, _BinaryOperation1, _Functor>{__binary_op1,
                                                                            _Functor{__binary_op2}}, // transform
        __binary_op1,                                                                                // combine
        unseq_backend::reduce<_Policy, _BinaryOperation1, _RepackedTp>{__binary_op1},                // reduce
        __buf1.all_view(), __buf2.all_view());

    return oneapi::dpl::__internal::__future<_Tp>(__res, __init, __binary_op1);
}

//------------------------------------------------------------------------
// transform_reduce (with unary and binary functions)
//------------------------------------------------------------------------

template <typename _ExecutionPolicy, typename _ForwardIterator, typename _Tp, typename _BinaryOperation,
          typename _UnaryOperation>
oneapi::dpl::__internal::__enable_if_async_execution_policy<_ExecutionPolicy, oneapi::dpl::__internal::__future<_Tp>>
__pattern_transform_reduce_async(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last,
                                 _Tp __init, _BinaryOperation __binary_op, _UnaryOperation __unary_op)
{
    if (__first == __last)
        return oneapi::dpl::__internal::__future<_Tp>(sycl::event{}, __init);

    using _Policy = _ExecutionPolicy;
    using _Functor = unseq_backend::walk_n<_Policy, _UnaryOperation>;
    using _RepackedTp = __par_backend_hetero::__repacked_tuple_t<_Tp>;

    auto __keep = oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read, _ForwardIterator>();
    auto __buf = __keep(__first, __last);

    auto __res = oneapi::dpl::__par_backend_hetero::__parallel_transform_reduce_async<_RepackedTp>(
        ::std::forward<_ExecutionPolicy>(__exec),
        unseq_backend::transform_init<_Policy, _BinaryOperation, _Functor>{__binary_op,
                                                                           _Functor{__unary_op}}, // transform
        __binary_op,                                                                              // combine
        unseq_backend::reduce<_Policy, _BinaryOperation, _RepackedTp>{__binary_op},               // reduce
        __buf.all_view());
    return oneapi::dpl::__internal::__future<_Tp>(::std::forward<oneapi::dpl::__internal::__future<_Tp>>(__res), __init,
                                                  __binary_op);
}

template <typename _ExecutionPolicy, typename _ForwardIterator, typename _T>
oneapi::dpl::__internal::__enable_if_async_execution_policy<_ExecutionPolicy,
                                                            oneapi::dpl::__par_backend_hetero::__future<void>>
__pattern_fill_async(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last, const _T& __value)
{
    auto ret_val =
        __pattern_walk1_async(::std::forward<_ExecutionPolicy>(__exec),
                              __par_backend_hetero::make_iter_mode<__par_backend_hetero::access_mode::write>(__first),
                              __par_backend_hetero::make_iter_mode<__par_backend_hetero::access_mode::write>(__last),
                              fill_functor<_T>{__value});
    return ret_val;
}

} // namespace __internal
} // namespace dpl
} // namespace oneapi

#endif /* _ONEDPL_ASYNC_IMPL_HETERO_H */
