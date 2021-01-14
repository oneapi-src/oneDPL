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

#ifndef _ONEDPL_ASYNC_HPP
#define _ONEDPL_ASYNC_HPP

#include <CL/sycl.hpp>

#include "future.hpp"

namespace oneapi
{
namespace dpl
{
namespace __internal
{

template <typename _T>
struct __is_async_execution_policy : ::std::false_type
{
};

template <typename _ExecPolicy, typename _T, typename... _Events>
using __enable_if_async_execution_policy = typename ::std::enable_if<
    oneapi::dpl::__internal::__is_device_execution_policy<typename ::std::decay<_ExecPolicy>::type>::value && ( true && ... && ::std::is_convertible_v<_Events,event> ) , _T>::type;

template <typename _ExecPolicy, typename _T, typename _Op1, typename... _Events>
using __enable_if_async_execution_policy_single_no_default = typename ::std::enable_if<
    oneapi::dpl::__internal::__is_device_execution_policy<typename ::std::decay<_ExecPolicy>::type>::value && !::std::is_convertible_v<_Op1,event> && ( true && ... && ::std::is_convertible_v<_Events,event> ) , _T>::type;

template <typename _ExecPolicy, typename _T, typename _Op1, typename _Op2, typename... _Events>
using __enable_if_async_execution_policy_double_no_default = typename ::std::enable_if<
    oneapi::dpl::__internal::__is_device_execution_policy<typename ::std::decay<_ExecPolicy>::type>::value && !::std::is_convertible_v<_Op1,event> && !::std::is_convertible_v<_Op2,event> && ( true && ... && ::std::is_convertible_v<_Events,event> ) , _T>::type;

namespace async
{

template <class _ExecutionPolicy, class InputIter, class OutputIter>
oneapi::dpl::__internal::__enable_if_async_execution_policy<
    _ExecutionPolicy, oneapi::dpl::__internal::__future<OutputIter>>
copy(_ExecutionPolicy&& __exec, InputIter __input_first, InputIter __input_last, OutputIter __output_first);

template <class _ExecutionPolicy, class _ForwardIterator, class _Tp, class _BinaryOperation>
oneapi::dpl::__internal::__enable_if_async_execution_policy<
    _ExecutionPolicy, oneapi::dpl::__internal::__future<_Tp>>
reduce(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last, _Tp __init,
       _BinaryOperation __binary_op);

template <class _ExecutionPolicy, class InputIter, class UnaryFunction>
oneapi::dpl::__internal::__enable_if_async_execution_policy<_ExecutionPolicy, oneapi::dpl::__internal::__future<void>>
for_each(_ExecutionPolicy&& __exec, InputIter __first, InputIter __last, UnaryFunction __f);

template <class _ExecutionPolicy, class _RandomAccessIterator, class _Compare>
oneapi::dpl::__internal::__enable_if_async_execution_policy<
    _ExecutionPolicy, oneapi::dpl::__internal::__future<void>>
sort(_ExecutionPolicy&& __exec, _RandomAccessIterator __first, _RandomAccessIterator __last, _Compare __comp);

template <class _ExecutionPolicy, class _ForwardIt1, class _ForwardIt2, class _UnaryOperation>
oneapi::dpl::__internal::__enable_if_async_execution_policy<
    _ExecutionPolicy, oneapi::dpl::__internal::__future<_ForwardIt2>>
transform(_ExecutionPolicy&& policy, _ForwardIt1 first1, _ForwardIt1 last1, _ForwardIt2 d_first, _UnaryOperation unary_op);

template< class _ExecutionPolicy, class _ForwardIt1, class _ForwardIt2, class _ForwardIt3, class _BinaryOperation >
oneapi::dpl::__internal::__enable_if_async_execution_policy<
    _ExecutionPolicy, oneapi::dpl::__internal::__future<_ForwardIt3>>
transform( _ExecutionPolicy&& policy, _ForwardIt1 first1, _ForwardIt1 last1,
                    _ForwardIt2 first2, _ForwardIt3 d_first, _BinaryOperation binary_op );

template <class _ExecutionPolicy, class _ForwardIterator, class _Tp>
oneapi::dpl::__internal::__enable_if_async_execution_policy<
    _ExecutionPolicy, oneapi::dpl::__internal::__future<void>>
fill(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last, const _Tp& __value);

// merge();

} // namespace async

} // namespace __internal

// Public API for asynch algorithms:

template <typename... _Ts>
void
wait_for_all(const _Ts&... __Events) {
    ::std::vector<__internal::event> __wait_list = {__Events...};
    for(auto _a : __wait_list) _a.wait();
}

template <class _ExecutionPolicy, class InputIter, class OutputIter, class... _Events>
oneapi::dpl::__internal::__enable_if_async_execution_policy<
    _ExecutionPolicy, oneapi::dpl::__internal::__future<OutputIter>>
copy_async(_ExecutionPolicy&& __exec, InputIter __input_first, InputIter __input_last, OutputIter __output_first, _Events&&... __dependencies) {
    wait_for_all(__dependencies...);
    return __internal::async::copy(std::forward<_ExecutionPolicy>(__exec), __input_first, __input_last, __output_first);
}

template <class _ExecutionPolicy, class InputIter, class UnaryFunction, class... _Events>
oneapi::dpl::__internal::__enable_if_async_execution_policy<_ExecutionPolicy, oneapi::dpl::__internal::__future<void>>
for_each_async(_ExecutionPolicy&& __exec, InputIter __first, InputIter __last, UnaryFunction __f,  _Events&&... __dependencies) {
    wait_for_all(__dependencies...);
    return __internal::async::for_each(std::forward<_ExecutionPolicy>(__exec), __first, __last, __f);
}

template <class _ExecutionPolicy, class _ForwardIterator, class _Tp, class _BinaryOperation, class... _Events>
oneapi::dpl::__internal::__enable_if_async_execution_policy<
    _ExecutionPolicy, oneapi::dpl::__internal::__future<_Tp>, _Events...>
reduce_async(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last, _Tp __init, _BinaryOperation __binary_op, _Events&&... __dependencies) {
    wait_for_all(__dependencies...);
    return __internal::async::reduce(std::forward<_ExecutionPolicy>(__exec), __first, __last, __init, __binary_op);
}

template <class _ExecutionPolicy, class _ForwardIt1, class _ForwardIt2, class _UnaryOperation, class... _Events>
oneapi::dpl::__internal::__enable_if_async_execution_policy<
    _ExecutionPolicy, oneapi::dpl::__internal::__future<_ForwardIt2>, _Events...>
transform_async(_ExecutionPolicy&& __exec, _ForwardIt1 first1, _ForwardIt1 last1, _ForwardIt2 d_first, _UnaryOperation unary_op, _Events&&... __dependencies) {
    wait_for_all(__dependencies...);
    return __internal::async::transform( std::forward<_ExecutionPolicy>(__exec), first1, last1, d_first, unary_op ); 
}

template <class _ExecutionPolicy, class _ForwardIt1, class _ForwardIt2, class _ForwardIt3, class _BinaryOperation, class... _Events>
oneapi::dpl::__internal::__enable_if_async_execution_policy_single_no_default<
    _ExecutionPolicy, oneapi::dpl::__internal::__future<_ForwardIt3>, _BinaryOperation, _Events...>
transform_async(_ExecutionPolicy&& __exec, _ForwardIt1 first1, _ForwardIt1 last1, _ForwardIt2 first2, _ForwardIt3 d_first , _BinaryOperation binary_op, _Events&&... __dependencies) {
    wait_for_all(__dependencies...);
    return __internal::async::transform( std::forward<_ExecutionPolicy>(__exec), first1, last1, first2, d_first, binary_op );
}

template <class _ExecutionPolicy, class _RandomAccessIterator, class... _Events>
oneapi::dpl::__internal::__enable_if_async_execution_policy<
    _ExecutionPolicy, oneapi::dpl::__internal::__future<void>, _Events...>
sort_async(_ExecutionPolicy&& __exec, _RandomAccessIterator __first, _RandomAccessIterator __last, _Events&&... __dependencies) {
    using __T = typename ::std::iterator_traits<_RandomAccessIterator>::value_type;
    wait_for_all(__dependencies...);
    return __internal::async::sort( std::forward<_ExecutionPolicy>(__exec), __first, __last, ::std::less<__T>() );
}

template <class _ExecutionPolicy, class _RandomAccessIterator, class _Compare, class... _Events>
oneapi::dpl::__internal::__enable_if_async_execution_policy_single_no_default<
    _ExecutionPolicy, oneapi::dpl::__internal::__future<void>, _Compare, _Events...>
sort_async(_ExecutionPolicy&& __exec, _RandomAccessIterator __first, _RandomAccessIterator __last, _Compare __comp, _Events&&... __dependencies) {
    wait_for_all(__dependencies...);
    return __internal::async::sort( std::forward<_ExecutionPolicy>(__exec), __first, __last, __comp );
}

template <class _ExecutionPolicy, class _ForwardIterator, class _Tp, class... _Events>
oneapi::dpl::__internal::__enable_if_async_execution_policy<
    _ExecutionPolicy, oneapi::dpl::__internal::__future<void>, _Events...>
fill_async(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last, const _Tp& __value, _Events&&... __dependencies) {
    wait_for_all(__dependencies...);
    return __internal::async::fill( std::forward<_ExecutionPolicy>(__exec), __first, __last, __value );
}

} // namespace dpl

} // namespace oneapi

#include "async_impl.hpp"

#endif
