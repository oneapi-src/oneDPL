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

#ifndef _ONEDPL_ASYNC_EXTENSION_DEFS_H
#define _ONEDPL_ASYNC_EXTENSION_DEFS_H

#include <CL/sycl.hpp>

namespace oneapi
{
namespace dpl
{
namespace __internal
{

using sycl::event;

template <typename _T, typename _Tp = sycl_iterator<sycl::access::mode::read_write, _T, sycl::buffer_allocator>>
struct __async_value
{
    virtual _T
    data() = 0;
    virtual ~__async_value() = default;
    virtual _Tp
    raw_data() const;
};

template <typename _T>
struct __async_init : public __async_value<_T>
{
    _T __data;
    __async_init(_T _d) : __data(_d) {}
    _T
    data()
    {
        return __data;
    }
    virtual sycl_iterator<sycl::access::mode::read_write, _T, sycl::buffer_allocator>
    raw_data() const
    {
        return oneapi::dpl::begin(sycl::buffer<_T>{&this->__data, 1});
    }
};

template <typename _Tp, typename _T = typename std::iterator_traits<_Tp>::value_type>
struct __async_direct : public __async_value<_T>
{
    _Tp __data;
    __async_direct(_Tp _d) : __data(_d) {}
    _T
    data()
    {
        return __data.get_buffer().template get_access<access_mode::read>()[0];
    }
    virtual sycl_iterator<sycl::access::mode::read_write, _T, sycl::buffer_allocator>
    raw_data() const
    {
        return __data;
    }
};

template <typename _Tp, typename _Op = ::std::plus<_Tp>, typename _Buf = sycl::buffer<_Tp>>
struct __async_transform : public __async_value<_Tp>
{
    _Buf __buf;
    _Op __op;
    _Tp __init;
    __async_transform(_Buf _b, _Tp _v = 0, _Op _o = std::plus<_Tp>()) : __buf(_b), __op(_o), __init(_v) {}
    _Tp
    data()
    {
        auto ret_val = __buf.template get_access<access_mode::read>()[0];
        return __op(ret_val, __init);
    }
    virtual sycl_iterator<sycl::access::mode::read_write, _Tp, sycl::buffer_allocator>
    raw_data() const
    {
        return oneapi::dpl::begin(__buf);
    }
};

template <typename _Tp>
class __future : public __par_backend_hetero::__future_base
{
    ::std::unique_ptr<__async_value<_Tp>> __data; // This is a value/buffer for read access!
    __par_backend_hetero::__tmp_base __tmp;

  public:
    template <typename... _Ts>
    __future(sycl::event __e, sycl::buffer<_Tp> __d, _Ts... __t)
        : __par_backend_hetero::__future_base(__e),
          __data(::std::unique_ptr<__async_transform<_Tp>>(new __async_transform<_Tp>(__d))),
          __tmp(__par_backend_hetero::__TempObjs<_Ts...>{__t...})
    {
    }
    // Constructor for reduce_transform pattern
    template <typename _Op>
    __future(const __future<_Tp>& _fp, _Tp __i, _Op __o)
        : __par_backend_hetero::__future_base(_fp.get_event()),
          __data(::std::unique_ptr<__async_transform<_Tp, _Op>>(
              new __async_transform<_Tp, _Op>(_fp.raw_data().get_buffer(), __i, __o))),
          __tmp(_fp.__tmp)
    {
    }
    __future(sycl::event __e, _Tp __i)
        : __par_backend_hetero::__future_base(__e),
          __data(::std::unique_ptr<__async_init<_Tp>>(new __async_init<_Tp>(__i)))
    {
    }
    // Return underlying buffer
    auto
    raw_data() const
    {
        return __data->raw_data();
    }
    _Tp
    get()
    {
        this->wait();
        return __data->data();
    }
};

// Specialization for sycl_iterator
template <typename T>
class __future<sycl_iterator<sycl::access::mode::read_write, T, sycl::buffer_allocator>>
    : public __par_backend_hetero::__future_base
{
    using _Tp = sycl_iterator<sycl::access::mode::read_write, T, sycl::buffer_allocator>;
    ::std::unique_ptr<__async_value<T>> __data;
    __par_backend_hetero::__tmp_base __tmp;

  public:
    template <typename... _Ts>
    __future(sycl::event __e, _Tp __d, _Ts... __t)
        : __future_base(__e), __data(::std::unique_ptr<__async_direct<_Tp>>(new __async_direct<_Tp>(__d))),
          __tmp(__par_backend_hetero::__TempObjs<_Ts...>{__t...})
    {
    }
    _Tp
    get()
    {
        this->wait();
        return __data->data();
    }
};

template <typename _T>
struct __is_async_execution_policy : ::std::false_type
{
};

template <typename _ExecPolicy, typename _T, typename... _Events>
using __enable_if_async_execution_policy = typename ::std::enable_if<
    oneapi::dpl::__internal::__is_device_execution_policy<typename ::std::decay<_ExecPolicy>::type>::value &&
        (true && ... && ::std::is_convertible_v<_Events, event>),
    _T>::type;

template <typename _ExecPolicy, typename _T, typename _Op1, typename... _Events>
using __enable_if_async_execution_policy_single_no_default = typename ::std::enable_if<
    oneapi::dpl::__internal::__is_device_execution_policy<typename ::std::decay<_ExecPolicy>::type>::value &&
        !::std::is_convertible_v<_Op1, event> && (true && ... && ::std::is_convertible_v<_Events, event>),
    _T>::type;

template <typename _ExecPolicy, typename _T, typename _Op1, typename _Op2, typename... _Events>
using __enable_if_async_execution_policy_double_no_default = typename ::std::enable_if<
    oneapi::dpl::__internal::__is_device_execution_policy<typename ::std::decay<_ExecPolicy>::type>::value &&
        !::std::is_convertible_v<_Op1, event> && !::std::is_convertible_v<_Op2, event> &&
        (true && ... && ::std::is_convertible_v<_Events, event>),
    _T>::type;

} // namespace __internal

// Public API for asynch algorithms:
namespace experimental
{

template <typename... _Ts>
void
wait_for_all(const _Ts&... __Events);

template <class _ExecutionPolicy, class InputIter, class OutputIter, class... _Events>
oneapi::dpl::__internal::__enable_if_async_execution_policy<_ExecutionPolicy,
                                                            oneapi::dpl::__internal::__future<OutputIter>>
copy_async(_ExecutionPolicy&& __exec, InputIter __input_first, InputIter __input_last, OutputIter __output_first,
           _Events&&... __dependencies);

template <class _ExecutionPolicy, class InputIter, class UnaryFunction, class... _Events>
oneapi::dpl::__internal::__enable_if_async_execution_policy<_ExecutionPolicy,
                                                            oneapi::dpl::__par_backend_hetero::__future<void>>
for_each_async(_ExecutionPolicy&& __exec, InputIter __first, InputIter __last, UnaryFunction __f,
               _Events&&... __dependencies);

template <class _ExecutionPolicy, class _ForwardIt, class... _Events>
oneapi::dpl::__internal::__enable_if_async_execution_policy<
    _ExecutionPolicy, oneapi::dpl::__internal::__future<typename std::iterator_traits<_ForwardIt>::value_type>,
    _Events...>
reduce_async(_ExecutionPolicy&& __exec, _ForwardIt __first, _ForwardIt __last, _Events&&... __dependencies);

template <class _ExecutionPolicy, class _ForwardIt, class _T, class... _Events>
oneapi::dpl::__internal::__enable_if_async_execution_policy_single_no_default<
    _ExecutionPolicy, oneapi::dpl::__internal::__future<_T>, _T, _Events...>
reduce_async(_ExecutionPolicy&& __exec, _ForwardIt __first, _ForwardIt __last, _T init, _Events&&... __dependencies);

template <class _ExecutionPolicy, class _ForwardIterator, class _Tp, class _BinaryOperation, class... _Events>
oneapi::dpl::__internal::__enable_if_async_execution_policy_double_no_default<
    _ExecutionPolicy, oneapi::dpl::__internal::__future<_Tp>, _Tp, _BinaryOperation, _Events...>
reduce_async(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last, _Tp __init,
             _BinaryOperation __binary_op, _Events&&... __dependencies);

template <class _ExecutionPolicy, class _ForwardIt1, class _ForwardIt2, class _UnaryOperation, class... _Events>
oneapi::dpl::__internal::__enable_if_async_execution_policy<_ExecutionPolicy,
                                                            oneapi::dpl::__internal::__future<_ForwardIt2>, _Events...>
transform_async(_ExecutionPolicy&& __exec, _ForwardIt1 first1, _ForwardIt1 last1, _ForwardIt2 d_first,
                _UnaryOperation unary_op, _Events&&... __dependencies);

template <class _ExecutionPolicy, class _ForwardIt1, class _ForwardIt2, class _ForwardIt3, class _BinaryOperation,
          class... _Events>
oneapi::dpl::__internal::__enable_if_async_execution_policy_single_no_default<
    _ExecutionPolicy, oneapi::dpl::__internal::__future<_ForwardIt3>, _BinaryOperation, _Events...>
transform_async(_ExecutionPolicy&& __exec, _ForwardIt1 first1, _ForwardIt1 last1, _ForwardIt2 first2,
                _ForwardIt3 d_first, _BinaryOperation binary_op, _Events&&... __dependencies);

template <class _ExecutionPolicy, class _ForwardIt1, class _ForwardIt2, class _T, class _BinaryOp1, class _BinaryOp2,
          class... _Events>
oneapi::dpl::__internal::__enable_if_async_execution_policy_double_no_default<
    _ExecutionPolicy, oneapi::dpl::__internal::__future<_T>, _BinaryOp1, _BinaryOp2, _Events...>
transform_reduce_async(_ExecutionPolicy&& __exec, _ForwardIt1 __first1, _ForwardIt1 __last1, _ForwardIt2 __first2,
                       _T __init, _BinaryOp1 __binary_op1, _BinaryOp2 __binary_op2, _Events&&... __dependencies);

template <class _ExecutionPolicy, class _ForwardIt1, class _ForwardIt2, class _T, class... _Events>
oneapi::dpl::__internal::__enable_if_async_execution_policy<_ExecutionPolicy, oneapi::dpl::__internal::__future<_T>,
                                                            _Events...>
transform_reduce_async(_ExecutionPolicy&& __exec, _ForwardIt1 __first1, _ForwardIt1 __last1, _ForwardIt2 __first2,
                       _T __init, _Events&&... __dependencies);

template <class _ExecutionPolicy, class _ForwardIt, class _T, class _BinaryOp, class _UnaryOp, class... _Events>
oneapi::dpl::__internal::__enable_if_async_execution_policy_single_no_default<
    _ExecutionPolicy, oneapi::dpl::__internal::__future<_T>, _UnaryOp, _Events...>
transform_reduce_async(_ExecutionPolicy&& __exec, _ForwardIt __first, _ForwardIt __last, _T __init,
                       _BinaryOp __binary_op, _UnaryOp __unary_op, _Events&&... __dependencies);

template <class _ExecutionPolicy, class _RandomAccessIterator, class... _Events>
oneapi::dpl::__internal::__enable_if_async_execution_policy<
    _ExecutionPolicy, oneapi::dpl::__par_backend_hetero::__future<void>, _Events...>
sort_async(_ExecutionPolicy&& __exec, _RandomAccessIterator __first, _RandomAccessIterator __last,
           _Events&&... __dependencies);

template <class _ExecutionPolicy, class _RandomAccessIterator, class _Compare, class... _Events>
oneapi::dpl::__internal::__enable_if_async_execution_policy_single_no_default<
    _ExecutionPolicy, oneapi::dpl::__par_backend_hetero::__future<void>, _Compare, _Events...>
sort_async(_ExecutionPolicy&& __exec, _RandomAccessIterator __first, _RandomAccessIterator __last, _Compare __comp,
           _Events&&... __dependencies);

template <class _ExecutionPolicy, class _ForwardIterator, class _Tp, class... _Events>
oneapi::dpl::__internal::__enable_if_async_execution_policy<
    _ExecutionPolicy, oneapi::dpl::__par_backend_hetero::__future<void>, _Events...>
fill_async(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last, const _Tp& __value,
           _Events&&... __dependencies);

} // namespace experimental

} // namespace dpl

} // namespace oneapi

#endif /* _ONEDPL_ASYNC_EXTENSION_DEFS_H */
