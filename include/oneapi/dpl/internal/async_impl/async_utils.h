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

#ifndef _ONEDPL_ASYNC_UTILS_H
#define _ONEDPL_ASYNC_UTILS_H

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
    raw_data() const = 0;
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
    ::std::unique_ptr<__par_backend_hetero::__tmp_base> __tmp;

  public:
    template <typename... _Ts>
    __future(sycl::event __e, sycl::buffer<_Tp> __d, _Ts... __t)
        : __par_backend_hetero::__future_base(__e),
          __data(::std::unique_ptr<__async_transform<_Tp>>(new __async_transform<_Tp>(__d)))
    {
        if (sizeof...(_Ts) != 0)
            __tmp = ::std::unique_ptr<__par_backend_hetero::__TempObjs<_Ts...>>(
                new __par_backend_hetero::__TempObjs<_Ts...>(__t...));
    }
    // Constructor for reduce_transform pattern
    template <typename _Op>
    __future(__future<_Tp>&& _fp, _Tp __i, _Op __o)
        : __par_backend_hetero::__future_base(_fp),
          __data(::std::unique_ptr<__async_transform<_Tp, _Op>>(
              new __async_transform<_Tp, _Op>(_fp.raw_data().get_buffer(), __i, __o)))
    {
        __tmp.swap(_fp.__tmp);
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
    ::std::unique_ptr<__par_backend_hetero::__tmp_base> __tmp;

  public:
    template <typename... _Ts>
    __future(sycl::event __e, _Tp __d, _Ts... __t)
        : __future_base(__e), __data(::std::unique_ptr<__async_direct<_Tp>>(new __async_direct<_Tp>(__d)))
    {
        if (sizeof...(_Ts) != 0)
            __tmp = ::std::unique_ptr<__par_backend_hetero::__TempObjs<_Ts...>>(
                new __par_backend_hetero::__TempObjs<_Ts...>(__t...));
    }
    _Tp
    get()
    {
        this->wait();
        return __data->data();
    }
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

} // namespace dpl

} // namespace oneapi

#endif /* _ONEDPL_ASYNC_UTILS_H */
