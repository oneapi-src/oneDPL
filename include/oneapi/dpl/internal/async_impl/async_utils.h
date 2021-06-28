/*
 *  Copyright (c) Intel Corporation
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

#if _ONEDPL_BACKEND_SYCL
#    include <CL/sycl.hpp>
#endif

namespace oneapi
{
namespace dpl
{
namespace __internal
{

template <typename _T>
struct async_value_base
{
    virtual ~async_value_base() = default;
    virtual _T data(_T) = 0;
};

template <typename _T, typename _Buf, typename _Op>
class async_value : public async_value_base<_T>
{
    _Buf __my_buffer;
    _Op __my_op;
    size_t __my_offset;

  public:
    async_value(_Buf __b, _Op __o, size_t __i) : __my_buffer(__b), __my_op(__o), __my_offset(__i) {}
    _T
    data(_T __init)
    {
        return __my_op(__my_buffer.template get_access<access_mode::read>()[__my_offset], __init);
    }
};

template <typename _T, class _Enable /*= void*/>
class __future : public __par_backend_hetero::__future_base
{
    ::std::unique_ptr<async_value_base<_T>> __ret_val;
    _T __init;

  public:
    // empty sequence including ready event
    __future(_T __i) : __par_backend_hetero::__future_base(), __init(__i) {}
    _T
    get()
    {
        this->wait();
        return __ret_val->data(__init);
    }
    // transform from internal future returned by __parallel_transform_reduce pattern
    template <typename _Op>
    __future(__par_backend_hetero::__future<_T> __o, _T __i, _Op __op)
        : __par_backend_hetero::__future_base(__o.__my_event), __init(__i)
    {
        using _Buf = decltype(__o.__data);
        __ret_val = ::std::unique_ptr<async_value<_T, _Buf, _Op>>(
            new async_value<_T, _Buf, _Op>(__o.__data, __op, __o.__result_idx));
    }
};

#if _ONEDPL_BACKEND_SYCL
// Specialization for hetero iterator and usm pointer
template <typename _T>
class __future<_T, typename std::enable_if<__par_backend_hetero::__internal::is_hetero_iterator<_T>::value ||
                                           __par_backend_hetero::__internal::is_passed_directly<_T>::value>::type>
    : public __par_backend_hetero::__future_base
{
    _T __data;
    ::std::unique_ptr<__par_backend_hetero::__lifetime_keeper_base> __tmp;

  public:
    // empty sequence including ready event.
    __future(_T __d) : __par_backend_hetero::__future_base(), __data(__d) {}
    _T
    get()
    {
        this->wait();
        return __data;
    }
    // transform from internal future returned by __parallel_transform_scan pattern.
    __future(__par_backend_hetero::__future<typename ::std::iterator_traits<_T>::value_type>&& __o, _T __d)
        : __par_backend_hetero::__future_base(::std::move(__o.__my_event)), __data(__d)
    {
        __tmp = ::std::unique_ptr<__par_backend_hetero::__lifetime_keeper<decltype(__o.__data)>>(
            new __par_backend_hetero::__lifetime_keeper<decltype(__o.__data)>(__o.__data));
    }
    // transform from internal future returned by __parallel_for pattern.
    __future(__par_backend_hetero::__future<void>&& __o, _T __d)
        : __par_backend_hetero::__future_base(::std::move(__o.__my_event)), __data(__d), __tmp(::std::move(__o.__tmps))
    {
    }
};
#endif

template <typename _ExecPolicy, typename _T, typename _Op1, typename... _Events>
using __enable_if_device_execution_policy_single_no_default = typename ::std::enable_if<
    oneapi::dpl::__internal::__is_device_execution_policy<typename ::std::decay<_ExecPolicy>::type>::value &&
        !::std::is_convertible<_Op1, sycl::event>::value &&
        oneapi::dpl::__internal::__is_convertible_to_event<_Events...>::value,
    _T>::type;

template <typename _ExecPolicy, typename _T, typename _Op1, typename _Op2, typename... _Events>
using __enable_if_device_execution_policy_double_no_default = typename ::std::enable_if<
    oneapi::dpl::__internal::__is_device_execution_policy<typename ::std::decay<_ExecPolicy>::type>::value &&
        !::std::is_convertible<_Op1, sycl::event>::value && !::std::is_convertible<_Op2, sycl::event>::value &&
        oneapi::dpl::__internal::__is_convertible_to_event<_Events...>::value,
    _T>::type;

} // namespace __internal

} // namespace dpl

} // namespace oneapi

#endif /* _ONEDPL_ASYNC_UTILS_H */
