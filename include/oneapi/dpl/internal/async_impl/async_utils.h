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
#    include "../../pstl/hetero/dpcpp/sycl_defs.h"
#    include "../../pstl/hetero/dpcpp/execution_sycl_defs.h"
#endif

#include <type_traits>

namespace oneapi
{
namespace dpl
{
namespace __internal
{

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
