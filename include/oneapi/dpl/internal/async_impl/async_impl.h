// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2021 Intel Corporation
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

#ifndef _ONEDPL_ASYNC_IMPL_H
#define _ONEDPL_ASYNC_IMPL_H

#if _ONEDPL_HETERO_BACKEND
#    include "async_impl_hetero.h"
#endif

#include "glue_async_impl.h"

namespace oneapi
{
namespace dpl
{
namespace experimental
{

// [wait_for_all]
template <typename... _Ts>
void
wait_for_all(const _Ts&... __Events)
{
    ::std::vector<sycl::event> __wait_list = {__Events...};
    for (auto _a : __wait_list)
        _a.wait();
}

// [async.reduce]
template <class _ExecutionPolicy, class _ForwardIt, class... _Events>
oneapi::dpl::__internal::__enable_if_async_execution_policy<
    _ExecutionPolicy, oneapi::dpl::__internal::__future<typename std::iterator_traits<_ForwardIt>::value_type>,
    _Events...>
reduce_async(_ExecutionPolicy&& __exec, _ForwardIt __first, _ForwardIt __last, _Events&&... __dependencies)
{
    using _Tp = typename std::iterator_traits<_ForwardIt>::value_type;
    return reduce_async(std::forward<_ExecutionPolicy>(__exec), __first, __last, _Tp(0), ::std::plus<_Tp>(), __dependencies...);
}

template <class _ExecutionPolicy, class _ForwardIt, class _T, class... _Events>
oneapi::dpl::__internal::__enable_if_async_execution_policy_single_no_default<
    _ExecutionPolicy, oneapi::dpl::__internal::__future<_T>, _T, _Events...>
reduce_async(_ExecutionPolicy&& __exec, _ForwardIt __first, _ForwardIt __last, _T __init, _Events&&... __dependencies)
{
    return reduce_async(std::forward<_ExecutionPolicy>(__exec), __first, __last, __init, ::std::plus<_T>(), __dependencies...);
}

// [async.transform_reduce]
template <class _ExecutionPolicy, class _ForwardIt1, class _ForwardIt2, class _T, class... _Events>
oneapi::dpl::__internal::__enable_if_async_execution_policy<_ExecutionPolicy, oneapi::dpl::__internal::__future<_T>,
                                                            _Events...>
transform_reduce_async(_ExecutionPolicy&& __exec, _ForwardIt1 __first1, _ForwardIt1 __last1, _ForwardIt2 __first2,
                       _T __init, _Events&&... __dependencies)
{
    return transform_reduce_async(std::forward<_ExecutionPolicy>(__exec), __first1, __last1, __first2, __init,
                                  ::std::plus<>(), ::std::multiplies<>(), __dependencies...);
}

// [async.sort]
template <class _ExecutionPolicy, class _RandomAccessIterator, class... _Events>
oneapi::dpl::__internal::__enable_if_async_execution_policy<
    _ExecutionPolicy, oneapi::dpl::__par_backend_hetero::__future<void>, _Events...>
sort_async(_ExecutionPolicy&& __exec, _RandomAccessIterator __first, _RandomAccessIterator __last,
           _Events&&... __dependencies)
{
    using __T = typename ::std::iterator_traits<_RandomAccessIterator>::value_type;
    return sort_async(std::forward<_ExecutionPolicy>(__exec), __first, __last, ::std::less<__T>(), __dependencies...);
}

} // namespace experimental

} // namespace dpl

} // namespace oneapi

#endif /* _ONEDPL_GLUE_ASYNC_IMPL_H */
