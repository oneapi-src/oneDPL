// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) Intel Corporation
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

#ifndef _ONEDPL_EXECUTION_POLICY_DEFS_H
#define _ONEDPL_EXECUTION_POLICY_DEFS_H

#include <type_traits>
#include <iterator>

namespace oneapi
{
namespace dpl
{
namespace execution
{
inline namespace v1
{

// 2.4, Sequential execution policy
class sequenced_policy
{
  public:
    // For internal use only
    static constexpr ::std::false_type
    __allow_unsequenced()
    {
        return ::std::false_type{};
    }
    static constexpr ::std::false_type
    __allow_vector()
    {
        return ::std::false_type{};
    }
    static constexpr ::std::false_type
    __allow_parallel()
    {
        return ::std::false_type{};
    }
};

// 2.5, Parallel execution policy
class parallel_policy
{
  public:
    // For internal use only
    static constexpr ::std::false_type
    __allow_unsequenced()
    {
        return ::std::false_type{};
    }
    static constexpr ::std::false_type
    __allow_vector()
    {
        return ::std::false_type{};
    }
    static constexpr ::std::true_type
    __allow_parallel()
    {
        return ::std::true_type{};
    }
};

// 2.6, Parallel+Vector execution policy
class parallel_unsequenced_policy
{
  public:
    // For internal use only
    static constexpr ::std::true_type
    __allow_unsequenced()
    {
        return ::std::true_type{};
    }
    static constexpr ::std::true_type
    __allow_vector()
    {
        return ::std::true_type{};
    }
    static constexpr ::std::true_type
    __allow_parallel()
    {
        return ::std::true_type{};
    }
};

class unsequenced_policy
{
  public:
    // For internal use only
    static constexpr ::std::true_type
    __allow_unsequenced()
    {
        return ::std::true_type{};
    }
    static constexpr ::std::true_type
    __allow_vector()
    {
        return ::std::true_type{};
    }
    static constexpr ::std::false_type
    __allow_parallel()
    {
        return ::std::false_type{};
    }
};

// 2.8, Execution policy objects
inline constexpr sequenced_policy seq{};
inline constexpr parallel_policy par{};
inline constexpr parallel_unsequenced_policy par_unseq{};
inline constexpr unsequenced_policy unseq{};

// 2.3, Execution policy type trait
template <class T>
struct is_execution_policy : ::std::false_type
{
};

template <>
struct is_execution_policy<oneapi::dpl::execution::sequenced_policy> : ::std::true_type
{
};
template <>
struct is_execution_policy<oneapi::dpl::execution::parallel_policy> : ::std::true_type
{
};
template <>
struct is_execution_policy<oneapi::dpl::execution::parallel_unsequenced_policy> : ::std::true_type
{
};
template <>
struct is_execution_policy<oneapi::dpl::execution::unsequenced_policy> : ::std::true_type
{
};

template <class T>
inline constexpr bool is_execution_policy_v = oneapi::dpl::execution::is_execution_policy<T>::value;

} // namespace v1
} // namespace execution

namespace __internal
{

// Extension: host execution policy type trait
template <class _T>
struct __is_host_execution_policy : ::std::false_type
{
};

template <>
struct __is_host_execution_policy<oneapi::dpl::execution::sequenced_policy> : ::std::true_type
{
};
template <>
struct __is_host_execution_policy<oneapi::dpl::execution::parallel_policy> : ::std::true_type
{
};
template <>
struct __is_host_execution_policy<oneapi::dpl::execution::parallel_unsequenced_policy> : ::std::true_type
{
};
template <>
struct __is_host_execution_policy<oneapi::dpl::execution::unsequenced_policy> : ::std::true_type
{
};

template <class _ExecPolicy, class _T>
using __enable_if_execution_policy = typename ::std::enable_if<
    oneapi::dpl::execution::is_execution_policy<typename ::std::decay<_ExecPolicy>::type>::value, _T>::type;

template <class _ExecPolicy, class _T>
using __enable_if_host_execution_policy = typename ::std::enable_if<
    oneapi::dpl::__internal::__is_host_execution_policy<typename ::std::decay<_ExecPolicy>::type>::value, _T>::type;

template <class _ExecPolicy, const bool __condition, class _T>
using __enable_if_host_execution_policy_conditional = typename ::std::enable_if<
    oneapi::dpl::__internal::__is_host_execution_policy<typename ::std::decay<_ExecPolicy>::type>::value && __condition,
    _T>::type;

template <typename _ExecPolicy, typename _T>
struct __ref_or_copy_impl
{
    using type = _T&;
};

template <typename _ExecPolicy, typename _T>
using __ref_or_copy =
    typename oneapi::dpl::__internal::__ref_or_copy_impl<typename ::std::decay<_ExecPolicy>::type, _T>::type;

// utilities for Range API
template <typename _R>
auto
__check_size(int) -> decltype(::std::declval<_R&>().size());

template <typename _R>
auto
__check_size(...) -> decltype(::std::declval<_R&>().get_count());

template <typename _R>
using __difference_t = typename ::std::make_signed<decltype(__check_size<_R>(0))>::type;

} // namespace __internal

} // namespace dpl
} // namespace oneapi

#if _ONEDPL_BACKEND_SYCL
#    include "hetero/dpcpp/execution_sycl_defs.h"
#endif

#endif // _ONEDPL_EXECUTION_POLICY_DEFS_H
