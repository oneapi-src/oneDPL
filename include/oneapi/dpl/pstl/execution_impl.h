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

#ifndef _ONEDPL_EXECUTION_IMPL_H
#define _ONEDPL_EXECUTION_IMPL_H

#include <iterator>
#include <type_traits>

#include "parallel_backend.h"
#include "execution_defs.h"
#include "iterator_defs.h"

namespace oneapi
{
namespace dpl
{
namespace __internal
{

//------------------------------------------------------------------------
// backend selector with tags
//------------------------------------------------------------------------

#if _ONEDPL_PAR_BACKEND_TBB
using __par_backend_tag = __tbb_backend_tag;
#elif _ONEDPL_PAR_BACKEND_OPENMP
using __par_backend_tag = __omp_backend_tag;
#elif _ONEDPL_PAR_BACKEND_SERIAL
using __par_backend_tag = __serial_backend_tag;
#else
#    error "Parallel backend was not specified"
#endif

template <class _IsVector>
struct __serial_tag
{
    using __is_vector = _IsVector;
    using __backend_tag = __par_backend_tag;
};

template <class _IsVector>
struct __parallel_tag
{
    using __is_vector = _IsVector;
    using __backend_tag = __par_backend_tag;
};

struct __parallel_forward_tag
{
    using __is_vector = ::std::false_type;
    using __backend_tag = __par_backend_tag;
};

template <class _IsVector, class... _IteratorTypes>
using __tag_type = ::std::conditional_t<
    __internal::__is_random_access_iterator_v<_IteratorTypes...>, __parallel_tag<_IsVector>,
    ::std::conditional_t<__is_forward_iterator_v<_IteratorTypes...>, __parallel_forward_tag, __serial_tag<_IsVector>>>;

template <class _ExecutionPolicy, class... _IteratorTypes>
constexpr ::std::enable_if_t<
    ::std::is_same_v<::std::decay_t<_ExecutionPolicy>, oneapi::dpl::execution::sequenced_policy>,
    __serial_tag<std::false_type>>
__select_backend()
{
    return {};
}

template <class _ExecutionPolicy, class... _IteratorTypes>
constexpr ::std::enable_if_t<
    ::std::is_same_v<::std::decay_t<_ExecutionPolicy>, oneapi::dpl::execution::unsequenced_policy>,
    __serial_tag<__internal::__is_random_access_iterator<_IteratorTypes...>>>
__select_backend()
{
    return {};
}

template <class _ExecutionPolicy, class... _IteratorTypes>
constexpr ::std::enable_if_t<
    ::std::is_same_v<::std::decay_t<_ExecutionPolicy>, oneapi::dpl::execution::parallel_policy>,
    __tag_type<std::false_type, _IteratorTypes...>>
__select_backend()
{
    return {};
}

template <class _ExecutionPolicy, class... _IteratorTypes>
constexpr ::std::enable_if_t<
    ::std::is_same_v<::std::decay_t<_ExecutionPolicy>, oneapi::dpl::execution::parallel_unsequenced_policy>,
    __tag_type<__internal::__is_random_access_iterator<_IteratorTypes...>, _IteratorTypes...>>
__select_backend()
{
    return {};
}

//----------------------------------------------------------
// __is_backend_tag_serial, __is_backend_tag_serial_v
//----------------------------------------------------------

template <class _Tag>
struct __is_backend_tag_serial : ::std::false_type
{
};

template <class _IsVector>
struct __is_backend_tag_serial<__serial_tag<_IsVector>> : ::std::true_type
{
};

template <class _Tag>
inline constexpr bool __is_backend_tag_serial_v = __is_backend_tag_serial<::std::decay_t<_Tag>>::value;

//----------------------------------------------------------
// __is_backend_tag_parallel_forward, __is_backend_tag_parallel_forward_v
//----------------------------------------------------------

template <class _Tag>
struct __is_backend_tag_parallel_forward : ::std::false_type
{
};

template <>
struct __is_backend_tag_parallel_forward<__parallel_forward_tag> : ::std::true_type
{
};

template <class _Tag>
inline constexpr bool __is_backend_tag_parallel_forward_v =
    __is_backend_tag_parallel_forward<::std::decay_t<_Tag>>::value;

//----------------------------------------------------------
// __is_backend_tag_parallel, __is_backend_tag_parallel_v
//----------------------------------------------------------

template <class _Tag>
struct __is_backend_tag_parallel : ::std::false_type
{
};

template <class _IsVector>
struct __is_backend_tag_parallel<__parallel_tag<_IsVector>> : ::std::true_type
{
};

template <class _Tag>
inline constexpr bool __is_backend_tag_parallel_v = __is_backend_tag_parallel<::std::decay_t<_Tag>>::value;

//----------------------------------------------------------
// __is_host_backend_tag_v
//----------------------------------------------------------

template <class _Tag>
inline constexpr bool __is_host_backend_tag_v =
    __is_backend_tag_serial_v<_Tag> || __is_backend_tag_parallel_forward_v<_Tag> || __is_backend_tag_parallel_v<_Tag>;

} // namespace __internal
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_EXECUTION_IMPL_H
