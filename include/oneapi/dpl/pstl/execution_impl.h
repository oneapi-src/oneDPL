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

#include "execution_defs.h"
#include "iterator_defs.h"

namespace oneapi
{
namespace dpl
{
namespace __internal
{

/* predicate */

template <typename _Tp>
::std::false_type __lazy_and(_Tp, ::std::false_type)
{
    return ::std::false_type{};
}

template <typename _Tp>
inline _Tp
__lazy_and(_Tp __a, ::std::true_type)
{
    return __a;
}

template <typename _Tp>
::std::true_type __lazy_or(_Tp, ::std::true_type)
{
    return ::std::true_type{};
}

template <typename _Tp>
inline _Tp
__lazy_or(_Tp __a, ::std::false_type)
{
    return __a;
}

/* policy */
template <typename Policy>
struct __policy_traits
{
};

template <>
struct __policy_traits<oneapi::dpl::execution::sequenced_policy>
{
    typedef ::std::false_type __allow_parallel;
    typedef ::std::false_type __allow_unsequenced;
    typedef ::std::false_type __allow_vector;
};

template <>
struct __policy_traits<oneapi::dpl::execution::unsequenced_policy>
{
    typedef ::std::false_type __allow_parallel;
    typedef ::std::true_type __allow_unsequenced;
    typedef ::std::true_type __allow_vector;
};

template <>
struct __policy_traits<oneapi::dpl::execution::parallel_policy>
{
    typedef ::std::true_type __allow_parallel;
    typedef ::std::false_type __allow_unsequenced;
    typedef ::std::false_type __allow_vector;
};

template <>
struct __policy_traits<oneapi::dpl::execution::parallel_unsequenced_policy>
{
    typedef ::std::true_type __allow_parallel;
    typedef ::std::true_type __allow_unsequenced;
    typedef ::std::true_type __allow_vector;
};

template <typename _ExecutionPolicy>
using __collector_t = typename __internal::__policy_traits<::std::decay_t<_ExecutionPolicy>>::__collector_type;

template <typename _ExecutionPolicy>
using __allow_vector = typename __internal::__policy_traits<::std::decay_t<_ExecutionPolicy>>::__allow_vector;

template <typename _ExecutionPolicy>
using __allow_unsequenced = typename __internal::__policy_traits<::std::decay_t<_ExecutionPolicy>>::__allow_unsequenced;

template <typename _ExecutionPolicy>
using __allow_parallel = typename __internal::__policy_traits<::std::decay_t<_ExecutionPolicy>>::__allow_parallel;

template <typename _ExecutionPolicy, typename... _IteratorTypes>
auto
__is_vectorization_preferred(_ExecutionPolicy& __exec)
    -> decltype(__internal::__lazy_and(__exec.__allow_vector(),
                                       __internal::__is_random_access_iterator_t<_IteratorTypes...>()))
{
    return __internal::__lazy_and(__exec.__allow_vector(),
                                  __internal::__is_random_access_iterator_t<_IteratorTypes...>());
}

template <typename _ExecutionPolicy, typename... _IteratorTypes>
auto
__is_parallelization_preferred(_ExecutionPolicy& __exec)
    -> decltype(__internal::__lazy_and(__exec.__allow_parallel(),
                                       __internal::__is_random_access_iterator_t<_IteratorTypes...>()))
{
    return __internal::__lazy_and(__exec.__allow_parallel(),
                                  __internal::__is_random_access_iterator_t<_IteratorTypes...>());
}

//------------------------------------------------------------------------
// backend selector with tags
//------------------------------------------------------------------------

template <class _Policy, class... _IteratorTypes>
struct __vectorable_tag
{
    using __is_vector = std::conjunction<__allow_unsequenced<_Policy>,
                                         typename __internal::__is_random_access_iterator<_IteratorTypes...>>;
};

template <class _Policy, class... _IteratorTypes>
struct __serial_tag : __vectorable_tag<_Policy, _IteratorTypes...>
{
};

template <class _Policy, class... _IteratorTypes>
struct __parallel_tag : __vectorable_tag<_Policy, _IteratorTypes...>
{
};

template <class _Policy, class... _IteratorTypes>
struct __parallel_forward_tag : __vectorable_tag<_Policy, _IteratorTypes...>
{
};

template <typename _Policy, typename... _IteratorTypes>
using __tag_type =
    typename ::std::conditional<__internal::__is_random_access_iterator<_IteratorTypes...>::value,
                                __parallel_tag<_Policy, _IteratorTypes...>,
                                typename ::std::conditional<__is_forward_iterator<_IteratorTypes...>::value,
                                                            __parallel_forward_tag<_Policy, _IteratorTypes...>,
                                                            __serial_tag<_Policy, _IteratorTypes...>>::type>::type;

template <typename _Policy, class... _IteratorTypes>
typename ::std::enable_if<!__allow_parallel<_Policy>::value,
                          __serial_tag<_Policy, typename ::std::decay<_IteratorTypes>::type...>>::type
__select_backend(_Policy&&, _IteratorTypes&&...)
{
    return {};
}

template <typename _Policy, class... _IteratorTypes>
typename ::std::enable_if<__allow_parallel<_Policy>::value,
                          __tag_type<_Policy, typename ::std::decay<_IteratorTypes>::type...>>::type
__select_backend(_Policy&&, _IteratorTypes&&...)
{
    return {};
}

} // namespace __internal
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_EXECUTION_IMPL_H
