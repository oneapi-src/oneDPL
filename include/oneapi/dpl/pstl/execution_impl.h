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

struct __tbb_backend_tag
{
};
// KSATODO required create tag for dpcpp -> already implemented: __device_backend_tag, __fpga_backend_tag
// KSATODO required create tag for omp

template <class _IsVector>
struct __serial_tag
{
    using __is_vector = _IsVector;
};

template <class _IsVector>
struct __parallel_tag
{
    using __is_vector = _IsVector;
    // backend tag can be change depending on
    // TBB availability in the environment
    using __backend_tag = __tbb_backend_tag;
};

struct __parallel_forward_tag
{
    using __is_vector = ::std::false_type;
    // backend tag can be change depending on
    // TBB availability in the environment
    using __backend_tag = __tbb_backend_tag;
};

template <class _IsVector, class... _IteratorTypes>
using __tag_type = ::std::conditional_t<
    __internal::__is_random_access_iterator_v<_IteratorTypes...>, __parallel_tag<_IsVector>,
    ::std::conditional_t<__is_forward_iterator_v<_IteratorTypes...>, __parallel_forward_tag, __serial_tag<_IsVector>>>;

template <class... _IteratorTypes>
__serial_tag<std::false_type>
__select_backend(oneapi::dpl::execution::sequenced_policy, _IteratorTypes&&...)
{
    return {};
}

template <class... _IteratorTypes>
__serial_tag<__internal::__is_random_access_iterator<_IteratorTypes...>>
__select_backend(oneapi::dpl::execution::unsequenced_policy, _IteratorTypes&&...)
{
    return {};
}

template <class... _IteratorTypes>
__tag_type<std::false_type, _IteratorTypes...>
__select_backend(oneapi::dpl::execution::parallel_policy, _IteratorTypes&&...)
{
    return {};
}

template <class... _IteratorTypes>
__tag_type<__internal::__is_random_access_iterator<_IteratorTypes...>, _IteratorTypes...>
__select_backend(oneapi::dpl::execution::parallel_unsequenced_policy, _IteratorTypes&&...)
{
    return {};
}

} // namespace __internal
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_EXECUTION_IMPL_H
