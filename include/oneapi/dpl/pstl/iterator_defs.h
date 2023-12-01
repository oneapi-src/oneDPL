// -*- C++ -*-
//===-- iterator_defs.h ---------------------------------------------------===//
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

#ifndef _ONEDPL_ITERATOR_DEFS_H
#define _ONEDPL_ITERATOR_DEFS_H

#include <iterator>
#include <type_traits>
#include "utils.h"
#include "onedpl_config.h"

namespace oneapi
{
namespace dpl
{
namespace __internal
{

#if _ONEDPL_CPP20_ITERATOR_CONCEPTS_PRESENT

// Until C++23, std::move_iterator::iterator_concept is std::input_iterator_tag and
// std::random_access_iterator<std::move_iterator<...>> returns false even if the
// base iterator is random access. Making the dispatch based on the base iterator type
// considers std::move_iterator<random-access> a random access iterator
template <typename _IteratorType>
struct __move_iter_base_helper
{
    using type = _IteratorType;
};

template <typename _BaseIteratorType>
struct __move_iter_base_helper<::std::move_iterator<_BaseIteratorType>>
{
    using type = _BaseIteratorType;
};

template <typename _IteratorType>
struct __is_random_access_iterator_impl
    : ::std::bool_constant<std::random_access_iterator<typename __move_iter_base_helper<_IteratorType>::type>>
{
};

#else

// Make is_random_access_iterator not to fail with a 'hard' error when it's used in SFINAE with
// a non-iterator type by providing a default value.
template <typename _IteratorType, typename = void>
struct __is_random_access_iterator_impl : ::std::false_type
{
};

template <typename _IteratorType>
struct __is_random_access_iterator_impl<_IteratorType,
                                        ::std::void_t<typename ::std::iterator_traits<_IteratorType>::iterator_category>>
    : ::std::is_base_of<typename ::std::iterator_traits<_IteratorType>::iterator_category, ::std::random_access_iterator_tag>
{
};

#else

template <typename _IteratorType>
struct __is_random_access_iterator_impl : __is_random_access_iterator_impl<_IteratorType>
{
};

#endif

/* iterator */
template <typename _IteratorType, typename... _OtherIteratorTypes>
struct __is_random_access_iterator
    : ::std::conjunction<__is_random_access_iterator_impl<_IteratorType>,
                         __is_random_access_iterator_impl<_OtherIteratorTypes>...>
{
};

template <typename... _IteratorTypes>
inline constexpr bool __is_random_access_iterator_v = __is_random_access_iterator<_IteratorTypes...>::value;

} // namespace __internal
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_ITERATOR_DEFS_H
