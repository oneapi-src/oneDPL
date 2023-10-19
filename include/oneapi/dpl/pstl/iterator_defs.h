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

namespace oneapi
{
namespace dpl
{
namespace __internal
{

// Internal wrapper around ::std::iterator_traits as it is required to be
// SFINAE-friendly(not produce "hard" error when _Ip is not an iterator)
// only starting with C++17. Although many standard library implementations
// provide it for older versions, we cannot rely on that.
template <typename _Ip, typename = void>
struct __iterator_traits
{
};

template <typename _Ip>
struct __iterator_traits<_Ip,
                         __void_type<typename _Ip::iterator_category, typename _Ip::value_type,
                                     typename _Ip::difference_type, typename _Ip::pointer, typename _Ip::reference>>
    : ::std::iterator_traits<_Ip>
{
};

// Handles _Tp* and const _Tp* specializations
template <typename _Tp>
struct __iterator_traits<_Tp*, void> : ::std::iterator_traits<_Tp*>
{
};

// Make is_random_access_iterator and is_forward_iterator not to fail with a 'hard' error when it's used in
//SFINAE with a non-iterator type by providing a default value.
template <typename _IteratorTag, typename... _IteratorTypes>
auto
__is_needed_iter(int)
    -> decltype(std::conjunction<::std::is_base_of<_IteratorTag, typename __iterator_traits<typename ::std::decay<
                                                                     _IteratorTypes>::type>::iterator_category>...>{});

template <typename... _IteratorTypes>
auto
__is_needed_iter(...) -> ::std::false_type;

template <typename... _IteratorTypes>
struct __is_random_access_iterator : decltype(__is_needed_iter<::std::random_access_iterator_tag, _IteratorTypes...>(0))
{
};

template <typename... _IteratorTypes>
struct __is_forward_iterator : decltype(__is_needed_iter<::std::forward_iterator_tag, _IteratorTypes...>(0))
{
};

template <typename... _IteratorTypes>
using __is_random_access_iterator_t = typename __is_random_access_iterator<_IteratorTypes...>::type;

template <typename... _IteratorTypes>
inline constexpr bool __is_random_access_iterator_v = __is_random_access_iterator<_IteratorTypes...>::value;

// struct for checking if iterator is heterogeneous or not
template <typename Iter, typename Void = void> // for non-heterogeneous iterators
struct is_hetero_iterator : ::std::false_type
{
};

template <typename Iter> // for heterogeneous iterators
struct is_hetero_iterator<Iter, ::std::enable_if_t<Iter::is_hetero::value>> : ::std::true_type
{
};
// struct for checking if iterator should be passed directly to device or not
template <typename Iter, typename Void = void> // for iterators that should not be passed directly
struct is_passed_directly : ::std::false_type
{
};

template <typename Iter> // for iterators defined as direct pass
struct is_passed_directly<Iter, ::std::enable_if_t<Iter::is_passed_directly::value>> : ::std::true_type
{
};

template <typename Iter> // for pointers to objects on device
struct is_passed_directly<Iter, ::std::enable_if_t<::std::is_pointer_v<Iter>>> : ::std::true_type
{
};

} // namespace __internal
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_ITERATOR_DEFS_H
