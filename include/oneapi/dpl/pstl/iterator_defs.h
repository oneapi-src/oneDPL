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

// Make is_random_access_iterator and is_forward_iterator not to fail with a 'hard' error when it's used in
// SFINAE with a non-iterator type by providing a default value.
template <typename _IteratorTag, typename... _IteratorTypes>
auto
__is_iterator_of(int) -> decltype(
    ::std::conjunction<::std::is_base_of<
        _IteratorTag, typename ::std::iterator_traits<::std::decay_t<_IteratorTypes>>::iterator_category>...>{});

template <typename... _IteratorTypes>
auto
__is_iterator_of(...) -> ::std::false_type;

template <typename... _IteratorTypes>
struct __is_random_access_iterator : decltype(__is_iterator_of<::std::random_access_iterator_tag, _IteratorTypes...>(0))
{
};

template <typename... _IteratorTypes>
struct __is_forward_iterator : decltype(__is_iterator_of<::std::forward_iterator_tag, _IteratorTypes...>(0))
{
};

template <typename... _IteratorTypes>
inline constexpr bool __is_random_access_iterator_v = __is_random_access_iterator<_IteratorTypes...>::value;

template <typename... _IteratorTypes>
inline constexpr bool __is_forward_iterator_v = __is_forward_iterator<_IteratorTypes...>::value;

} // namespace __internal
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_ITERATOR_DEFS_H
