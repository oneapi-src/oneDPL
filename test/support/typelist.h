// -*- C++ -*-
//===---------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------===//

#ifndef _TYPELIST_H
#define _TYPELIST_H

#include <type_traits>

namespace TestUtils
{
template <typename... Types>
class TList
{
};

template <typename List>
class HeadType;

template <typename HeadTypeItem, typename... RestOfTypeItems>
class HeadType<TList<HeadTypeItem, RestOfTypeItems...>>
{
  public:
    using type = HeadTypeItem;
};

template <typename List>
using GetHeadType = typename HeadType<List>::type;

template <typename List>
class PopHeadType;

template <typename HeadTypeItem, typename... RestOfTypeItems>
class PopHeadType<TList<HeadTypeItem, RestOfTypeItems...>>
{
  public:
    using type = TList<RestOfTypeItems...>;
};

template <typename List>
using GetRestTypes = typename PopHeadType<List>::type;

template <typename List>
class TypeListIsEmpty
{
  public:
    static constexpr ::std::false_type value;
};

template <>
class TypeListIsEmpty<TList<>>
{
  public:
    static constexpr ::std::true_type value;
};

template <typename List>
constexpr bool
type_list_is_empty()
{
    return TypeListIsEmpty<List>::value;
}

template <typename List, typename DestType>
constexpr bool
type_list_contain()
{
    if constexpr (TypeListIsEmpty<List>::value)
        return false;

    using HeadType = GetHeadType<List>;
    if constexpr (::std::is_same_v<HeadType, DestType>)
        return true;

    using RestList = GetRestTypes<List>;
    if constexpr (!TypeListIsEmpty<RestList>::value)
    {
        return type_list_contain<RestList, DestType>();
    }

    return false;
}

} /* namespace TestUtils */

#endif // _TYPELIST_H
