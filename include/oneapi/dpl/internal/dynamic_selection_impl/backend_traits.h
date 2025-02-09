// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _ONEDPL_INTERNAL_BACKEND_TRAITS_H
#define _ONEDPL_INTERNAL_BACKEND_TRAITS_H

#include <utility>
#include <type_traits>

namespace oneapi
{
namespace dpl
{
namespace experimental
{
namespace internal
{
//lazy_report
template <typename Backend>
auto
has_lazy_report_impl(...) -> std::false_type;

template <typename Backend>
auto
has_lazy_report_impl(int) -> decltype(std::declval<Backend>().lazy_report(), std::true_type{});

template <typename Backend>
struct has_lazy_report : decltype(has_lazy_report_impl<Backend>(0))
{
};

//get_resources
template <typename Backend>
auto
has_get_resources_impl(...) -> std::false_type;

template <typename Backend>
auto
has_get_resources_impl(int) -> decltype(std::declval<Backend>().get_resources(), std::true_type{});

template <typename Backend>
struct has_get_resources : decltype(has_get_resources_impl<Backend>(0))
{
};

} //namespace internal

namespace backend_traits
{
//lazy_report
template <typename S>
struct lazy_report_value
{
    static constexpr bool value = ::oneapi::dpl::experimental::internal::has_lazy_report<S>::value;
};
template <typename S>
inline constexpr bool lazy_report_v = lazy_report_value<S>::value;

//get_resources
template <typename S>
struct get_resources_value
{
    static constexpr bool value = ::oneapi::dpl::experimental::internal::has_get_resources<S>::value;
};
template <typename S>
inline constexpr get_resources_v = get_resources_value<S>::value;
} //namespace backend_traits

} // namespace experimental
} // namespace dpl
} // namespace oneapi

#endif /*_ONEDPL_INTERNAL_BACKEND_TRAITS_H*/
