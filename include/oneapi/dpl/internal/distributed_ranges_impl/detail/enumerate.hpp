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

#pragma once

#include <oneapi/dpl/internal/distributed_ranges_impl/detail/ranges_shim.hpp>
#include <oneapi/dpl/internal/distributed_ranges_impl/shp/zip_view.hpp>

namespace oneapi::dpl::experimental::dr
{

namespace __detail
{

namespace
{

template <rng::range R>
struct range_size
{
    using type = std::size_t;
};

template <rng::sized_range R>
struct range_size<R>
{
    using type = rng::range_size_t<R>;
};

template <rng::range R>
using range_size_t = typename range_size<R>::type;

} // namespace

class enumerate_adapter_closure
{
  public:
    template <rng::viewable_range R>
    auto
    operator()(R&& r) const
    {
        using S = range_size_t<R>;
        // NOTE: This line only necessary due to bug in range-v3 where views
        //       have non-weakly-incrementable size types. (Standard mandates
        //       size type must be weakly incrementable.)
        using W = std::conditional_t<std::weakly_incrementable<S>, S, std::size_t>;
        if constexpr (rng::sized_range<R>)
        {
            return oneapi::dpl::experimental::dr::shp::views::zip(rng::views::iota(W{0}, W{rng::size(r)}), std::forward<R>(r));
        }
        else
        {
            return oneapi::dpl::experimental::dr::shp::views::zip(rng::views::iota(W{0}), std::forward<R>(r));
        }
    }

    template <rng::viewable_range R>
    friend auto
    operator|(R&& r, const enumerate_adapter_closure& closure)
    {
        return closure(std::forward<R>(r));
    }
};

class enumerate_fn_
{
  public:
    template <rng::viewable_range R>
    constexpr auto
    operator()(R&& r) const
    {
        return enumerate_adapter_closure{}(std::forward<R>(r));
    }

    inline auto
    enumerate() const
    {
        return enumerate_adapter_closure{};
    }
};

inline constexpr auto enumerate = enumerate_fn_{};

} // namespace __detail

} // namespace oneapi::dpl::experimental::dr
