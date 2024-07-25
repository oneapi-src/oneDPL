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

#include "ranges_shim.hpp"

namespace oneapi::dpl::experimental::dr
{

namespace __detail
{

template <stdrng::range R>
struct range_size
{
    using type = std::size_t;
};

template <stdrng::sized_range R>
struct range_size<R>
{
    using type = stdrng::range_size_t<R>;
};

template <stdrng::range R>
using range_size_t = typename range_size<R>::type;

class enumerate_adapter_closure
{
  public:
    template <stdrng::viewable_range R>
    auto
    operator()(R&& r) const
    {
        using S = range_size_t<R>;
        // NOTE: This line only necessary due to bug in range-v3 where views
        //       have non-weakly-incrementable size types. (Standard mandates
        //       size type must be weakly incrementable.)
        using W = std::conditional_t<std::weakly_incrementable<S>, S, std::size_t>;
        if constexpr (stdrng::sized_range<R>)
        {
            return stdrng::views::zip(stdrng::views::iota(W{0}, W{stdrng::size(r)}), std::forward<R>(r));
        }
        else
        {
            return stdrng::views::zip(stdrng::views::iota(W{0}), std::forward<R>(r));
        }
    }

    template <stdrng::viewable_range R>
    friend auto
    operator|(R&& r, const enumerate_adapter_closure& closure)
    {
        return closure(std::forward<R>(r));
    }
};

class enumerate_fn_
{
  public:
    template <stdrng::viewable_range R>
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
