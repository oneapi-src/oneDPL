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

#ifndef _ONEDPL_DR_DETAIL_REMOTE_SUBRANGE_HPP
#define _ONEDPL_DR_DETAIL_REMOTE_SUBRANGE_HPP

#include <iterator>
#include "std_ranges_shim.hpp"

namespace oneapi::dpl::experimental::dr
{

template <std::forward_iterator I>
class remote_subrange : public stdrng::subrange<I, I>
{
    using base = stdrng::subrange<I, I>;

  public:
    remote_subrange() requires std::default_initializable<I>
    = default;

    constexpr remote_subrange(I first, I last, std::size_t rank) : base(first, last), rank_(rank) {}

    template <stdrng::forward_range R>
    constexpr remote_subrange(R&& r, std::size_t rank) : base(stdrng::begin(r), stdrng::end(r)), rank_(rank)
    {
    }

    template <remote_range R>
    constexpr remote_subrange(R&& r) : base(stdrng::begin(r), stdrng::end(r)), rank_(ranges::rank(r))
    {
    }

    constexpr std::size_t
    rank() const noexcept
    {
        return rank_;
    }

  private:
    std::size_t rank_;
};

template <stdrng::forward_range R>
remote_subrange(R&&, std::size_t) -> remote_subrange<stdrng::iterator_t<R>>;

template <remote_range R>
remote_subrange(R&&) -> remote_subrange<stdrng::iterator_t<R>>;

} // namespace oneapi::dpl::experimental::dr

#if !defined(DR_SPEC)

// Needed to satisfy concepts for stdrng::begin
template <typename R>
inline constexpr bool stdrng::enable_borrowed_range<oneapi::dpl::experimental::dr::remote_subrange<R>> = true;

#endif

#endif /* _ONEDPL_DR_DETAIL_REMOTE_SUBRANGE_HPP */
