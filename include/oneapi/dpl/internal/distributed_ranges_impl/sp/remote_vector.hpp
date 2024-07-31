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

#include "allocators.hpp"
#include "vector.hpp"

namespace oneapi::dpl::experimental::dr::sp
{

// A `remote_vector` is simply a normal `std::vector` that's
// been decorated with an extra `rank()` function, showing
// which rank its memory is located on.
// (Thus fulfilling the `remote_range` concept.)

template <typename T, typename Allocator>
class remote_vector : public vector<T, Allocator>
{
  public:
    constexpr remote_vector() noexcept {}

    using base = vector<T, Allocator>;

    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::size_t;

    constexpr remote_vector(size_type count, const Allocator& alloc, size_type rank) : base(count, alloc), rank_(rank)
    {
    }

    constexpr std::size_t
    rank() const noexcept
    {
        return rank_;
    }

  private:
    std::size_t rank_ = 0;
};

template <class Alloc>
remote_vector(std::size_t, const Alloc, std::size_t) -> remote_vector<typename Alloc::value_type, Alloc>;

} // namespace oneapi::dpl::experimental::dr::sp
