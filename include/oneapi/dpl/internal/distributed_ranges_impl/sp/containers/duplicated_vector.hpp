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
#include "remote_vector.hpp"

namespace oneapi::dpl::experimental::dr::sp
{

template <typename T, typename Allocator = device_allocator<T>>
class duplicated_vector
{
  public:
    using segment_type = remote_vector<T, Allocator>;

    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    using allocator_type = Allocator;

    duplicated_vector(std::size_t count = 0)
    {
        size_ = count;
        capacity_ = count;

        std::size_t rank = 0;
        for (auto&& device : devices())
        {
            segments_.emplace_back(segment_type(size(), Allocator(context(), device), rank++));
        }
    }

    size_type
    size() const noexcept
    {
        return size_;
    }

    segment_type&
    local_vector(std::size_t rank)
    {
        return segments_[rank];
    }

    const segment_type&
    local_vector(std::size_t rank) const
    {
        return segments_[rank];
    }

  private:
    std::vector<segment_type> segments_;
    std::size_t capacity_ = 0;
    std::size_t size_ = 0;
};

} // namespace oneapi::dpl::experimental::dr::sp
