// -*- C++ -*-
//===-- sycl_iterator.h ---------------------------------------------------===//
//
// Copyright (C) 2019-2020 Intel Corporation
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

#ifndef _PSTL_sycl_iterator_H
#define _PSTL_sycl_iterator_H

#include <CL/sycl.hpp>

#include <iterator>
#include "../../dpstd_config.h"

namespace dpstd
{
namespace __internal
{

// Iterator that hides sycl::buffer to pass those to algorithms.
// SYCL iterator is a pair of sycl::buffer and integer value
template <cl::sycl::access::mode Mode, typename T, typename Allocator = cl::sycl::buffer_allocator>
struct sycl_iterator
{
  private:
    using Size = std::size_t;
    static constexpr int dim = 1;
    cl::sycl::buffer<T, dim, Allocator> buffer;
    Size idx;

  public:
    using value_type = T;
    using difference_type = std::make_signed<Size>::type;
    using pointer = T*;
    using reference = T&;
    using iterator_category = std::random_access_iterator_tag;
    using is_hetero = std::true_type;
    static constexpr cl::sycl::access::mode mode = Mode;

    // required for make_sycl_iterator
    sycl_iterator(cl::sycl::buffer<T, dim, Allocator> vec, Size index) : buffer(vec), idx(index) {}
    // required for iter_mode
    template <cl::sycl::access::mode inMode>
    sycl_iterator(const sycl_iterator<inMode, T, Allocator>& in) : buffer(in.get_buffer())
    {
        auto old_iter = sycl_iterator<inMode, T, Allocator>{in.get_buffer(), 0};
        idx = in - old_iter;
    }
    sycl_iterator&
    operator=(const sycl_iterator& in)
    {
        buffer = in.buffer;
        idx = in.idx;
        return *this;
    }
    sycl_iterator
    operator+(difference_type forward) const
    {
        return {buffer, idx + forward};
    }
    sycl_iterator
    operator-(difference_type backward) const
    {
        return {buffer, idx - backward};
    }
    friend sycl_iterator
    operator+(difference_type forward, const sycl_iterator& it)
    {
        return it + forward;
    }
    friend sycl_iterator
    operator-(difference_type forward, const sycl_iterator& it)
    {
        return it - forward;
    }
    difference_type
    operator-(const sycl_iterator& it) const
    {
        return idx - it.idx;
    }
    bool
    operator==(const sycl_iterator& it) const
    {
        return *this - it == 0;
    }
    bool
    operator!=(const sycl_iterator& it) const
    {
        return !(*this == it);
    }

    cl::sycl::buffer<T, dim, Allocator>
    get_buffer() const
    {
        return buffer;
    }
};

} // namespace __internal

template <cl::sycl::access::mode Mode, typename T, typename Allocator, typename Size>
_ITERATORS_DEPRECATED __internal::sycl_iterator<Mode, T, Allocator>
    make_sycl_iterator(cl::sycl::buffer<T, /*dim=*/1, Allocator> buf, Size i)
{
    return __internal::sycl_iterator<Mode, T, Allocator>{buf, i};
}

template <typename T, typename Allocator, typename Size>
_ITERATORS_DEPRECATED __internal::sycl_iterator<cl::sycl::access::mode::read_write, T, Allocator>
    make_sycl_iterator(cl::sycl::buffer<T, /*dim=*/1, Allocator> buf, Size i)
{
    return make_sycl_iterator<cl::sycl::access::mode::read_write, T, Allocator>(buf, i);
}

template <cl::sycl::access::mode Mode, typename T, typename Allocator>
__internal::sycl_iterator<Mode, T, Allocator> begin(cl::sycl::buffer<T, /*dim=*/1, Allocator> buf)
{
    return __internal::sycl_iterator<Mode, T, Allocator>{buf, 0};
}

template <typename T, typename Allocator>
__internal::sycl_iterator<cl::sycl::access::mode::read_write, T, Allocator>
    begin(cl::sycl::buffer<T, /*dim=*/1, Allocator> buf)
{
    return __internal::sycl_iterator<cl::sycl::access::mode::read_write, T, Allocator>{buf, 0};
}

template <cl::sycl::access::mode Mode, typename T, typename Allocator>
__internal::sycl_iterator<Mode, T, Allocator> end(cl::sycl::buffer<T, /*dim=*/1, Allocator> buf)
{
    return __internal::sycl_iterator<Mode, T, Allocator>{buf, buf.get_count()};
}

template <typename T, typename Allocator>
__internal::sycl_iterator<cl::sycl::access::mode::read_write, T, Allocator>
    end(cl::sycl::buffer<T, /*dim=*/1, Allocator> buf)
{
    return __internal::sycl_iterator<cl::sycl::access::mode::read_write, T, Allocator>{buf, buf.get_count()};
}

} // namespace dpstd
#endif /* _PSTL_sycl_iterator_H */
