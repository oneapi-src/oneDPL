// -*- C++ -*-
//===-- sycl_iterator.h ---------------------------------------------------===//
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

#ifndef _ONEDPL_SYCL_ITERATOR_H
#define _ONEDPL_SYCL_ITERATOR_H

#include <iterator>
#include "../../onedpl_config.h"
#include "sycl_defs.h"

namespace oneapi
{
namespace dpl
{
namespace __internal
{
namespace
{
template <typename T, typename Allocator, typename IsConst = ::std::false_type>
struct sycl_iterator_trait
{
    using value_type = T;
    using pointer = T*;
    using reference = T&;

    using is_hetero = ::std::true_type;
    using is_hetero_const = ::std::false_type;

    using Buffer = sycl::buffer<T, 1, Allocator>;
    using BufferReturnType = Buffer;
};

template <typename T, typename Allocator>
struct sycl_iterator_trait<T, Allocator, ::std::true_type>
{
    using value_type = const T;
    using pointer = const T*;
    using reference = const T&;

    using is_hetero = ::std::true_type;
    using is_hetero_const = ::std::true_type;

    using Buffer = sycl::buffer<T, 1, Allocator>;
    using BufferReturnType = const Buffer;
};

} // namespace;

// Iterator that hides sycl::buffer to pass those to algorithms.
// SYCL iterator is a pair of sycl::buffer and integer value
template <typename T, typename Allocator, typename IsConst>
struct sycl_iterator_impl
{
  public:

    using Size = ::std::size_t;

    using iterator_category = ::std::random_access_iterator_tag;
    using is_hetero         = ::std::true_type;
    using difference_type   = ::std::make_signed<Size>::type;

    using value_type        = typename sycl_iterator_trait<T, Allocator, IsConst>::value_type;
    using pointer           = typename sycl_iterator_trait<T, Allocator, IsConst>::pointer;
    using reference         = typename sycl_iterator_trait<T, Allocator, IsConst>::reference;
    using is_hetero_const   = typename sycl_iterator_trait<T, Allocator, IsConst>::is_hetero_const;
    using Buffer            = typename sycl_iterator_trait<T, Allocator, IsConst>::Buffer;
    using BufferReturnType  = typename sycl_iterator_trait<T, Allocator, IsConst>::BufferReturnType;

  private:
    Buffer buffer;
    Size idx = 0;

  public:

    // required for make_sycl_iterator
    //TODO: sycl::buffer doesn't have a default constructor (SYCL API issue), so we have to create a trivial size buffer
    sycl_iterator_impl(Buffer vec = Buffer(0), Size index = 0) : buffer(vec), idx(index) {}
    sycl_iterator_impl(const sycl_iterator_impl<T, Allocator, IsConst>& in) : buffer(in.get_buffer())
    {
        auto old_iter = sycl_iterator_impl<T, Allocator, IsConst>{in.get_buffer(), 0};
        idx = in - old_iter;
    }
    sycl_iterator_impl&
    operator=(const sycl_iterator_impl& in)
    {
        buffer = in.buffer;
        idx = in.idx;
        return *this;
    }
    sycl_iterator_impl
    operator+(difference_type forward) const
    {
        return {buffer, idx + forward};
    }
    sycl_iterator_impl
    operator-(difference_type backward) const
    {
        return {buffer, idx - backward};
    }
    friend sycl_iterator_impl
    operator+(difference_type forward, const sycl_iterator_impl& it)
    {
        return it + forward;
    }
    friend sycl_iterator_impl
    operator-(difference_type forward, const sycl_iterator_impl& it)
    {
        return it - forward;
    }
    difference_type
    operator-(const sycl_iterator_impl& it) const
    {
        assert(buffer == it.get_buffer());
        return idx - it.idx;
    }
    bool
    operator==(const sycl_iterator_impl& it) const
    {
        assert(buffer == it.get_buffer());
        return *this - it == 0;
    }
    bool
    operator!=(const sycl_iterator_impl& it) const
    {
        assert(buffer == it.get_buffer());
        return !(*this == it);
    }
    bool
    operator<(const sycl_iterator_impl& it) const
    {
        assert(buffer == it.get_buffer());
        return *this - it < 0;
    }

    BufferReturnType
    get_buffer() const
    {
        return buffer;
    }
};

template <typename T, typename Allocator = __dpl_sycl::__buffer_allocator<T> >
using sycl_iterator = sycl_iterator_impl<T, Allocator, ::std::false_type>;

template <typename T, typename Allocator = __dpl_sycl::__buffer_allocator<T> >
using sycl_const_iterator = sycl_iterator_impl<T, Allocator, ::std::true_type>;

} // namespace __internal

// begin
template <typename T, typename Allocator>
__internal::sycl_iterator<T, Allocator>
begin(sycl::buffer<T, /*dim=*/1, Allocator> buf)
{
    return {buf, 0};
}

// end
template <typename T, typename Allocator>
__internal::sycl_iterator<T, Allocator>
end(sycl::buffer<T, /*dim=*/1, Allocator> buf)
{
    return {buf, __dpl_sycl::__get_buffer_size(buf)};
}

// cbegin
template <typename T, typename Allocator>
__internal::sycl_const_iterator<T, Allocator>
cbegin(sycl::buffer<T, /*dim=*/1, Allocator> buf)
{
    return {buf, 0};
}

// cend
template <typename T, typename Allocator>
__internal::sycl_const_iterator<T, Allocator>
cend(sycl::buffer<T, /*dim=*/1, Allocator> buf)
{
    return {buf, __dpl_sycl::__get_buffer_size(buf)};
}

// Old variants of begin/end for compatibility with old code
inline namespace deprecated
{
    using access_mode = sycl::access::mode;

    // begin
    template <typename T, typename Allocator, access_mode Mode>
    _ONEDPL_DEPRECATED("Please use begin(sycl::buffer<T, 1, Allocator> buf)")
    __internal::sycl_iterator<T, Allocator>
    begin(sycl::buffer<T, /*dim=*/1, Allocator> buf, sycl::mode_tag_t<Mode>)
    {
        return oneapi::dpl::begin(buf);
    }

    template <typename T, typename Allocator, access_mode Mode>
    _ONEDPL_DEPRECATED("Please use begin(sycl::buffer<T, 1, Allocator> buf)")
    __internal::sycl_iterator<T, Allocator>
    begin(sycl::buffer<T, /*dim=*/1, Allocator> buf, sycl::mode_tag_t<Mode>, __dpl_sycl::__no_init)
    {
        return oneapi::dpl::begin(buf);
    }

    template <typename T, typename Allocator>
    _ONEDPL_DEPRECATED("Please use begin(sycl::buffer<T, 1, Allocator> buf)")
    __internal::sycl_iterator<T, Allocator>
    begin(sycl::buffer<T, /*dim=*/1, Allocator> buf, __dpl_sycl::__no_init)
    {
        return oneapi::dpl::begin(buf);
    }

    // end
    template <typename T, typename Allocator, access_mode Mode>
    _ONEDPL_DEPRECATED("Please use end(sycl::buffer<T, 1, Allocator> buf)")
    __internal::sycl_iterator<T, Allocator>
    end(sycl::buffer<T, /*dim=*/1, Allocator> buf, sycl::mode_tag_t<Mode>)
    {
        return oneapi::dpl::end(buf);
    }

    template <typename T, typename Allocator, access_mode Mode>
    _ONEDPL_DEPRECATED("Please use end(sycl::buffer<T, 1, Allocator> buf)")
    __internal::sycl_iterator<T, Allocator>
    end(sycl::buffer<T, /*dim=*/1, Allocator> buf, sycl::mode_tag_t<Mode>, __dpl_sycl::__no_init)
    {
        return oneapi::dpl::end(buf);
    }

    template <typename T, typename Allocator>
    _ONEDPL_DEPRECATED("Please use end(sycl::buffer<T, 1, Allocator> buf)")
    __internal::sycl_iterator<T, Allocator>
    end(sycl::buffer<T, /*dim=*/1, Allocator> buf, __dpl_sycl::__no_init)
    {
        return oneapi::dpl::end(buf);
    }

} // namespace depreated

} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_SYCL_ITERATOR_H
