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

using access_mode = sycl::access::mode;

namespace __internal
{
template <typename T, typename Allocator, typename IsConst = ::std::false_type>
struct sycl_iterator_trait
{
    using is_hetero = ::std::true_type;
    using is_hetero_const = ::std::false_type;

    using Buffer = sycl::buffer<T, 1, Allocator>;
    using BufferReturnType = Buffer;
};

template <typename T, typename Allocator>
struct sycl_iterator_trait<T, Allocator, ::std::true_type>
{
    using is_hetero = ::std::true_type;
    using is_hetero_const = ::std::true_type;

    using Buffer = sycl::buffer<T, 1, Allocator>;
    using BufferReturnType = const Buffer;
};

// Iterator that hides sycl::buffer to pass those to algorithms.
// SYCL iterator is a pair of sycl::buffer and integer value
template <access_mode Mode, typename T, typename Allocator, typename IsConst>
struct sycl_iterator_impl
{
  public:

    using Size = ::std::size_t;

    using iterator_category = ::std::random_access_iterator_tag;
    using is_hetero         = typename sycl_iterator_trait<T, Allocator, IsConst>::is_hetero;
    using is_hetero_const   = typename sycl_iterator_trait<T, Allocator, IsConst>::is_hetero_const;
    using difference_type   = ::std::make_signed<Size>::type;

    using value_type        = T;
    using pointer           = T*;
    using reference         = T&;
    using Buffer            = typename sycl_iterator_trait<T, Allocator, IsConst>::Buffer;
    using BufferReturnType  = typename sycl_iterator_trait<T, Allocator, IsConst>::BufferReturnType;

  private:
    Buffer buffer;
    Size idx = 0;

  public:

    // required for make_sycl_iterator
    //TODO: sycl::buffer doesn't have a default constructor (SYCL API issue), so we have to create a trivial size buffer
    sycl_iterator_impl(Buffer vec = Buffer(0), Size index = 0)
        : buffer(vec), idx(index)
    {
    }
    // required for iter_mode
    template <access_mode inMode>
    sycl_iterator_impl(const sycl_iterator_impl<inMode, T, Allocator, IsConst>& in) : buffer(in.get_buffer())
    {
        auto old_iter = sycl_iterator_impl<inMode, T, Allocator, IsConst>{in.get_buffer(), 0};
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
        return idx - it.idx;
    }
    template <access_mode OtherMode, typename OtherIsConst>
    bool
    operator==(const sycl_iterator_impl<OtherMode, T, Allocator, OtherIsConst>& it) const
    {
        return idx == it.get_index();
    }
    template <access_mode OtherMode, typename OtherIsConst>
    bool
    operator!=(const sycl_iterator_impl<OtherMode, T, Allocator, OtherIsConst>& it) const
    {
        return !(*this == it);
    }
    bool
    operator<(const sycl_iterator_impl& it) const
    {
        return *this - it < 0;
    }

    BufferReturnType
    get_buffer() const
    {
        return buffer;
    }

    Size
    get_index() const
    {
        return idx;
    }
};

template <access_mode Mode, typename T, typename Allocator = __dpl_sycl::__buffer_allocator<T>>
using sycl_iterator = sycl_iterator_impl<Mode, T, Allocator, ::std::false_type>;

template <access_mode Mode, typename T, typename Allocator = __dpl_sycl::__buffer_allocator<T>>
using sycl_const_iterator = sycl_iterator_impl<Mode, T, Allocator, ::std::true_type>;

// mode converter when property::noinit present
template <access_mode __mode>
struct _ModeConverter
{
    static constexpr access_mode __value = __mode;
};

template <>
struct _ModeConverter<access_mode::read_write>
{
    static constexpr access_mode __value = access_mode::discard_read_write;
};

template <>
struct _ModeConverter<access_mode::write>
{
    static constexpr access_mode __value = access_mode::discard_write;
};

} // namespace __internal

template <typename T, typename Allocator>
__internal::sycl_iterator<access_mode::read_write, T, Allocator> begin(sycl::buffer<T, /*dim=*/1, Allocator> buf)
{
    return __internal::sycl_iterator<access_mode::read_write, T, Allocator>{buf, 0};
}

template <typename T, typename Allocator>
__internal::sycl_iterator<access_mode::read_write, T, Allocator> end(sycl::buffer<T, /*dim=*/1, Allocator> buf)
{
    return __internal::sycl_iterator<access_mode::read_write, T, Allocator>{buf, __dpl_sycl::__get_buffer_size(buf)};
}

// begin
template <typename T, typename Allocator, access_mode Mode>
__internal::sycl_iterator<Mode, T, Allocator> begin(sycl::buffer<T, /*dim=*/1, Allocator> buf, sycl::mode_tag_t<Mode>)
{
    return __internal::sycl_iterator<Mode, T, Allocator>{buf, 0};
}

template <typename T, typename Allocator, access_mode Mode>
__internal::sycl_iterator<__internal::_ModeConverter<Mode>::__value, T, Allocator>
    begin(sycl::buffer<T, /*dim=*/1, Allocator> buf, sycl::mode_tag_t<Mode>, __dpl_sycl::__no_init)
{
    return __internal::sycl_iterator<__internal::_ModeConverter<Mode>::__value, T, Allocator>{buf, 0};
}

template <typename T, typename Allocator>
__internal::sycl_iterator<access_mode::discard_read_write, T, Allocator>
    begin(sycl::buffer<T, /*dim=*/1, Allocator> buf, __dpl_sycl::__no_init)
{
    return __internal::sycl_iterator<access_mode::discard_read_write, T, Allocator>{buf, 0};
}

// end
template <typename T, typename Allocator, access_mode Mode>
__internal::sycl_iterator<Mode, T, Allocator> end(sycl::buffer<T, /*dim=*/1, Allocator> buf, sycl::mode_tag_t<Mode>)
{
    return __internal::sycl_iterator<Mode, T, Allocator>{buf, __dpl_sycl::__get_buffer_size(buf)};
}

template <typename T, typename Allocator, access_mode Mode>
__internal::sycl_iterator<__internal::_ModeConverter<Mode>::__value, T, Allocator>
    end(sycl::buffer<T, /*dim=*/1, Allocator> buf, sycl::mode_tag_t<Mode>, __dpl_sycl::__no_init)
{
    return __internal::sycl_iterator<__internal::_ModeConverter<Mode>::__value, T, Allocator>{
        buf, __dpl_sycl::__get_buffer_size(buf)};
}

template <typename T, typename Allocator>
__internal::sycl_iterator<access_mode::discard_read_write, T, Allocator> end(sycl::buffer<T, /*dim=*/1, Allocator> buf,
                                                                             __dpl_sycl::__no_init)
{
    return __internal::sycl_iterator<access_mode::discard_read_write, T, Allocator>{buf,
                                                                                    __dpl_sycl::__get_buffer_size(buf)};
}

// cbegin
template <typename T, typename Allocator>
__internal::sycl_const_iterator<access_mode::read, T, Allocator>
cbegin(sycl::buffer<T, /*dim=*/1, Allocator> buf)
{
    return __internal::sycl_const_iterator<access_mode::read, T, Allocator>{buf, 0};
}

// cend
template <typename T, typename Allocator>
__internal::sycl_const_iterator<access_mode::read, T, Allocator>
cend(sycl::buffer<T, /*dim=*/1, Allocator> buf)
{
    return __internal::sycl_const_iterator<access_mode::read, T, Allocator>{buf, __dpl_sycl::__get_buffer_size(buf)};
}

} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_SYCL_ITERATOR_H
