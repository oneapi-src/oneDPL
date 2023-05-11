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
// Iterator that hides sycl::buffer to pass those to algorithms.
// SYCL iterator is a pair of sycl::buffer and integer value
template <access_mode Mode, typename T, typename Allocator = __dpl_sycl::__buffer_allocator<T>>
struct sycl_iterator
{
  private:
    using Size = ::std::size_t;
    static constexpr int dim = 1;
    sycl::buffer<T, dim, Allocator> buffer;
    Size idx;

  public:
    using value_type = T;
    using difference_type = ::std::make_signed<Size>::type;
    using pointer = T*;
    using reference = T&;
    using iterator_category = ::std::random_access_iterator_tag;
    using is_hetero = ::std::true_type;
    static constexpr access_mode mode = Mode;

    // required for make_sycl_iterator
    //TODO: sycl::buffer doesn't have a default constructor (SYCL API issue), so we have to create a trivial size buffer
    sycl_iterator(sycl::buffer<T, dim, Allocator> vec = sycl::buffer<T, dim, Allocator>(0), Size index = 0)
        : buffer(vec), idx(index)
    {
    }
    // required for iter_mode
    template <access_mode inMode>
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
        assert(buffer == it.get_buffer());
        return idx - it.idx;
    }
    bool
    operator==(const sycl_iterator& it) const
    {
        assert(buffer == it.get_buffer());
        return *this - it == 0;
    }
    bool
    operator!=(const sycl_iterator& it) const
    {
        assert(buffer == it.get_buffer());
        return !(*this == it);
    }
    bool
    operator<(const sycl_iterator& it) const
    {
        assert(buffer == it.get_buffer());
        return *this - it < 0;
    }

    sycl::buffer<T, dim, Allocator>
    get_buffer() const
    {
        return buffer;
    }
};

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
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_SYCL_ITERATOR_H
