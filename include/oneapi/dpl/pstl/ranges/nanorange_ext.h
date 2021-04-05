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

#ifndef _ONEDPL_NANO_RANGES_EXT_H
#define _ONEDPL_NANO_RANGES_EXT_H

NANO_BEGIN_NAMESPACE

template <typename I>
class offset_iterator
{
  public:
    using iterator_type = I;
    using difference_type = iter_difference_t<I>;
    using value_type = iter_value_t<I>;
    using iterator_category = detail::legacy_iterator_category_t<I>;
    using reference = iter_reference_t<I>;
    using pointer = I;

    offset_iterator() = default;
    offset_iterator(const offset_iterator&) = default;
    constexpr offset_iterator&
    operator=(const offset_iterator&) = default;

    explicit offset_iterator(I it, difference_type n, difference_type offset, I b)
        : current_(it), beg_(b), n_(n), offset_(offset)
    {
        assert(it >= b);
        assert(it < b + n);
    }

  private:
    constexpr difference_type
    offset_position() const
    {
        return current_ - beg_ + offset_;
    }

  public:
    constexpr reference operator*() const { return *(beg_ + offset_position() % n_); }
    constexpr reference operator[](difference_type __i) const { return *(*this + __i); }
    constexpr difference_type
    operator-(const offset_iterator& __it) const
    {
        return offset_position() - __it.offset_position();
    }
    constexpr offset_iterator&
    operator+=(difference_type __forward)
    {
        current_ += __forward;
        return *this;
    }
    constexpr offset_iterator&
    operator-=(difference_type __backward)
    {
        return *this += -__backward;
    }
    constexpr offset_iterator&
    operator++()
    {
        return *this += 1;
    }
    constexpr offset_iterator&
    operator--()
    {
        return *this -= 1;
    }
    constexpr offset_iterator
    operator++(int)
    {
        offset_iterator __it(*this);
        ++(*this);
        return __it;
    }
    constexpr offset_iterator
    operator--(int)
    {
        offset_iterator __it(*this);
        --(*this);
        return __it;
    }

    constexpr offset_iterator
    operator-(difference_type __backward) const
    {
        offset_iterator it(*this);
        return it -= __backward;
    }
    constexpr offset_iterator
    operator+(difference_type __forward) const
    {
        offset_iterator it(*this);
        return it += __forward;
    }
    friend constexpr offset_iterator
    operator+(difference_type __forward, const offset_iterator __it)
    {
        return __it + __forward;
    }
    constexpr bool
    operator==(const offset_iterator& __it) const
    {
        return *this - __it == 0;
    }
    constexpr bool
    operator!=(const offset_iterator& __it) const
    {
        return !(*this == __it);
    }
    constexpr bool
    operator<(const offset_iterator& __it) const
    {
        return *this - __it < 0;
    }
    constexpr bool
    operator>(const offset_iterator& __it) const
    {
        return __it < *this;
    }
    constexpr bool
    operator<=(const offset_iterator& __it) const
    {
        return !(*this > __it);
    }
    constexpr bool
    operator>=(const offset_iterator& __it) const
    {
        return !(*this < __it);
    }

  private:
    I current_{};
    I beg_{};
    difference_type n_;
    difference_type offset_;
};

namespace rotate_view_
{

template <typename V>
struct rotate_view : view_interface<rotate_view<V>>
{
  private:
    static_assert(range<V>);
    static_assert(input_iterator<iterator_t<V>>);
    static_assert(view<V>);

    V base_ = V();
    range_difference_t<V> offset_;

  public:
    rotate_view() = default;

    constexpr rotate_view(V base, range_difference_t<V> offset) : base_(::std::move(base)), offset_(offset) {}

    template <typename R,
              ::std::enable_if_t<input_range<R> && viewable_range<R> && constructible_from<V, all_view<R>>, int> = 0>
    constexpr rotate_view(R&& r, range_difference_t<V> offset)
        : base_(views::all(::std::forward<R>(r))), offset_(offset)
    {
    }

    constexpr V
    base() const
    {
        return base_;
    }

    constexpr auto
    begin()
    {
        return offset_iterator(ranges::begin(base()), base().size(), offset_, ranges::begin(base()));
    }

    constexpr auto
    end()
    {
        return begin() + base().size();
    }
};

template <typename R>
rotate_view(R &&)->rotate_view<all_view<R>>;

} // namespace rotate_view_

using rotate_view_::rotate_view;

namespace detail
{

struct rotate_view_fn
{
    template <typename E>
    constexpr auto
    operator()(E&& e, range_difference_t<E> offset) const -> decltype(rotate_view{::std::forward<E>(e), offset})
    {
        return rotate_view{::std::forward<E>(e), offset};
    }

    template <typename D>
    constexpr auto
    operator()(D offset) const
    {
        return detail::rao_proxy{[offset](auto&& r) mutable
#ifndef NANO_MSVC_LAMBDA_PIPE_WORKAROUND
                                 -> decltype(rotate_view{::std::forward<decltype(r)>(r), offset})
#endif
                                 {
                                     return rotate_view{::std::forward<decltype(r)>(r), offset};
                                 }};
    }
};

} // namespace detail

namespace views
{
NANO_INLINE_VAR(nano::detail::rotate_view_fn, rotate)
}

namespace detail
{

struct generate_view_fn
{
    template <typename G, typename Bound = unreachable_sentinel_t>
    constexpr auto
    operator()(G g, Bound size) const
    {
        return transform_view{iota_view{0, size}, [g](auto) { return g(); }};
    }
};

struct fill_view_fn
{
    template <typename T, typename Bound = unreachable_sentinel_t>
    constexpr auto
    operator()(const T& value, Bound size) const
    {
        return transform_view{iota_view{0, size}, [value](auto) { return value; }};
    }
};

} // namespace detail

namespace views
{
NANO_INLINE_VAR(nano::detail::generate_view_fn, generate)
NANO_INLINE_VAR(nano::detail::fill_view_fn, fill)
} // namespace views

NANO_END_NAMESPACE

#endif //_ONEDPL_NANO_RANGES_EXT_H
