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

#ifndef _ONEDPL_DR_DETAIL_RANGES_HPP
#define _ONEDPL_DR_DETAIL_RANGES_HPP

#include <any>
#include <iterator>
#include <type_traits>

#include "std_ranges_shim.hpp"

namespace oneapi::dpl::experimental::dr
{

namespace ranges
{

template <typename>
inline constexpr bool disable_rank = false;

namespace __detail
{

template <typename T>
concept has_rank_method = requires(T t)
{
    {
        t.rank()
        } -> std::weakly_incrementable;
};

template <typename R>
concept has_rank_adl = requires(R& r)
{
    {
        rank_(r)
        } -> std::weakly_incrementable;
};

template <typename Iter>
concept is_remote_iterator_shadow_impl_ =
    std::forward_iterator<Iter> && has_rank_method<Iter> && !disable_rank<std::remove_cv_t<Iter>>;

struct rank_fn_
{
    // Return the rank associated with a remote range.
    // This is either:
    // 1) r.rank(), if the remote range has a `rank()` method
    // OR, if not available,
    // 2) r.begin().rank(), if iterator is `remote_iterator`
    template <stdrng::forward_range R>
    requires((has_rank_method<R> && !disable_rank<std::remove_cv_t<R>>) ||
             (has_rank_adl<R> && !disable_rank<std::remove_cv_t<R>>) ||
             is_remote_iterator_shadow_impl_<stdrng::iterator_t<R>>) constexpr auto
    operator()(R&& r) const
    {
        if constexpr (has_rank_method<R> && !disable_rank<std::remove_cv_t<R>>)
        {
            return std::forward<R>(r).rank();
        }
        else if constexpr (is_remote_iterator_shadow_impl_<stdrng::iterator_t<R>>)
        {
            // stdrng::begin needs an lvalue or borrowed_range. We only need
            // the rank from the stdrng::begin so creating a local lvalue is ok.
            auto t = r;
            return operator()(stdrng::begin(t));
        }
        else if constexpr (has_rank_adl<R> && !disable_rank<std::remove_cv_t<R>>)
        {
            return rank_(std::forward<R>(r));
        }
    }

    template <std::forward_iterator Iter>
    requires(has_rank_method<Iter> && !disable_rank<std::remove_cv_t<Iter>>) auto
    operator()(Iter iter) const
    {
        if constexpr (has_rank_method<Iter> && !disable_rank<std::remove_cv_t<Iter>>)
        {
            return std::forward<Iter>(iter).rank();
        }
    }
};

} // namespace __detail

inline constexpr auto rank = __detail::rank_fn_{};

namespace __detail
{

template <typename R>
concept remote_range_shadow_impl_ = stdrng::forward_range<R> && requires(R& r)
{
    ranges::rank(r);
};

template <typename R>
concept segments_range = stdrng::forward_range<R> && remote_range_shadow_impl_<stdrng::range_value_t<R>>;

template <typename R>
concept has_segments_method = requires(R r)
{
    {
        r.segments()
        } -> segments_range;
};

template <typename R>
concept has_segments_adl = requires(R& r)
{
    {
        segments_(r)
        } -> segments_range;
};

struct segments_fn_
{
    template <stdrng::forward_range R>
    requires(has_segments_method<R> || has_segments_adl<R>) constexpr decltype(auto)
    operator()(R&& r) const
    {
        if constexpr (has_segments_method<R>)
        {
            return std::forward<R>(r).segments();
        }
        else if constexpr (has_segments_adl<R>)
        {
            return segments_(std::forward<R>(r));
        }
    }

    template <std::forward_iterator I>
    requires(has_segments_method<I> || has_segments_adl<I>) constexpr decltype(auto)
    operator()(I iter) const
    {
        if constexpr (has_segments_method<I>)
        {
            return std::forward<I>(iter).segments();
        }
        else if constexpr (has_segments_adl<I>)
        {
            return segments_(std::forward<I>(iter));
        }
    }
};

} // namespace __detail

inline constexpr auto segments = __detail::segments_fn_{};

namespace __local
{

template <typename Iter>
concept has_local_adl = requires(Iter& iter)
{
    {
        local_(iter)
        } -> std::forward_iterator;
};

template <typename Iter>
concept iter_has_local_method = std::forward_iterator<Iter> && requires(Iter iter)
{
    {
        iter.local()
        } -> std::forward_iterator;
};

template <typename T>
struct is_localizable_helper : std::false_type
{
};

template <has_local_adl T>
struct is_localizable_helper<T> : std::true_type
{
};

template <iter_has_local_method T>
struct is_localizable_helper<T> : std::true_type
{
};

template <std::forward_iterator Iter>
requires(not iter_has_local_method<Iter> && not has_local_adl<Iter>) && requires() { std::iter_value_t<Iter>(); }
struct is_localizable_helper<Iter> : is_localizable_helper<std::iter_value_t<Iter>>
{
};

template <stdrng::forward_range R>
struct is_localizable_helper<R> : is_localizable_helper<stdrng::iterator_t<R>>
{
};

template <typename T>
concept is_localizable = is_localizable_helper<T>::value;

template <typename Segment>
concept segment_has_local_method = stdrng::forward_range<Segment> && requires(Segment segment)
{
    {
        segment.local()
        } -> stdrng::forward_range;
};

struct local_fn_
{

    // based on https://ericniebler.github.io/range-v3/#autotoc_md30  "Create
    // custom iterators"
    // TODO: rewrite using iterator_interface from
    //  https://github.com/boostorg/stl_interfaces
    template <typename Iter>
    requires stdrng::forward_range<typename Iter::value_type>
    struct cursor_over_local_ranges
    {
        Iter iter;
        auto
        make_begin_for_counted() const
        {
            if constexpr (iter_has_local_method<stdrng::iterator_t<typename Iter::value_type>>)
                return stdrng::begin(*iter).local();
            else
                return std::iterator<std::bidirectional_iterator_tag,
                                     cursor_over_local_ranges<stdrng::iterator_t<typename Iter::value_type>>>(
                    stdrng::begin(*iter));
        }
        auto
        read() const
        {
            return stdrng::views::counted(make_begin_for_counted(), stdrng::size(*iter));
        }
        bool
        equal(const cursor_over_local_ranges& other) const
        {
            return iter == other.iter;
        }
        void
        next()
        {
            ++iter;
        }
        void
        prev()
        {
            --iter;
        }
        void
        advance(std::ptrdiff_t n)
        {
            this->iter += n;
        }
        std::ptrdiff_t
        distance_to(const cursor_over_local_ranges& other) const
        {
            return other.iter - this->iter;
        }
        cursor_over_local_ranges() = default;
        cursor_over_local_ranges(Iter iter) : iter(iter) {}
    };

    template <std::forward_iterator Iter>
    requires(has_local_adl<Iter> || iter_has_local_method<Iter> || std::contiguous_iterator<Iter> ||
             is_localizable<Iter>) auto
    operator()(Iter iter) const
    {
        if constexpr (iter_has_local_method<Iter>)
        {
            return iter.local();
        }
        else if constexpr (has_local_adl<Iter>)
        {
            return local_(iter);
        }
        else if constexpr (is_localizable<Iter>)
        {
            return std::iterator<std::bidirectional_iterator_tag, cursor_over_local_ranges<Iter>>(iter);
        }
        else if constexpr (std::contiguous_iterator<Iter>)
        {
            return iter;
        }
    }

    template <stdrng::forward_range R>
    requires(has_local_adl<R> || iter_has_local_method<stdrng::iterator_t<R>> || segment_has_local_method<R> ||
             std::contiguous_iterator<stdrng::iterator_t<R>> || is_localizable<R> || stdrng::contiguous_range<R>) auto
    operator()(R&& r) const
    {
        if constexpr (segment_has_local_method<R>)
        {
            return r.local();
        }
        else if constexpr (iter_has_local_method<stdrng::iterator_t<R>>)
        {
            return stdrng::views::counted(stdrng::begin(r).local(), stdrng::size(r));
        }
        else if constexpr (has_local_adl<R>)
        {
            return local_(std::forward<R>(r));
        }
        else if constexpr (is_localizable<R>)
        {
            return stdrng::views::counted(
                std::iterator<std::input_iterator_tag, cursor_over_local_ranges<stdrng::iterator_t<R>>>(
                    stdrng::begin(r)),
                stdrng::size(r));
        }
        else if constexpr (std::contiguous_iterator<stdrng::iterator_t<R>>)
        {
            return std::span(stdrng::begin(r), stdrng::size(r));
        }
    }
};

} // namespace __local

inline constexpr auto local = __local::local_fn_{};

namespace __detail
{

template <typename T>
concept has_local = requires(T& t)
{
    {
        ranges::local(t)
        } -> std::convertible_to<std::any>;
};

struct local_or_identity_fn_
{
    template <typename T>
    requires(has_local<T>) auto
    operator()(T&& t) const
    {
        return ranges::local(t);
    }

    template <typename T>
    decltype(auto)
    operator()(T&& t) const
    {
        return std::forward<T>(t);
    }
};


} // namespace __detail

inline constexpr auto local_or_identity = __detail::local_or_identity_fn_{};

} // namespace ranges

} // namespace oneapi::dpl::experimental::dr

#endif // _ONEDPL_DR_DETAIL_RANGES_HPP
