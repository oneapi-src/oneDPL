// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2020 Intel Corporation
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

#ifndef _ONEDPL_UTILS_RANGES_H
#define _ONEDPL_UTILS_RANGES_H

#include <iterator>
#include <type_traits>

#include "iterator_impl.h"

namespace oneapi
{
namespace dpl
{
namespace __ranges
{

// helpers to check implement "has_base"
template <typename U>
auto
test_pipeline_object(int) -> decltype(::std::declval<U>().base(), ::std::true_type{});

template <typename U>
auto
test_pipeline_object(...) -> ::std::false_type;

// has_base check definition
template <typename Range>
struct is_pipeline_object : decltype(test_pipeline_object<Range>(0))
{
};

// Recursive helper
template <typename Range, typename = void>
struct pipeline_base
{
    using type = Range;
};

template <typename Range>
struct pipeline_base<Range, typename ::std::enable_if<is_pipeline_object<Range>::value>::type>
{
    using type = typename pipeline_base<typename ::std::decay<decltype(::std::declval<Range>().base())>::type>::type;
};

//pipeline_base_range
template <typename Range, typename = void>
struct pipeline_base_range
{
    Range rng;

    pipeline_base_range(Range r) : rng(r) {}
    _PSTL_CONSTEXPR_FUN Range
    base_range()
    {
        return rng;
    };
};

// use ::std::conditional to understand what class to inherit from
template <typename Range>
struct pipeline_base_range<Range, typename ::std::enable_if<is_pipeline_object<Range>::value, void>::type>
{
    Range rng;

    pipeline_base_range(Range r) : rng(r) {}
    _PSTL_CONSTEXPR_FUN auto
    base_range() -> decltype(pipeline_base_range<decltype(rng.base())>(rng.base()).base_range())
    {
        return pipeline_base_range<decltype(rng.base())>(rng.base()).base_range();
    };
};

template <typename _TupleType, typename _F, ::std::size_t... _Ip>
void
invoke(const _TupleType& __t, _F __f, oneapi::dpl::__internal::__index_sequence<_Ip...>)
{
    __f(::std::get<_Ip>(__t)...);
}

struct __invoke_begin
{
    template <typename _R>
    _PSTL_CONSTEXPR_FUN auto
    operator()(_R __r) -> decltype(::std::begin(__r))
    {
        return ::std::begin(__r);
    }
};

template <typename... _Ranges>
class zip_view
{
    static_assert(sizeof...(_Ranges) > 0, "Cannot instantiate zip_view with empty template parameter pack");

    using _tuple_ranges_t = oneapi::dpl::__internal::tuple<_Ranges...>;

    template <::std::size_t... _Ip>
    auto
    make_reference(_tuple_ranges_t __t, int32_t __i, oneapi::dpl::__internal::__index_sequence<_Ip...>) const
        -> decltype(oneapi::dpl::__internal::tuple<decltype(::std::declval<_Ranges&>().operator[](__i))...>(
            ::std::get<_Ip>(__t).operator[](__i)...))
    {
        return oneapi::dpl::__internal::tuple<decltype(::std::declval<_Ranges&>().operator[](__i))...>(
            ::std::get<_Ip>(__t).operator[](__i)...);
    }

  public:
    static constexpr ::std::size_t __num_ranges = sizeof...(_Ranges);

    explicit zip_view(_Ranges... __args) : __m_ranges(oneapi::dpl::__internal::make_tuple(__args...)) {}

    auto
    size() const -> decltype(::std::get<0>(::std::declval<_tuple_ranges_t>()).size())
    {
        return ::std::get<0>(__m_ranges).size();
    }

    _PSTL_CONSTEXPR_FUN auto operator[](int32_t __i) const
        -> decltype(make_reference(::std::declval<_tuple_ranges_t>(), __i,
                                   oneapi::dpl::__internal::__make_index_sequence<__num_ranges>()))
    {
        return make_reference(__m_ranges, __i, oneapi::dpl::__internal::__make_index_sequence<__num_ranges>());
    }

    bool
    empty() const
    {
        return size() == 0;
    }

    _tuple_ranges_t
    tuple()
    {
        return __m_ranges;
    }
    _tuple_ranges_t
    tuple() const
    {
        return __m_ranges;
    }

  private:
    _tuple_ranges_t __m_ranges;
};

template <typename... _Ranges>
auto
make_zip_view(_Ranges&&... args) -> decltype(zip_view<_Ranges...>(::std::forward<_Ranges>(args)...))
{
    return zip_view<_Ranges...>(::std::forward<_Ranges>(args)...);
}

// a custom view, over a pair of "passed directly" iterators
template <typename _Iterator>
class guard_view
{
    using value_type = typename ::std::iterator_traits<_Iterator>::value_type;
    using reference = typename ::std::iterator_traits<_Iterator>::reference;
    using diff_type = typename ::std::iterator_traits<_Iterator>::difference_type;

  public:
    guard_view(_Iterator __first = _Iterator(), diff_type __n = 0) : m_p(__first), m_count(__n) {}
    guard_view(_Iterator __first, _Iterator __last) : m_p(__first), m_count(__last - __first) {}

    _Iterator
    begin() const
    {
        return m_p;
    }

    _Iterator
    end() const
    {
        return begin() + size();
    }

    auto operator[](int32_t i) const -> decltype(begin()[i]) { return begin()[i]; }

    diff_type
    size() const
    {
        return m_count;
    }
    bool
    empty() const
    {
        return size() == 0;
    }

  private:
    _Iterator m_p;     // a iterator (pointer)  to data in memory
    diff_type m_count; // size of the data
};

//It is kind of pseudo-view for transfom_iterator support.
template <typename _R, typename _F>
struct transform_view_simple
{
    _R __r;
    _F __f;

    template <typename Idx>
    auto operator[](Idx __i) const -> decltype(__f(__r[__i]))
    {
        return __f(__r[__i]);
    }

    auto
    size() const -> decltype(__r.size())
    {
        return __r.size();
    }

    bool
    empty() const
    {
        return __r.empty();
    }

    auto
    base() const -> decltype(__r)
    {
        return __r;
    }
};

template <typename _Map>
auto
test_map_view(int) -> decltype(::std::declval<_Map>().begin(), ::std::true_type{});

template <typename _Map>
auto
test_map_view(...) -> ::std::false_type;

//pseudo-checking on viewable range concept
template <typename _Map>
struct is_map_view : decltype(test_map_view<_Map>(0))
{
};

template <typename _Map>
struct is_map_iterator : oneapi::dpl::__internal::__is_random_access_iterator<_Map>
{
};

template <typename _Map>
using is_map_functor = ::std::integral_constant<bool, !is_map_iterator<_Map>::value && !is_map_view<_Map>::value>;

//It is kind of pseudo-view for permutation_iterator support.
template <typename _R, typename _M, typename = void>
struct permutation_view_simple;

//permutation view: specialization for an index map functor
template <typename _R, typename _M>
struct permutation_view_simple<_R, _M, typename ::std::enable_if<is_map_functor<_M>::value, void>::type>
{
    _R __r;
    _M __map_fn;

    permutation_view_simple(_R __rng, _M __m) : __r(__rng), __map_fn(__m) {}

    template <typename Idx>
    auto operator[](Idx __i) const -> decltype(__r[__map_fn[__i]])
    {
        return __r[__map_fn[__i]];
    }

    auto
    size() const -> decltype(__r.size())
    {
        return __r.size();
    }

    bool
    empty() const
    {
        return size() == 0;
    }

    auto
    base() const -> decltype(__r)
    {
        return __r;
    }
};

//permutation view: specialization for a map view (a viewable range concept)
template <typename _R, typename _M>
struct permutation_view_simple<_R, _M, typename ::std::enable_if<is_map_view<_M>::value>::type>
{
    zip_view<_R, _M> __data;

    permutation_view_simple(_R __r, _M __m) : __data(__r, __m) {}

    template <typename Idx>
    auto operator[](Idx __i) const -> decltype(::std::get<0>(__data.tuple())[::std::get<1>(__data.tuple())[__i]])
    {
        return ::std::get<0>(__data.tuple())[::std::get<1>(__data.tuple())[__i]];
    }

    auto
    size() const -> decltype(::std::get<1>(__data.tuple()).size())
    {
        return ::std::get<1>(__data.tuple()).size();
    }

    bool
    empty() const
    {
        return size() == 0;
    }

    auto
    base() const -> decltype(__data)
    {
        return __data;
    }
};

//permutation discard view:
struct permutation_discard_view
{
    using difference_type = ::std::ptrdiff_t;
    difference_type m_count;

    permutation_discard_view(difference_type __n) : m_count(__n) {}

    oneapi::dpl::internal::ignore_copyable operator[](difference_type) const { return oneapi::dpl::internal::ignore; }

    difference_type
    size() const
    {
        return m_count;
    }

    bool
    empty() const
    {
        return size() == 0;
    }
};

} // namespace __ranges
} // namespace dpl
} // namespace oneapi

#endif /* _ONEDPL_UTILS_RANGES_H */
