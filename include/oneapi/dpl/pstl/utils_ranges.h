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

#ifndef _ONEDPL_UTILS_RANGES_H
#define _ONEDPL_UTILS_RANGES_H

#include <iterator>
#include <type_traits>

#include "iterator_defs.h"
#include "iterator_impl.h"
#include "execution_defs.h"

namespace oneapi
{
namespace dpl
{

namespace __internal
{

template <typename _R>
auto
get_value_type(int) -> typename ::std::decay_t<_R>::value_type;

template <typename _R>
auto
get_value_type(long) ->
    typename ::std::iterator_traits<::std::decay_t<decltype(::std::declval<_R&>().begin())>>::value_type;

template <typename _R>
auto
get_value_type(...)
{
    //static_assert should always fail when this overload is chosen, so its condition must depend on
    //the template parameter and evaluate to false
    static_assert(std::is_same_v<_R, void>,
        "error: the range has no 'value_type'; define an alias or typedef named 'value_type' in the range class");
}

template <typename _R>
using __value_t = decltype(oneapi::dpl::__internal::get_value_type<_R>(0));

template <typename _Proj, typename _R>
using __key_t = ::std::remove_cv_t<::std::remove_reference_t<::std::invoke_result_t<_Proj&, __value_t<_R>>>>;

template <typename T, typename = void>
struct __range_has_raw_ptr_iterator : ::std::false_type
{
};

template <typename T>
struct __range_has_raw_ptr_iterator<T, ::std::void_t<decltype(::std::declval<T&>().begin())>>
    : ::std::is_pointer<decltype(::std::declval<T&>().begin())>
{
};

} //namespace __internal

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
    constexpr Range
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
    constexpr auto
    base_range() -> decltype(pipeline_base_range<decltype(rng.base())>(rng.base()).base_range())
    {
        return pipeline_base_range<decltype(rng.base())>(rng.base()).base_range();
    };
};

template <typename _TupleType, typename _F, ::std::size_t... _Ip>
void
invoke(const _TupleType& __t, _F __f, ::std::index_sequence<_Ip...>)
{
    __f(::std::get<_Ip>(__t)...);
}

template <typename... _Ranges>
class zip_view
{
    static_assert(sizeof...(_Ranges) > 0, "Cannot instantiate zip_view with empty template parameter pack");

    using _tuple_ranges_t = oneapi::dpl::__internal::tuple<_Ranges...>;

    template <typename Idx, ::std::size_t... _Ip>
    auto
    make_reference(_tuple_ranges_t __t, Idx __i, ::std::index_sequence<_Ip...>) const
        -> decltype(oneapi::dpl::__internal::tuple<decltype(::std::declval<_Ranges&>().operator[](__i))...>(
            ::std::get<_Ip>(__t).operator[](__i)...))
    {
        return oneapi::dpl::__internal::tuple<decltype(::std::declval<_Ranges&>().operator[](__i))...>(
            ::std::get<_Ip>(__t).operator[](__i)...);
    }

  public:
    using value_type = oneapi::dpl::__internal::tuple<oneapi::dpl::__internal::__value_t<_Ranges>...>;
    static constexpr ::std::size_t __num_ranges = sizeof...(_Ranges);

    explicit zip_view(_Ranges... __args) : __m_ranges(oneapi::dpl::__internal::make_tuple(__args...)) {}

    auto
    size() const -> decltype(::std::get<0>(::std::declval<_tuple_ranges_t>()).size())
    {
        return ::std::get<0>(__m_ranges).size();
    }

    //TODO: C++ Standard states that the operator[] index should be the diff_type of the underlying range.
    template <typename Idx>
    constexpr auto operator[](Idx __i) const
        -> decltype(make_reference(::std::declval<_tuple_ranges_t>(), __i, ::std::make_index_sequence<__num_ranges>()))
    {
        return make_reference(__m_ranges, __i, ::std::make_index_sequence<__num_ranges>());
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
    using diff_type = typename ::std::iterator_traits<_Iterator>::difference_type;

  public:
    using value_type = typename ::std::iterator_traits<_Iterator>::value_type;

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

    //TODO: to be consistent with C++ standard, this Idx should be changed to diff_type of underlying iterator
    template <typename Idx>
    auto operator[](Idx i) const -> decltype(begin()[i])
    {
        return begin()[i];
    }

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

//It is kind of pseudo-view for reverse_view support.
template <typename _R>
struct reverse_view_simple
{
    using value_type = typename ::std::decay_t<_R>::value_type;

    _R __r;

    reverse_view_simple(_R __rng) : __r(__rng) {}

    //TODO: to be consistent with C++ standard, this Idx should be changed to diff_type of underlying range
    template <typename Idx>
    auto operator[](Idx __i) const -> decltype(__r[__i])
    {
        return __r[size() - __i - 1];
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

//It is kind of pseudo-view for take_view support. We assume that the underlying range will not shrink
//after creation of the view to favor performance.
template <typename _R, typename _Size>
struct take_view_simple
{
    using value_type = typename ::std::decay_t<_R>::value_type;

    _R __r;
    _Size __n;

    take_view_simple(_R __rng, _Size __size) : __r(__rng), __n(__size) { assert(__n >= 0 && __n <= __r.size()); }

    //TODO: to be consistent with C++ standard, this Idx should be changed to diff_type of underlying range
    template <typename Idx>
    auto operator[](Idx __i) const -> decltype(__r[__i])
    {
        return __r[__i];
    }

    _Size
    size() const
    {
        return __n;
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

//It is kind of pseudo-view for drop_view support. We assume that the underlying range will not shrink
//after creation of the view to favor performance.
template <typename _R, typename _Size>
struct drop_view_simple
{
    using value_type = typename ::std::decay_t<_R>::value_type;

    _R __r;
    _Size __n;

    drop_view_simple(_R __rng, _Size __size) : __r(__rng), __n(__size) { assert(__n >= 0 && __n <= __r.size()); }

    //TODO: to be consistent with C++ standard, this Idx should be changed to diff_type of underlying range
    template <typename Idx>
    auto operator[](Idx __i) const -> decltype(__r[__i])
    {
        return __r[__n + __i];
    }

    _Size
    size() const
    {
        return __r.size() - __n;
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

//replicate_start_view_simple inserts replicates of the first element m times, then continues with the range as normal.
// For counting iterator range {0,1,2,3,4,5,...}, and __replicate_count = 3, the result is {0,0,0,0,1,2,3,4,5,...}
template <typename _R, typename _Size>
struct replicate_start_view_simple
{
    using value_type = typename ::std::decay_t<_R>::value_type;

    _R __r;
    _Size __repl_count;

    replicate_start_view_simple(_R __rng, _Size __replicate_count) : __r(__rng), __repl_count(__replicate_count)
    {
        assert(__repl_count >= 0);
    }

    //TODO: to be consistent with C++ standard, this Idx should be changed to diff_type of underlying range
    template <typename Idx>
    auto operator[](Idx __i) const -> decltype(__r[__i])
    {
        return (__i < __repl_count) ? __r[0] : __r[__i - __repl_count];
    }

    _Size
    size() const
    {
        // if base range is empty, replication does not extend the valid size
        return (__r.empty()) ? 0 : __r.size() + __repl_count;
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

//It is kind of pseudo-view for transfom_iterator support.
template <typename _R, typename _F>
struct transform_view_simple
{
    using value_type = ::std::decay_t<::std::invoke_result_t<_F&, decltype(::std::declval<_R&>()[0])>>;

    _R __r;
    _F __f;

    //TODO: to be consistent with C++ standard, this Idx should be changed to diff_type of underlying range
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
test_map_view(int) -> decltype(::std::declval<_Map>()[0], ::std::true_type{});

template <typename _Map>
auto
test_map_view(...) -> ::std::false_type;

//pseudo-checking on viewable range concept
template <typename _Map>
struct is_map_view : decltype(test_map_view<_Map>(0))
{
};

//It is kind of pseudo-view for permutation_iterator support.
template <typename _R, typename _M, typename = void>
struct permutation_view_simple;

//permutation view: specialization for an index map functor
template <typename _R, typename _M>
struct permutation_view_simple<_R, _M, typename ::std::enable_if<oneapi::dpl::__internal::__is_functor<_M>>::type>
{
    using value_type = typename ::std::decay_t<_R>::value_type;
    using _Size = oneapi::dpl::__internal::__difference_t<_R>;

    _R __r;
    _M __map_fn;
    _Size __size;

    permutation_view_simple(_R __rng, _M __m, _Size __s) : __r(__rng), __map_fn(__m), __size(__s) {}

    //TODO: to be consistent with C++ standard, this Idx should be changed to diff_type of underlying range
    template <typename Idx>
    auto operator[](Idx __i) const -> decltype(__r[__map_fn(__i)])
    {
        return __r[__map_fn(__i)];
    }

    auto
    size() const -> decltype(__size)
    {
        return __size;
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
    using value_type = typename ::std::decay_t<_R>::value_type;

    zip_view<_R, _M> __data;

    permutation_view_simple(_R __r, _M __m) : __data(__r, __m) {}

    //TODO: to be consistent with C++ standard, this Idx should be changed to diff_type of underlying range
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
    using value_type = oneapi::dpl::internal::ignore_copyable;
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

#endif // _ONEDPL_UTILS_RANGES_H
