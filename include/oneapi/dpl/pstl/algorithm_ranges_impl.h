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

#ifndef _ONEDPL_ALGORITHM_RANGES_IMPL_H
#define _ONEDPL_ALGORITHM_RANGES_IMPL_H

#if _ONEDPL_CPP20_RANGES_PRESENT

#include <ranges>
#include "algorithm_fwd.h"

namespace oneapi
{
namespace dpl
{
namespace __internal
{
namespace __ranges
{

//---------------------------------------------------------------------------------------------------------------------
// pattern_for_each
//---------------------------------------------------------------------------------------------------------------------

template <typename _Tag, typename _ExecutionPolicy, typename _R, typename _Proj, typename _Fun>
void
__pattern_for_each_impl(_Tag __tag, _ExecutionPolicy&& __exec, _R&& __r, _Fun __f, _Proj __proj)
{
    static_assert(__is_parallel_tag_v<_Tag> || typename _Tag::__is_vector{});

    auto __f_1 = 
        [__f, __proj](auto&& __val) { std::invoke(__f, std::invoke(__proj, std::forward<decltype(__val)>(__val)));};

    oneapi::dpl::__internal::__pattern_walk1(__tag, std::forward<_ExecutionPolicy>(__exec), std::ranges::begin(__r),
        std::ranges::begin(__r) + std::ranges::size(__r), __f_1);
}

template <typename _IsVector, typename _ExecutionPolicy, typename _R, typename _Proj, typename _Fun>
void
__pattern_for_each(__parallel_tag<_IsVector> __tag, _ExecutionPolicy&& __exec, _R&& __r, _Fun __f, _Proj __proj)
{
    __pattern_for_each_impl(__tag, std::forward<_ExecutionPolicy>(__exec), std::forward<_R>(__r), __f, __proj);
}

template <typename _Tag, typename _ExecutionPolicy, typename _R, typename _Proj, typename _Fun>
void
__pattern_for_each(_Tag __tag, _ExecutionPolicy&& __exec, _R&& __r, _Fun __f, _Proj __proj)
{
    if constexpr(typename _Tag::__is_vector{})
        __pattern_for_each_impl(__tag, std::forward<_ExecutionPolicy>(__exec), std::forward<_R>(__r), __f, __proj);
    else
        std::ranges::for_each(std::forward<_R>(__r), __f, __proj);
}

//---------------------------------------------------------------------------------------------------------------------
// pattern_transform
//---------------------------------------------------------------------------------------------------------------------

template<typename _Tag, typename _ExecutionPolicy, typename _InRange, typename _OutRange, typename _F, typename _Proj>
void
__pattern_transform_impl(_Tag __tag, _ExecutionPolicy&& __exec, _InRange&& __in_r, _OutRange&& __out_r, _F __op,
                         _Proj __proj)
{
    static_assert(__is_parallel_tag_v<_Tag> || typename _Tag::__is_vector{});
    assert(std::ranges::size(__in_r) == std::ranges::size(__out_r));

    auto __unary_op = [__op, __proj](auto&& __val) -> decltype(auto) { 
        return std::invoke(__op, std::invoke(__proj, std::forward<decltype(__val)>(__val)));};

    oneapi::dpl::__internal::__pattern_walk2(__tag, std::forward<_ExecutionPolicy>(__exec), std::ranges::begin(__in_r),
        std::ranges::begin(__in_r) + std::ranges::size(__in_r), std::ranges::begin(__out_r),
        oneapi::dpl::__internal::__transform_functor<decltype(__unary_op)>{std::move(__unary_op)});
}

template<typename _IsVector, typename _ExecutionPolicy, typename _InRange, typename _OutRange, typename _F,
         typename _Proj>
void
__pattern_transform(__parallel_tag<_IsVector> __tag, _ExecutionPolicy&& __exec, _InRange&& __in_r, _OutRange&& __out_r,
                    _F __op, _Proj __proj)
{
    __pattern_transform_impl(__tag, std::forward<_ExecutionPolicy>(__exec), std::forward<_InRange>(__in_r),
                             std::forward<_OutRange>(__out_r), __op, __proj);
}

template<typename _Tag, typename _ExecutionPolicy, typename _InRange, typename _OutRange, typename _F, typename _Proj>
void
__pattern_transform(_Tag __tag, _ExecutionPolicy&& __exec, _InRange&& __in_r, _OutRange&& __out_r,
                    _F __op, _Proj __proj)
{
    if constexpr(typename _Tag::__is_vector{})
        __pattern_transform_impl(__tag, std::forward<_ExecutionPolicy>(__exec), std::forward<_InRange>(__in_r),
                                 std::forward<_OutRange>(__out_r), __op, __proj);
    else
        std::ranges::transform(std::forward<_InRange>(__in_r), std::ranges::begin(__out_r), __op, __proj);
}

//---------------------------------------------------------------------------------------------------------------------
// pattern_transform (binary vesrion)
//---------------------------------------------------------------------------------------------------------------------

template<typename _Tag, typename _ExecutionPolicy, typename _InRange1, typename _InRange2, typename _OutRange,
         typename _F, typename _Proj1, typename _Proj2>
void
__pattern_transform_impl(_Tag __tag, _ExecutionPolicy&& __exec, _InRange1&& __in_r1, _InRange2&& __in_r2,
                         _OutRange&& __out_r, _F __binary_op, _Proj1 __proj1,_Proj2 __proj2)
{
    static_assert(__is_parallel_tag_v<_Tag> || typename _Tag::__is_vector{});

    auto __f = [__binary_op, __proj1, __proj2](auto&& __val1, auto&& __val2) -> decltype(auto) { 
        return std::invoke(__binary_op, std::invoke(__proj1, std::forward<decltype(__val1)>(__val1)),
            std::invoke(__proj2, std::forward<decltype(__val2)>(__val2)));};

    oneapi::dpl::__internal::__pattern_walk3(__tag, std::forward<_ExecutionPolicy>(__exec), std::ranges::begin(__in_r1),
        std::ranges::begin(__in_r1) + std::ranges::size(__in_r1), std::ranges::begin(__in_r2),
        std::ranges::begin(__out_r), oneapi::dpl::__internal::__transform_functor<decltype(__f)>{std::move(__f)});
}

template<typename _IsVector, typename _ExecutionPolicy, typename _InRange1, typename _InRange2, typename _OutRange, typename _F,
         typename _Proj1, typename _Proj2>
void
__pattern_transform(__parallel_tag<_IsVector> __tag, _ExecutionPolicy&& __exec, _InRange1&& __in_r1,
                    _InRange2&& __in_r2, _OutRange&& __out_r, _F __binary_op, _Proj1 __proj1, _Proj2 __proj2)
{
    __pattern_transform_impl(__tag, std::forward<_ExecutionPolicy>(__exec), std::forward<_InRange1>(__in_r1),
                             std::forward<_InRange2>(__in_r2), std::forward<_OutRange>(__out_r), __binary_op,
                             __proj1, __proj2);
}

template<typename _Tag, typename _ExecutionPolicy, typename _InRange1, typename _InRange2, typename _OutRange, typename _F,
         typename _Proj1, typename _Proj2>
void
__pattern_transform(_Tag __tag, _ExecutionPolicy&& __exec, _InRange1&& __in_r1, _InRange2&& __in_r2, _OutRange&& __out_r,
                    _F __binary_op, _Proj1 __proj1, _Proj2 __proj2)
{
    if constexpr(typename _Tag::__is_vector{})
        __pattern_transform_impl(__tag, std::forward<_ExecutionPolicy>(__exec), std::forward<_InRange1>(__in_r1),
                                        std::forward<_InRange2>(__in_r2), std::forward<_OutRange>(__out_r), __binary_op,
                                        __proj1, __proj2);
    else
        std::ranges::transform(std::forward<_InRange1>(__in_r1), std::forward<_InRange2>(__in_r2),
            std::ranges::begin(__out_r), __binary_op, __proj1, __proj2);
}

//---------------------------------------------------------------------------------------------------------------------
// pattern_find_if
//---------------------------------------------------------------------------------------------------------------------

template <typename _Tag, typename _ExecutionPolicy, typename _R, typename _Proj, typename _Pred>
auto
__pattern_find_if_impl(_Tag __tag, _ExecutionPolicy&& __exec, _R&& __r, _Pred __pred, _Proj __proj)
{
    static_assert(__is_parallel_tag_v<_Tag> || typename _Tag::__is_vector{});

    auto __pred_1 = [__pred, __proj](auto&& __val) { 
        return std::invoke(__pred, std::invoke(__proj, std::forward<decltype(__val)>(__val)));};

    return std::ranges::borrowed_iterator_t<_R>(oneapi::dpl::__internal::__pattern_find_if(__tag,
        std::forward<_ExecutionPolicy>(__exec), std::ranges::begin(__r), std::ranges::begin(__r) +
        std::ranges::size(__r), __pred_1));
}

template <typename _IsVector, typename _ExecutionPolicy, typename _R, typename _Proj, typename _Pred>
auto
__pattern_find_if(__parallel_tag<_IsVector> __tag, _ExecutionPolicy&& __exec, _R&& __r, _Pred __pred, _Proj __proj)
{
    return __pattern_find_if_impl(__tag, std::forward<_ExecutionPolicy>(__exec), std::forward<_R>(__r), __pred,
                                  __proj);
}

template <typename _Tag, typename _ExecutionPolicy, typename _R, typename _Proj, typename _Pred>
auto
__pattern_find_if(_Tag __tag, _ExecutionPolicy&& __exec, _R&& __r, _Pred __pred, _Proj __proj)
{
    if constexpr(typename _Tag::__is_vector{})
        return __pattern_find_if_impl(__tag, std::forward<_ExecutionPolicy>(__exec), std::forward<_R>(__r), __pred, __proj);
    else
        return std::ranges::find_if(std::forward<_R>(__r), __pred, __proj);
}

//---------------------------------------------------------------------------------------------------------------------
// pattern_any_of
//---------------------------------------------------------------------------------------------------------------------

template <typename _Tag, typename _ExecutionPolicy, typename _R, typename _Proj, typename _Pred>
bool
__pattern_any_of_impl(_Tag __tag, _ExecutionPolicy&& __exec, _R&& __r, _Pred __pred, _Proj __proj)
{
    static_assert(__is_parallel_tag_v<_Tag> || typename _Tag::__is_vector{});

    auto __pred_1 = [__pred, __proj](auto&& __val) { 
        return std::invoke(__pred, std::invoke(__proj, std::forward<decltype(__val)>(__val)));};
    return oneapi::dpl::__internal::__pattern_any_of(__tag, std::forward<_ExecutionPolicy>(__exec),
        std::ranges::begin(__r), std::ranges::begin(__r) + std::ranges::size(__r), __pred_1);
}

template <typename _IsVector, typename _ExecutionPolicy, typename _R, typename _Proj, typename _Pred>
bool
__pattern_any_of(__parallel_tag<_IsVector> __tag, _ExecutionPolicy&& __exec, _R&& __r, _Pred __pred, _Proj __proj)
{
    return __pattern_any_of_impl(__tag, std::forward<_ExecutionPolicy>(__exec), std::forward<_R>(__r), __pred, __proj);
}

template <typename _Tag, typename _ExecutionPolicy, typename _R, typename _Proj, typename _Pred>
bool
__pattern_any_of(_Tag __tag, _ExecutionPolicy&& __exec, _R&& __r, _Pred __pred, _Proj __proj)
{
    if constexpr(typename _Tag::__is_vector{})
        return __pattern_any_of_impl(__tag, std::forward<_ExecutionPolicy>(__exec), std::forward<_R>(__r), __pred,
                                     __proj);
    else
        return std::ranges::any_of(std::forward<_R>(__r), __pred, __proj);
}

//---------------------------------------------------------------------------------------------------------------------
// pattern_adjacent_find
//---------------------------------------------------------------------------------------------------------------------

template <typename _Tag, typename _ExecutionPolicy, typename _R, typename _Proj, typename _Pred>
auto
__pattern_adjacent_find_impl(_Tag __tag, _ExecutionPolicy&& __exec, _R&& __r, _Pred __pred,
                             _Proj __proj)
{
    static_assert(__is_parallel_tag_v<_Tag> || typename _Tag::__is_vector{});

    auto __pred_2 = [__pred, __proj](auto&& __val, auto&& __next) { return std::invoke(__pred, std::invoke(__proj,
        std::forward<decltype(__val)>(__val)), std::invoke(__proj, std::forward<decltype(__next)>(__next)));};

    auto __res = oneapi::dpl::__internal::__pattern_adjacent_find(__tag, std::forward<_ExecutionPolicy>(__exec),
        std::ranges::begin(__r), std::ranges::begin(__r) + std::ranges::size(__r), __pred_2,
        oneapi::dpl::__internal::__first_semantic());
    return std::ranges::borrowed_iterator_t<_R>(__res);
}

template <typename _IsVector, typename _ExecutionPolicy, typename _R, typename _Proj, typename _Pred>
auto
__pattern_adjacent_find_ranges(__parallel_tag<_IsVector> __tag, _ExecutionPolicy&& __exec, _R&& __r, _Pred __pred,
                               _Proj __proj)
{
    return __pattern_adjacent_find_impl(__tag, std::forward<_ExecutionPolicy>(__exec), std::forward<_R>(__r), __pred,
                                        __proj);
}

template <typename _Tag, typename _ExecutionPolicy, typename _R, typename _Proj, typename _Pred>
auto
__pattern_adjacent_find_ranges(_Tag __tag, _ExecutionPolicy&& __exec, _R&& __r, _Pred __pred, _Proj __proj)
{
    if constexpr(typename _Tag::__is_vector{})
        return __pattern_adjacent_find_impl(__tag, std::forward<_ExecutionPolicy>(__exec), std::forward<_R>(__r),
                                            __pred, __proj);
    else
        return std::ranges::adjacent_find(std::forward<_R>(__r), __pred, __proj);
}

//---------------------------------------------------------------------------------------------------------------------
// pattern_search
//---------------------------------------------------------------------------------------------------------------------

template<typename _Tag, typename _ExecutionPolicy, typename _R1, typename _R2, typename _Pred, typename _Proj1,
         typename _Proj2>
auto
__pattern_search_impl(_Tag __tag, _ExecutionPolicy&& __exec, _R1&& __r1, _R2&& __r2, _Pred __pred,
                      _Proj1 __proj1, _Proj2 __proj2)
{
    static_assert(__is_parallel_tag_v<_Tag> || typename _Tag::__is_vector{});

    auto __pred_2 = [__pred, __proj1, __proj2](auto&& __val1, auto&& __val2) { return std::invoke(__pred,
        std::invoke(__proj1, std::forward<decltype(__val1)>(__val1)),
        std::invoke(__proj2, std::forward<decltype(__val2)>(__val2)));};

    auto __res = oneapi::dpl::__internal::__pattern_search(__tag, std::forward<_ExecutionPolicy>(__exec),
        std::ranges::begin(__r1), std::ranges::begin(__r1) + std::ranges::size(__r1), std::ranges::begin(__r2),
        std::ranges::begin(__r2) + std::ranges::size(__r2), __pred_2);

    return std::ranges::borrowed_subrange_t<_R1>(__res, __res == std::ranges::end(__r1) ? __res : __res + __r2.size());
}

template<typename _IsVector, typename _ExecutionPolicy, typename _R1, typename _R2, typename _Pred, typename _Proj1,
         typename _Proj2>
auto
__pattern_search(__parallel_tag<_IsVector> __tag, _ExecutionPolicy&& __exec, _R1&& __r1, _R2&& __r2, _Pred __pred,
                 _Proj1 __proj1, _Proj2 __proj2)
{
    return __pattern_search_impl(__tag, std::forward<_ExecutionPolicy>(__exec), std::forward<_R1>(__r1),
                                 std::forward<_R2>(__r2), __pred, __proj1, __proj2);
}

template<typename _Tag, typename _ExecutionPolicy, typename _R1, typename _R2, typename _Pred, typename _Proj1,
         typename _Proj2>
auto
__pattern_search(_Tag __tag, _ExecutionPolicy&& __exec, _R1&& __r1, _R2&& __r2, _Pred __pred, _Proj1 __proj1, _Proj2 __proj2)
{
    if constexpr(typename _Tag::__is_vector{})
        return __pattern_search_impl(__tag, std::forward<_ExecutionPolicy>(__exec), std::forward<_R1>(__r1),
                                     std::forward<_R2>(__r2), __pred, __proj1, __proj2);
    else
        return std::ranges::search(std::forward<_R1>(__r1), std::forward<_R2>(__r2), __pred, __proj1, __proj2);
}

//---------------------------------------------------------------------------------------------------------------------
// pattern_search_n
//---------------------------------------------------------------------------------------------------------------------

template<typename _Tag, typename _ExecutionPolicy, typename _R, typename _T, typename _Pred, typename _Proj>
auto
__pattern_search_n_impl(_Tag __tag, _ExecutionPolicy&& __exec, _R&& __r,
                        std::ranges::range_difference_t<_R> __count, const _T& __value, _Pred __pred, _Proj __proj)
{
    static_assert(__is_parallel_tag_v<_Tag> || typename _Tag::__is_vector{});

    auto __pred_2 = [__pred, __proj](auto&& __val1, auto&& __val2) { return std::invoke(__pred,
        std::invoke(__proj, std::forward<decltype(__val1)>(__val1)), std::forward<decltype(__val2)>(__val2));};

    auto __res = oneapi::dpl::__internal::__pattern_search_n(__tag, std::forward<_ExecutionPolicy>(__exec),
        std::ranges::begin(__r), std::ranges::begin(__r) + std::ranges::size(__r), __count, __value, __pred_2);

    return std::ranges::borrowed_subrange_t<_R>(__res, __res == std::ranges::end(__r) ? __res : __res + __count);
}

template<typename _IsVector, typename _ExecutionPolicy, typename _R, typename _T, typename _Pred, typename _Proj>
auto
__pattern_search_n(__parallel_tag<_IsVector> __tag, _ExecutionPolicy&& __exec, _R&& __r,
                   std::ranges::range_difference_t<_R> __count, const _T& __value, _Pred __pred, _Proj __proj)
{
    return __pattern_search_n_impl(__tag, std::forward<_ExecutionPolicy>(__exec), std::forward<_R>(__r), __count,
                                   __value, __pred, __proj);
}

template<typename _Tag, typename _ExecutionPolicy, typename _R, typename _T, typename _Pred, typename _Proj>
auto
__pattern_search_n(_Tag __tag, _ExecutionPolicy&& __exec, _R&& __r, std::ranges::range_difference_t<_R> __count, const _T& __value,
                   _Pred __pred, _Proj __proj)
{
    if constexpr(typename _Tag::__is_vector{})
        return __pattern_search_n_impl(__tag, std::forward<_ExecutionPolicy>(__exec), std::forward<_R>(__r), __count,
                                       __value, __pred, __proj);
    else
        return std::ranges::search_n(std::forward<_R>(__r), __count, __value, __pred, __proj);
}

//---------------------------------------------------------------------------------------------------------------------
// pattern_count_if
//---------------------------------------------------------------------------------------------------------------------

template <typename _Tag, typename _ExecutionPolicy, typename _R, typename _Proj, typename _Pred>
std::ranges::range_difference_t<_R>
__pattern_count_if_impl(_Tag __tag, _ExecutionPolicy&& __exec, _R&& __r, _Pred __pred, _Proj __proj)
{
    static_assert(__is_parallel_tag_v<_Tag> || typename _Tag::__is_vector{});

    auto __pred_1 = [__pred, __proj](auto&& __val) { return std::invoke(__pred, std::invoke(__proj, __val));};
    return oneapi::dpl::__internal::__pattern_count(__tag, std::forward<_ExecutionPolicy>(__exec),
        std::ranges::begin(__r), std::ranges::begin(__r) + __r.size(), __pred_1);
}

template <typename _IsVector, typename _ExecutionPolicy, typename _R, typename _Proj, typename _Pred>
std::ranges::range_difference_t<_R>
__pattern_count_if(__parallel_tag<_IsVector> __tag, _ExecutionPolicy&& __exec, _R&& __r, _Pred __pred, _Proj __proj)
{
    return __pattern_count_if_impl(__tag, std::forward<_ExecutionPolicy>(__exec), std::forward<_R>(__r), __pred, __proj);
}

template <typename _Tag, typename _ExecutionPolicy, typename _R, typename _Proj, typename _Pred>
std::ranges::range_difference_t<_R>
__pattern_count_if(_Tag __tag, _ExecutionPolicy&& __exec, _R&& __r, _Pred __pred, _Proj __proj)
{
    if constexpr(typename _Tag::__is_vector{})
        return __pattern_count_if_impl(__tag, std::forward<_ExecutionPolicy>(__exec), std::forward<_R>(__r), __pred,
                                     __proj);
    else
        return std::ranges::count_if(std::forward<_R>(__r), __pred, __proj);
}

//---------------------------------------------------------------------------------------------------------------------
// pattern_equal
//---------------------------------------------------------------------------------------------------------------------
template<typename _Tag, typename _ExecutionPolicy, typename _R1, typename _R2, typename _Pred, typename _Proj1,
         typename _Proj2>
bool
__pattern_equal_impl(_Tag __tag, _ExecutionPolicy&& __exec, _R1&& __r1, _R2&& __r2, _Pred __pred,
                 _Proj1 __proj1, _Proj2 __proj2)
{
    static_assert(__is_parallel_tag_v<_Tag> || typename _Tag::__is_vector{});

    auto __pred_2 = [__pred, __proj1, __proj2](auto&& __val1, auto&& __val2)
        { return std::invoke(__pred, std::invoke(__proj1, __val1), std::invoke(__proj2, __val2));};

    return oneapi::dpl::__internal::__pattern_equal(__tag, std::forward<_ExecutionPolicy>(__exec),
        std::ranges::begin(__r1), std::ranges::begin(__r1) + __r1.size(), std::ranges::begin(__r2),
        std::ranges::begin(__r2) + __r2.size(), __pred_2);
}

template<typename _IsVector, typename _ExecutionPolicy, typename _R1, typename _R2, typename _Pred, typename _Proj1,
         typename _Proj2>
bool
__pattern_equal(__parallel_tag<_IsVector> __tag, _ExecutionPolicy&& __exec, _R1&& __r1, _R2&& __r2, _Pred __pred,
                 _Proj1 __proj1, _Proj2 __proj2)
{
    return __pattern_equal_impl(__tag, std::forward<_ExecutionPolicy>(__exec), std::forward<_R1>(__r1),
                                 std::forward<_R2>(__r2), __pred, __proj1, __proj2);
}

template<typename _Tag, typename _ExecutionPolicy, typename _R1, typename _R2, typename _Pred, typename _Proj1,
         typename _Proj2>
bool
__pattern_equal(_Tag __tag, _ExecutionPolicy&& __exec, _R1&& __r1, _R2&& __r2, _Pred __pred, _Proj1 __proj1, _Proj2 __proj2)
{
    if constexpr(typename _Tag::__is_vector{})
        return __pattern_equal_impl(__tag, std::forward<_ExecutionPolicy>(__exec), std::forward<_R1>(__r1),
                                     std::forward<_R2>(__r2), __pred, __proj1, __proj2);
    else
        return std::ranges::equal(std::forward<_R1>(__r1), std::forward<_R2>(__r2), __pred, __proj1, __proj2);
}

//---------------------------------------------------------------------------------------------------------------------
// pattern_is_sorted
//---------------------------------------------------------------------------------------------------------------------

template <typename _Tag, typename _ExecutionPolicy, typename _R, typename _Proj, typename _Comp>
bool
__pattern_is_sorted_impl(_Tag __tag, _ExecutionPolicy&& __exec, _R&& __r, _Comp __comp, _Proj __proj)
{
    static_assert(__is_parallel_tag_v<_Tag> || typename _Tag::__is_vector{});

    auto __pred_2 = [__comp, __proj](auto&& __val1, auto&& __val2)
        { return std::invoke(__comp, std::invoke(__proj, __val1), std::invoke(__proj, __val2));};

    return oneapi::dpl::__internal::__pattern_adjacent_find(__tag, std::forward<_ExecutionPolicy>(__exec),
        std::ranges::begin(__r), std::ranges::begin(__r) + __r.size(),
        oneapi::dpl::__internal::__reorder_pred(__pred_2), oneapi::dpl::__internal::__or_semantic()) == __r.end();
}

template <typename _IsVector, typename _ExecutionPolicy, typename _R, typename _Proj, typename _Comp>
bool
__pattern_is_sorted(__parallel_tag<_IsVector> __tag, _ExecutionPolicy&& __exec, _R&& __r, _Comp __comp, _Proj __proj)
{
    return __pattern_is_sorted_impl(__tag, std::forward<_ExecutionPolicy>(__exec), std::forward<_R>(__r), __comp, __proj);
}

template <typename _Tag, typename _ExecutionPolicy, typename _R, typename _Proj, typename _Comp>
bool
__pattern_is_sorted(_Tag __tag, _ExecutionPolicy&& __exec, _R&& __r, _Comp __comp, _Proj __proj)
{
    if constexpr(typename _Tag::__is_vector{})
        return __pattern_is_sorted_impl(__tag, std::forward<_ExecutionPolicy>(__exec), std::forward<_R>(__r), __comp,
                                     __proj);
    else
        return std::ranges::is_sorted(std::forward<_R>(__r), __comp, __proj);
}

//---------------------------------------------------------------------------------------------------------------------
// pattern_sort
//---------------------------------------------------------------------------------------------------------------------

template <typename _Tag, typename _ExecutionPolicy, typename _R, typename _Proj, typename _Comp>
auto
__pattern_sort_impl(_Tag __tag, _ExecutionPolicy&& __exec, _R&& __r, _Comp __comp, _Proj __proj)
{
    static_assert(__is_parallel_tag_v<_Tag> || typename _Tag::__is_vector{});

    auto __comp_2 = [__comp, __proj](auto&& __val1, auto&& __val2)
        { return std::invoke(__comp, std::invoke(__proj, __val1), std::invoke(__proj, __val2));};

    using _InputType  = std::ranges::range_value_t<_R>;
    oneapi::dpl::__internal::__pattern_sort(__tag, std::forward<_ExecutionPolicy>(__exec), std::ranges::begin(__r),
        std::ranges::begin(__r) + __r.size(), __comp_2);

    return std::ranges::borrowed_iterator_t<_R>(std::ranges::begin(__r) + __r.size());
}

template <typename _IsVector, typename _ExecutionPolicy, typename _R, typename _Proj, typename _Comp>
auto
__pattern_sort2(__parallel_tag<_IsVector> __tag, _ExecutionPolicy&& __exec, _R&& __r, _Comp __comp, _Proj __proj)
{
    return __pattern_sort_impl(__tag, std::forward<_ExecutionPolicy>(__exec), std::forward<_R>(__r), __comp, __proj);
}

template <typename _Tag, typename _ExecutionPolicy, typename _R, typename _Proj, typename _Comp>
auto
__pattern_sort2(_Tag __tag, _ExecutionPolicy&& __exec, _R&& __r, _Comp __comp, _Proj __proj)
{
    if constexpr(typename _Tag::__is_vector{})
        return __pattern_sort_impl(__tag, std::forward<_ExecutionPolicy>(__exec), std::forward<_R>(__r), __comp,
                                     __proj);
    else
        return std::ranges::stable_sort(std::forward<_R>(__r), __comp, __proj);
}

//---------------------------------------------------------------------------------------------------------------------
// pattern_min_element
//---------------------------------------------------------------------------------------------------------------------

template <typename _Tag, typename _ExecutionPolicy, typename _R, typename _Proj, typename _Comp>
auto
__pattern_min_element_impl(_Tag __tag, _ExecutionPolicy&& __exec, _R&& __r, _Comp __comp, _Proj __proj)
{
    static_assert(__is_parallel_tag_v<_Tag> || typename _Tag::__is_vector{});

    auto __comp_2 = [__comp, __proj](auto&& __val1, auto&& __val2)
        { return std::invoke(__comp, std::invoke(__proj, __val1), std::invoke(__proj, __val2));};

    auto __res = oneapi::dpl::__internal::__pattern_min_element(__tag, std::forward<_ExecutionPolicy>(__exec), std::ranges::begin(__r),
        std::ranges::begin(__r) + __r.size(), __comp_2);

    return std::ranges::borrowed_iterator_t<_R>(__res);
}

template <typename _IsVector, typename _ExecutionPolicy, typename _R, typename _Proj, typename _Comp>
auto
__pattern_min_element(__parallel_tag<_IsVector> __tag, _ExecutionPolicy&& __exec, _R&& __r, _Comp __comp, _Proj __proj)
{
    return __pattern_min_element_impl(__tag, std::forward<_ExecutionPolicy>(__exec), std::forward<_R>(__r), __comp, __proj);
}

template <typename _Tag, typename _ExecutionPolicy, typename _R, typename _Proj, typename _Comp>
auto
__pattern_min_element(_Tag __tag, _ExecutionPolicy&& __exec, _R&& __r, _Comp __comp, _Proj __proj)
{
    if constexpr(typename _Tag::__is_vector{})
        return __pattern_min_element_impl(__tag, std::forward<_ExecutionPolicy>(__exec), std::forward<_R>(__r), __comp,
                                     __proj);
    else
        return std::ranges::min_element(std::forward<_R>(__r), __comp, __proj);
}

//---------------------------------------------------------------------------------------------------------------------
// pattern_copy
//---------------------------------------------------------------------------------------------------------------------

template <typename _Tag, typename _ExecutionPolicy, typename _InRange, typename _OutRange>
auto
__pattern_copy_impl(_Tag __tag, _ExecutionPolicy&& __exec, _InRange&& __in_r, _OutRange&& __out_r)
{
    static_assert(__is_parallel_tag_v<_Tag> || typename _Tag::__is_vector{});

    assert(__in_r.size() == __out_r.size());

    oneapi::dpl::__internal::__pattern_walk2_brick(__tag, std::forward<_ExecutionPolicy>(__exec),
        std::ranges::begin(__in_r), std::ranges::begin(__in_r) + __in_r.size(), __out_r.begin(),
        oneapi::dpl::__internal::__brick_copy<decltype(__tag), _ExecutionPolicy>{});

    using __return_t = std::ranges::copy_result<std::ranges::borrowed_iterator_t<_InRange>,
        std::ranges::borrowed_iterator_t<_OutRange>>;

    return __return_t{__in_r.begin() + __in_r.size(), __out_r.begin() + __out_r.size()};
}

template <typename _IsVector, typename _ExecutionPolicy, typename _InRange, typename _OutRange>
auto
__pattern_copy(__parallel_tag<_IsVector> __tag, _ExecutionPolicy&& __exec, _InRange&& __in_r, _OutRange&& __out_r)
{
    return __pattern_copy_impl(__tag, std::forward<_ExecutionPolicy>(__exec), std::forward<_InRange>(__in_r),
        std::forward<_OutRange>(__out_r));
}

template<typename _Tag, typename _ExecutionPolicy, typename _InRange, typename _OutRange>
auto
__pattern_copy(_Tag __tag, _ExecutionPolicy&& __exec, _InRange&& __in_r, _OutRange&& __out_r)
{
    if constexpr(typename _Tag::__is_vector{})
        return __pattern_copy_impl(__tag, std::forward<_ExecutionPolicy>(__exec), std::forward<_InRange>(__in_r),
            std::forward<_OutRange>(__out_r));
    else
        return std::ranges::copy(std::forward<_InRange>(__in_r), __out_r.begin());
}
//---------------------------------------------------------------------------------------------------------------------
// pattern_copy_if
//---------------------------------------------------------------------------------------------------------------------

template <typename _Tag, typename _ExecutionPolicy, typename _InRange, typename _OutRange, typename _Pred,
          typename _Proj>
auto
__pattern_copy_if_impl(_Tag __tag, _ExecutionPolicy&& __exec, _InRange&& __in_r, _OutRange&& __out_r, _Pred __pred, _Proj __proj)
{
    static_assert(__is_parallel_tag_v<_Tag> || typename _Tag::__is_vector{});

    auto __pred_1 = [__pred, __proj](auto&& __val) { return std::invoke(__pred, std::invoke(__proj, __val));};

    auto __res_idx = oneapi::dpl::__internal::__pattern_copy_if(__tag, std::forward<_ExecutionPolicy>(__exec), std::ranges::begin(__in_r),
        std::ranges::begin(__in_r) + __in_r.size(), __out_r.begin(), __pred_1) - __out_r.begin();

    using __return_t = std::ranges::copy_if_result<std::ranges::borrowed_iterator_t<_InRange>,
        std::ranges::borrowed_iterator_t<_OutRange>>;

    return __return_t{__in_r.begin() + __in_r.size(), __out_r.begin() + __res_idx};
}

template <typename _IsVector, typename _ExecutionPolicy, typename _InRange, typename _OutRange, typename _Pred,
          typename _Proj>
auto
__pattern_copy_if_2(__parallel_tag<_IsVector> __tag, _ExecutionPolicy&& __exec, _InRange&& __in_r, _OutRange&& __out_r,
    _Pred __pred, _Proj __proj)
{
    return __pattern_copy_if_impl(__tag, std::forward<_ExecutionPolicy>(__exec), std::forward<_InRange>(__in_r),
        std::forward<_OutRange>(__out_r), __pred, __proj);
}

template<typename _Tag, typename _ExecutionPolicy, typename _InRange, typename _OutRange, typename _Pred, typename _Proj,
             oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, int> = 0>
auto
__pattern_copy_if_2(_Tag __tag, _ExecutionPolicy&& __exec, _InRange&& __in_r, _OutRange&& __out_r, _Pred __pred, _Proj __proj)
{
    if constexpr(typename _Tag::__is_vector{})
        return __pattern_copy_if_impl(__tag, std::forward<_ExecutionPolicy>(__exec), std::forward<_InRange>(__in_r),
            std::forward<_OutRange>(__out_r), __pred, __proj);
    else
        return std::ranges::copy_if(std::forward<_InRange>(__in_r), __out_r.begin(), __pred, __proj);
}

//---------------------------------------------------------------------------------------------------------------------
// pattern_merge
//---------------------------------------------------------------------------------------------------------------------

template<typename _Tag, typename _ExecutionPolicy, typename _R1, typename _R2, typename _OutRange, typename _Comp,
         typename _Proj1, typename _Proj2>
auto
__pattern_merge_impl(_Tag __tag, _ExecutionPolicy&& __exec, _R1&& __r1, _R2&& __r2, _OutRange&& __out_r, _Comp __comp,
    _Proj1 __proj1, _Proj2 __proj2)
{
    static_assert(__is_parallel_tag_v<_Tag> || typename _Tag::__is_vector{});

    auto __comp_2 = [__comp, __proj1, __proj2](auto&& __val1, auto&& __val2)
        { return std::invoke(__comp, std::invoke(__proj1, __val1), std::invoke(__proj2, __val2));};

    auto __res = oneapi::dpl::__internal::__pattern_merge(__tag, std::forward<_ExecutionPolicy>(__exec),
        std::ranges::begin(__r1), std::ranges::begin(__r1) + __r1.size(), std::ranges::begin(__r2),
        std::ranges::begin(__r2) + __r2.size(), __out_r.begin(), __comp_2);

    using __return_t = std::ranges::merge_result<std::ranges::borrowed_iterator_t<_R1>, std::ranges::borrowed_iterator_t<_R2>,
        std::ranges::borrowed_iterator_t<_OutRange>>;

    return __return_t{__r1.begin() + __r1.size(), __r2.begin() + __r2.size(), __res};
}

template<typename _IsVector, typename _ExecutionPolicy, typename _R1, typename _R2, typename _OutRange, typename _Comp,
         typename _Proj1, typename _Proj2>
auto
__pattern_merge(__parallel_tag<_IsVector> __tag, _ExecutionPolicy&& __exec, _R1&& __r1, _R2&& __r2, _OutRange&& __out_r,
    _Comp __comp, _Proj1 __proj1, _Proj2 __proj2)
{
    return __pattern_merge_impl(__tag, std::forward<_ExecutionPolicy>(__exec), std::forward<_R1>(__r1),
        std::forward<_R2>(__r2), std::forward<_OutRange>(__out_r), __comp, __proj1, __proj2);
}

template<typename _Tag, typename _ExecutionPolicy, typename _R1, typename _R2, typename _OutRange, typename _Comp,
         typename _Proj1, typename _Proj2>
auto
__pattern_merge(_Tag __tag, _ExecutionPolicy&& __exec, _R1&& __r1, _R2&& __r2, _OutRange&& __out_r, _Comp __comp,
    _Proj1 __proj1, _Proj2 __proj2)
{
    if constexpr(typename _Tag::__is_vector{})
        return __pattern_merge_impl(__tag, std::forward<_ExecutionPolicy>(__exec), std::forward<_R1>(__r1),
            std::forward<_R2>(__r2), std::forward<_OutRange>(__out_r), __comp, __proj1, __proj2);
    else
        return std::ranges::merge(std::forward<_R1>(__r1), std::forward<_R2>(__r2), __out_r.begin(), __comp, __proj1,
            __proj2);
}

} // namespace __ranges
} // namespace __internal
} // namespace dpl
} // namespace oneapi

#endif //_ONEDPL_CPP20_RANGES_PRESENT

#endif // _ONEDPL_ALGORITHM_RANGES_IMPL_H