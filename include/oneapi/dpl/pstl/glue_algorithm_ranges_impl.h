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

#ifndef _ONEDPL_GLUE_ALGORITHM_RANGES_IMPL_H
#define _ONEDPL_GLUE_ALGORITHM_RANGES_IMPL_H

#include "execution_defs.h"
#include "glue_algorithm_defs.h"

#if _ONEDPL_HETERO_BACKEND
#    include "hetero/algorithm_ranges_impl_hetero.h"
#    include "hetero/algorithm_impl_hetero.h" //TODO: for __brick_copy
#endif

namespace oneapi
{
namespace dpl
{
namespace experimental
{
namespace ranges
{

// [alg.any_of]

template <typename _ExecutionPolicy, typename _Range, typename _Predicate>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, bool>
any_of(_ExecutionPolicy&& __exec, _Range&& __rng, _Predicate __pred)
{
    return oneapi::dpl::__internal::__ranges::__pattern_any_of(::std::forward<_ExecutionPolicy>(__exec),
                                                               views::all_read(::std::forward<_Range>(__rng)), __pred);
}

// [alg.all_of]

template <typename _ExecutionPolicy, typename _Range, typename _Predicate>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, bool>
all_of(_ExecutionPolicy&& __exec, _Range&& __rng, _Predicate __pred)
{
    return !any_of(
        ::std::forward<_ExecutionPolicy>(__exec), ::std::forward<_Range>(__rng),
        oneapi::dpl::__internal::__not_pred<oneapi::dpl::__internal::__ref_or_copy<_ExecutionPolicy, _Predicate>>(
            __pred));
}

// [alg.none_of]

template <typename _ExecutionPolicy, typename _Range, typename _Predicate>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, bool>
none_of(_ExecutionPolicy&& __exec, _Range&& __rng, _Predicate __pred)
{
    return !any_of(::std::forward<_ExecutionPolicy>(__exec), ::std::forward<_Range>(__rng), __pred);
}

// [alg.foreach]

template <typename _ExecutionPolicy, typename _Range, typename _Function>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, void>
for_each(_ExecutionPolicy&& __exec, _Range&& __rng, _Function __f)
{
    oneapi::dpl::__internal::__ranges::__pattern_walk_n(::std::forward<_ExecutionPolicy>(__exec), __f,
                                                        views::all(::std::forward<_Range>(__rng)));
}

// [alg.find]

template <typename _ExecutionPolicy, typename _Range, typename _Predicate>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, oneapi::dpl::__internal::__difference_t<_Range>>
find_if(_ExecutionPolicy&& __exec, _Range&& __rng, _Predicate __pred)
{
    return oneapi::dpl::__internal::__ranges::__pattern_find_if(::std::forward<_ExecutionPolicy>(__exec),
                                                                views::all_read(::std::forward<_Range>(__rng)), __pred);
}

template <typename _ExecutionPolicy, typename _Range, typename _Predicate>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, oneapi::dpl::__internal::__difference_t<_Range>>
find_if_not(_ExecutionPolicy&& __exec, _Range&& __rng, _Predicate __pred)
{
    return find_if(
        ::std::forward<_ExecutionPolicy>(__exec), ::std::forward<_Range>(__rng),
        oneapi::dpl::__internal::__not_pred<oneapi::dpl::__internal::__ref_or_copy<_ExecutionPolicy, _Predicate>>(
            __pred));
}

template <typename _ExecutionPolicy, typename _Range, typename _Tp>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, oneapi::dpl::__internal::__difference_t<_Range>>
find(_ExecutionPolicy&& __exec, _Range&& __rng, const _Tp& __value)
{
    return find_if(
        ::std::forward<_ExecutionPolicy>(__exec), ::std::forward<_Range>(__rng),
        oneapi::dpl::__internal::__equal_value<oneapi::dpl::__internal::__ref_or_copy<_ExecutionPolicy, const _Tp>>(
            __value));
}

// [alg.find.end]

template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _BinaryPredicate>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy,
                                                      oneapi::dpl::__internal::__difference_t<_Range1>>
find_end(_ExecutionPolicy&& __exec, _Range1&& __rng1, _Range2&& __rng2, _BinaryPredicate __pred)
{
    return oneapi::dpl::__internal::__ranges::__pattern_find_end(
        ::std::forward<_ExecutionPolicy>(__exec), views::all_read(::std::forward<_Range1>(__rng1)),
        views::all_read(::std::forward<_Range2>(__rng2)), __pred);
}

template <typename _ExecutionPolicy, typename _Range1, typename _Range2>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy,
                                                      oneapi::dpl::__internal::__difference_t<_Range1>>
find_end(_ExecutionPolicy&& __exec, _Range1&& __rng1, _Range2&& __rng2)
{
    return find_end(::std::forward<_ExecutionPolicy>(__exec), ::std::forward<_Range1>(__rng1),
                    ::std::forward<_Range2>(__rng2), oneapi::dpl::__internal::__pstl_equal());
}

// [alg.find_first_of]

template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _BinaryPredicate>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy,
                                                      oneapi::dpl::__internal::__difference_t<_Range1>>
find_first_of(_ExecutionPolicy&& __exec, _Range1&& __rng1, _Range2&& __rng2, _BinaryPredicate __pred)
{
    return oneapi::dpl::__internal::__ranges::__pattern_find_first_of(
        ::std::forward<_ExecutionPolicy>(__exec), views::all_read(::std::forward<_Range1>(__rng1)),
        views::all_read(::std::forward<_Range2>(__rng2)), __pred);
}

template <typename _ExecutionPolicy, typename _Range1, typename _Range2>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy,
                                                      oneapi::dpl::__internal::__difference_t<_Range1>>
find_first_of(_ExecutionPolicy&& __exec, _Range1&& __rng1, _Range2&& __rng2)
{
    return find_first_of(::std::forward<_ExecutionPolicy>(__exec), ::std::forward<_Range1>(__rng1),
                         ::std::forward<_Range2>(__rng2), oneapi::dpl::__internal::__pstl_equal());
}

// [alg.adjacent_find]

template <typename _ExecutionPolicy, typename _Range, typename _BinaryPredicate>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, oneapi::dpl::__internal::__difference_t<_Range>>
adjacent_find(_ExecutionPolicy&& __exec, _Range&& __rng, _BinaryPredicate __pred)
{
    return oneapi::dpl::__internal::__ranges::__pattern_adjacent_find(
        ::std::forward<_ExecutionPolicy>(__exec), views::all_read(::std::forward<_Range>(__rng)), __pred,
        oneapi::dpl::__internal::__first_semantic());
}

template <typename _ExecutionPolicy, typename _Range>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, oneapi::dpl::__internal::__difference_t<_Range>>
adjacent_find(_ExecutionPolicy&& __exec, _Range&& __rng)
{
    using _ValueType = oneapi::dpl::__internal::__value_t<_Range>;
    return adjacent_find(::std::forward<_ExecutionPolicy>(__exec), ::std::forward<_Range>(__rng),
                         ::std::equal_to<_ValueType>());
}

// [alg.count]

template <typename _ExecutionPolicy, typename _Range, typename _Predicate>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, oneapi::dpl::__internal::__difference_t<_Range>>
count_if(_ExecutionPolicy&& __exec, _Range&& __rng, _Predicate __pred)
{
    return oneapi::dpl::__internal::__ranges::__pattern_count(::std::forward<_ExecutionPolicy>(__exec),
                                                              views::all_read(::std::forward<_Range>(__rng)), __pred);
}

template <typename _ExecutionPolicy, typename _Range, typename _Tp>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, oneapi::dpl::__internal::__difference_t<_Range>>
count(_ExecutionPolicy&& __exec, _Range&& __rng, const _Tp& __value)
{
    return count_if(
        ::std::forward<_ExecutionPolicy>(__exec), ::std::forward<_Range>(__rng),
        oneapi::dpl::__internal::__equal_value<oneapi::dpl::__internal::__ref_or_copy<_ExecutionPolicy, const _Tp>>(
            __value));
}

// [alg.search]

template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _BinaryPredicate>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy,
                                                      oneapi::dpl::__internal::__difference_t<_Range1>>
search(_ExecutionPolicy&& __exec, _Range1&& __rng1, _Range2&& __rng2, _BinaryPredicate __pred)
{
    return oneapi::dpl::__internal::__ranges::__pattern_search(
        ::std::forward<_ExecutionPolicy>(__exec), views::all_read(::std::forward<_Range1>(__rng1)),
        views::all_read(::std::forward<_Range2>(__rng2)), __pred);
}

template <typename _ExecutionPolicy, typename _Range1, typename _Range2>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy,
                                                      oneapi::dpl::__internal::__difference_t<_Range1>>
search(_ExecutionPolicy&& __exec, _Range1&& __rng1, _Range2&& __rng2)
{
    return search(::std::forward<_ExecutionPolicy>(__exec), ::std::forward<_Range1>(__rng1),
                  ::std::forward<_Range2>(__rng2), oneapi::dpl::__internal::__pstl_equal());
}

template <typename _ExecutionPolicy, typename _Range, typename _Size, typename _Tp, typename _BinaryPredicate>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, oneapi::dpl::__internal::__difference_t<_Range>>
search_n(_ExecutionPolicy&& __exec, _Range&& __rng, _Size __count, const _Tp& __value, _BinaryPredicate __pred)
{
    return oneapi::dpl::__internal::__ranges::__pattern_search_n(::std::forward<_ExecutionPolicy>(__exec),
                                                                 views::all_read(::std::forward<_Range>(__rng)),
                                                                 __count, __value, __pred);
}

template <typename _ExecutionPolicy, typename _Range, typename _Size, typename _Tp>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, oneapi::dpl::__internal::__difference_t<_Range>>
search_n(_ExecutionPolicy&& __exec, _Range&& __rng, _Size __count, const _Tp& __value)
{
    return search_n(::std::forward<_ExecutionPolicy>(__exec), ::std::forward<_Range>(__rng), __count, __value,
                    oneapi::dpl::__internal::__pstl_equal());
}

// [alg.copy]

template <typename _ExecutionPolicy, typename _Range1, typename _Range2>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, void>
copy(_ExecutionPolicy&& __exec, _Range1&& __rng, _Range2&& __result)
{
    oneapi::dpl::__internal::__ranges::__pattern_walk_n(
        ::std::forward<_ExecutionPolicy>(__exec), oneapi::dpl::__internal::__brick_copy<_ExecutionPolicy>{},
        views::all_read(::std::forward<_Range1>(__rng)), views::all_write(::std::forward<_Range2>(__result)));
}

template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _Predicate>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy,
                                                      oneapi::dpl::__internal::__difference_t<_Range2>>
copy_if(_ExecutionPolicy&& __exec, _Range1&& __rng, _Range2&& __result, _Predicate __pred)
{
    return oneapi::dpl::__internal::__ranges::__pattern_copy_if(
        ::std::forward<_ExecutionPolicy>(__exec), views::all_read(::std::forward<_Range1>(__rng)),
        views::all_write(::std::forward<_Range2>(__result)), __pred, oneapi::dpl::__internal::__pstl_assign());
}

// [alg.swap]

template <typename _ExecutionPolicy, typename _Range1, typename _Range2>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy,
                                                      oneapi::dpl::__internal::__difference_t<_Range1>>
swap_ranges(_ExecutionPolicy&& __exec, _Range1&& __rng1, _Range2&& __rng2)
{
    using _ReferenceType1 = oneapi::dpl::__internal::__value_t<_Range1>&;
    using _ReferenceType2 = oneapi::dpl::__internal::__value_t<_Range2>&;

    return oneapi::dpl::__internal::__ranges::__pattern_swap(
        ::std::forward<_ExecutionPolicy>(__exec), views::all(::std::forward<_Range1>(__rng1)),
        views::all(::std::forward<_Range2>(__rng2)), [](_ReferenceType1 __x, _ReferenceType2 __y) {
            using ::std::swap;
            swap(__x, __y);
        });
}

// [alg.transform]

template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _UnaryOperation>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, void>
transform(_ExecutionPolicy&& __exec, _Range1&& __rng, _Range2&& __result, _UnaryOperation __op)
{
    oneapi::dpl::__internal::__ranges::__pattern_walk_n(
        ::std::forward<_ExecutionPolicy>(__exec), [__op](auto x, auto& z) { z = __op(x); },
        views::all_read(::std::forward<_Range1>(__rng)), views::all_write(::std::forward<_Range2>(__result)));
}

template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _Range3, typename _BinaryOperation>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, void>
transform(_ExecutionPolicy&& __exec, _Range1&& __rng1, _Range2&& __rng2, _Range3&& __result, _BinaryOperation __op)
{
    oneapi::dpl::__internal::__ranges::__pattern_walk_n(
        ::std::forward<_ExecutionPolicy>(__exec), [__op](auto x, auto y, auto& z) { z = __op(x, y); },
        views::all_read(::std::forward<_Range1>(__rng1)), views::all_read(::std::forward<_Range2>(__rng2)),
        views::all_write(::std::forward<_Range3>(__result)));
}

// [alg.remove]

template <typename _ExecutionPolicy, typename _Range, typename _UnaryPredicate>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, oneapi::dpl::__internal::__difference_t<_Range>>
remove_if(_ExecutionPolicy&& __exec, _Range&& __rng, _UnaryPredicate __pred)
{
    return oneapi::dpl::__internal::__ranges::__pattern_remove_if(::std::forward<_ExecutionPolicy>(__exec),
                                                                  views::all(::std::forward<_Range>(__rng)), __pred);
}

template <typename _ExecutionPolicy, typename _Range, typename _Tp>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, oneapi::dpl::__internal::__difference_t<_Range>>
remove(_ExecutionPolicy&& __exec, _Range&& __rng, const _Tp& __value)
{
    return remove_if(
        ::std::forward<_ExecutionPolicy>(__exec), ::std::forward<_Range>(__rng),
        oneapi::dpl::__internal::__equal_value<oneapi::dpl::__internal::__ref_or_copy<_ExecutionPolicy, const _Tp>>(
            __value));
}

template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _Predicate>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy,
                                                      oneapi::dpl::__internal::__difference_t<_Range2>>
remove_copy_if(_ExecutionPolicy&& __exec, _Range1&& __rng, _Range2&& __result, _Predicate __pred)
{
    return copy_if(
        ::std::forward<_ExecutionPolicy>(__exec), ::std::forward<_Range1>(__rng), ::std::forward<_Range2>(__result),
        oneapi::dpl::__internal::__not_pred<oneapi::dpl::__internal::__ref_or_copy<_ExecutionPolicy, _Predicate>>(
            __pred));
}

template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _Tp>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy,
                                                      oneapi::dpl::__internal::__difference_t<_Range2>>
remove_copy(_ExecutionPolicy&& __exec, _Range1&& __rng, _Range2&& __result, const _Tp& __value)
{
    return copy_if(
        ::std::forward<_ExecutionPolicy>(__exec), ::std::forward<_Range1>(__rng), ::std::forward<_Range2>(__result),
        oneapi::dpl::__internal::__not_equal_value<oneapi::dpl::__internal::__ref_or_copy<_ExecutionPolicy, const _Tp>>(
            __value));
}

// [alg.unique]

template <typename _ExecutionPolicy, typename _Range, typename _BinaryPredicate>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, oneapi::dpl::__internal::__difference_t<_Range>>
unique(_ExecutionPolicy&& __exec, _Range&& __rng, _BinaryPredicate __pred)
{
    return oneapi::dpl::__internal::__ranges::__pattern_unique(::std::forward<_ExecutionPolicy>(__exec),
                                                               views::all(::std::forward<_Range>(__rng)), __pred);
}

template <typename _ExecutionPolicy, typename _Range>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, oneapi::dpl::__internal::__difference_t<_Range>>
unique(_ExecutionPolicy&& __exec, _Range&& __rng)
{
    return unique(::std::forward<_ExecutionPolicy>(__exec), ::std::forward<_Range>(__rng),
                  oneapi::dpl::__internal::__pstl_equal());
}

template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _BinaryPredicate>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy,
                                                      oneapi::dpl::__internal::__difference_t<_Range2>>
unique_copy(_ExecutionPolicy&& __exec, _Range1&& __rng, _Range2&& __result, _BinaryPredicate __pred)
{
    return oneapi::dpl::__internal::__ranges::__pattern_unique_copy(
        ::std::forward<_ExecutionPolicy>(__exec), views::all_read(::std::forward<_Range1>(__rng)),
        views::all_write(::std::forward<_Range2>(__result)), __pred, oneapi::dpl::__internal::__pstl_assign());
}

template <typename _ExecutionPolicy, typename _Range1, typename _Range2>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy,
                                                      oneapi::dpl::__internal::__difference_t<_Range2>>
unique_copy(_ExecutionPolicy&& __exec, _Range1&& __rng, _Range2&& __result)
{
    return unique_copy(::std::forward<_ExecutionPolicy>(__exec), ::std::forward<_Range1>(__rng),
                       ::std::forward<_Range2>(__result), oneapi::dpl::__internal::__pstl_equal());
}

// [alg.reverse]

template <typename _ExecutionPolicy, typename _Range>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, void>
reverse(_ExecutionPolicy&& __exec, _Range&& __rng)
{
    auto __v = views::all(::std::forward<_Range>(__rng));
    auto __n = __v.size();
    auto __n_2 = __n / 2;

    auto __r1 = __v | views::take(__n_2);
    auto __r2 = __v | views::reverse | views::take(__n_2);
    swap_ranges(::std::forward<_ExecutionPolicy>(__exec), __r1, __r2);
}

template <typename _ExecutionPolicy, typename _Range1, typename _Range2>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy,
                                                      oneapi::dpl::__internal::__difference_t<_Range1>>
reverse_copy(_ExecutionPolicy&& __exec, _Range1&& __rng, _Range2&& __result)
{
    auto __src = views::all_read(::std::forward<_Range1>(__rng));
    copy(::std::forward<_ExecutionPolicy>(__exec), __src | views::reverse, ::std::forward<_Range2>(__result));
    return __src.size();
}

template <typename _ExecutionPolicy, typename _Range1, typename _Range2>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy,
                                                      oneapi::dpl::__internal::__difference_t<_Range1>>
rotate_copy(_ExecutionPolicy&& __exec, _Range1&& __rng, oneapi::dpl::__internal::__difference_t<_Range1> __rotate_value,
            _Range2&& __result)
{
    auto __src = views::all_read(::std::forward<_Range1>(__rng));
    copy(::std::forward<_ExecutionPolicy>(__exec), __src | views::rotate(__rotate_value),
         ::std::forward<_Range2>(__result));
    return __src.size();
}

// [alg.replace]

template <typename _ExecutionPolicy, typename _Range, typename _UnaryPredicate, typename _Tp>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, void>
replace_if(_ExecutionPolicy&& __exec, _Range&& __rng, _UnaryPredicate __pred, const _Tp& __new_value)
{
    oneapi::dpl::__internal::__ranges::__pattern_walk_n(
        ::std::forward<_ExecutionPolicy>(__exec),
        oneapi::dpl::__internal::__replace_functor<
            oneapi::dpl::__internal::__ref_or_copy<_ExecutionPolicy, const _Tp>,
            oneapi::dpl::__internal::__ref_or_copy<_ExecutionPolicy, _UnaryPredicate>>(__new_value, __pred),
        views::all(::std::forward<_Range>(__rng)));
}

template <typename _ExecutionPolicy, typename _Range, typename _Tp>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, void>
replace(_ExecutionPolicy&& __exec, _Range&& __rng, const _Tp& __old_value, const _Tp& __new_value)
{
    replace_if(
        ::std::forward<_ExecutionPolicy>(__exec), ::std::forward<_Range>(__rng),
        oneapi::dpl::__internal::__equal_value<oneapi::dpl::__internal::__ref_or_copy<_ExecutionPolicy, const _Tp>>(
            __old_value),
        __new_value);
}

template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _UnaryPredicate, typename _Tp>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy,
                                                      oneapi::dpl::__internal::__difference_t<_Range1>>
replace_copy_if(_ExecutionPolicy&& __exec, _Range1&& __rng, _Range2&& __result, _UnaryPredicate __pred,
                const _Tp& __new_value)
{
    auto __src = views::all_read(::std::forward<_Range1>(__rng));
    oneapi::dpl::__internal::__ranges::__pattern_walk_n(
        ::std::forward<_ExecutionPolicy>(__exec),
        oneapi::dpl::__internal::__replace_copy_functor<
            oneapi::dpl::__internal::__ref_or_copy<_ExecutionPolicy, const _Tp>,
            typename ::std::conditional<
                oneapi::dpl::__internal::__is_const_callable_object<_UnaryPredicate>::value, _UnaryPredicate,
                oneapi::dpl::__internal::__ref_or_copy<_ExecutionPolicy, _UnaryPredicate>>::type>(__new_value, __pred),
        __src, views::all_write(::std::forward<_Range2>(__result)));
    return __src.size();
}

template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _Tp>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy,
                                                      oneapi::dpl::__internal::__difference_t<_Range1>>
replace_copy(_ExecutionPolicy&& __exec, _Range1&& __rng, _Range2&& __result, const _Tp& __old_value,
             const _Tp& __new_value)
{
    return replace_copy_if(
        ::std::forward<_ExecutionPolicy>(__exec), ::std::forward<_Range1>(__rng), ::std::forward<_Range2>(__result),
        oneapi::dpl::__internal::__equal_value<oneapi::dpl::__internal::__ref_or_copy<_ExecutionPolicy, const _Tp>>(
            __old_value),
        __new_value);
}

// [alg.sort]

template <typename _ExecutionPolicy, typename _Range, typename _Compare, typename _Proj>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, void>
sort(_ExecutionPolicy&& __exec, _Range&& __rng, _Compare __comp, _Proj __proj)
{
    oneapi::dpl::__internal::__ranges::__pattern_sort(::std::forward<_ExecutionPolicy>(__exec),
                                                      views::all(::std::forward<_Range>(__rng)), __comp, __proj);
}

template <typename _ExecutionPolicy, typename _Range>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, void>
sort(_ExecutionPolicy&& __exec, _Range&& __rng)
{
    sort(::std::forward<_ExecutionPolicy>(__exec), ::std::forward<_Range>(__rng),
         oneapi::dpl::__internal::__pstl_less());
}

// [stable.sort]

template <typename _ExecutionPolicy, typename _Range, typename _Compare>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, void>
stable_sort(_ExecutionPolicy&& __exec, _Range&& __rng, _Compare __comp)
{
    sort(::std::forward<_ExecutionPolicy>(__exec), ::std::forward<_Range>(__rng), __comp);
}

template <typename _ExecutionPolicy, typename _Range>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, void>
stable_sort(_ExecutionPolicy&& __exec, _Range&& __rng)
{
    sort(::std::forward<_ExecutionPolicy>(__exec), ::std::forward<_Range>(__rng),
         oneapi::dpl::__internal::__pstl_less());
}

// [is.sorted]

template <typename _ExecutionPolicy, typename _Range, typename _Compare>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, oneapi::dpl::__internal::__difference_t<_Range>>
is_sorted_until(_ExecutionPolicy&& __exec, _Range&& __rng, _Compare __comp)
{
    auto __view = views::all_read(::std::forward<_Range>(__rng));
    const auto __res = oneapi::dpl::__internal::__ranges::__pattern_adjacent_find(
        ::std::forward<_ExecutionPolicy>(__exec), __view, oneapi::dpl::__internal::__reorder_pred<_Compare>(__comp),
        oneapi::dpl::__internal::__first_semantic());

    return __res == __view.size() ? __res : __res + 1;
}

template <typename _ExecutionPolicy, typename _Range>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, oneapi::dpl::__internal::__difference_t<_Range>>
is_sorted_until(_ExecutionPolicy&& __exec, _Range&& __rng)
{
    return is_sorted_until(::std::forward<_ExecutionPolicy>(__exec), ::std::forward<_Range>(__rng),
                           oneapi::dpl::__internal::__pstl_less());
}

template <typename _ExecutionPolicy, typename _Range, typename _Compare>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, bool>
is_sorted(_ExecutionPolicy&& __exec, _Range&& __rng, _Compare __comp)
{
    auto __view = views::all_read(::std::forward<_Range>(__rng));
    return oneapi::dpl::__internal::__ranges::__pattern_adjacent_find(
               ::std::forward<_ExecutionPolicy>(__exec), __view,
               oneapi::dpl::__internal::__reorder_pred<_Compare>(__comp),
               oneapi::dpl::__internal::__or_semantic()) == __view.size();
}

template <typename _ExecutionPolicy, typename _Range>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, bool>
is_sorted(_ExecutionPolicy&& __exec, _Range&& __rng)
{
    return is_sorted(::std::forward<_ExecutionPolicy>(__exec), ::std::forward<_Range>(__rng),
                     oneapi::dpl::__internal::__pstl_less());
}

// [alg.equal]

template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _BinaryPredicate>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, bool>
equal(_ExecutionPolicy&& __exec, _Range1&& __rng1, _Range2&& __rng2, _BinaryPredicate __p)
{
    return oneapi::dpl::__internal::__ranges::__pattern_equal(::std::forward<_ExecutionPolicy>(__exec),
                                                              views::all_read(::std::forward<_Range1>(__rng1)),
                                                              views::all_read(::std::forward<_Range2>(__rng2)), __p);
}

template <typename _ExecutionPolicy, typename _Range1, typename _Range2>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, bool>
equal(_ExecutionPolicy&& __exec, _Range1&& __rng1, _Range2&& __rng2)
{
    return equal(::std::forward<_ExecutionPolicy>(__exec), ::std::forward<_Range1>(__rng1),
                 ::std::forward<_Range2>(__rng2), oneapi::dpl::__internal::__pstl_equal());
}

// [alg.move]

template <typename _ExecutionPolicy, typename _Range1, typename _Range2>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, void>
move(_ExecutionPolicy&& __exec, _Range1&& __rng1, _Range2&& __rng2)
{
    using _DecayedExecutionPolicy = typename ::std::decay<_ExecutionPolicy>::type;

    oneapi::dpl::__internal::__ranges::__pattern_walk_n(
        ::std::forward<_ExecutionPolicy>(__exec), oneapi::dpl::__internal::__brick_move<_DecayedExecutionPolicy>{},
        views::all_read(::std::forward<_Range1>(__rng1)), views::all_write(::std::forward<_Range2>(__rng2)));
}

// [alg.merge]

template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _Range3, typename _Compare>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy,
                                                      oneapi::dpl::__internal::__difference_t<_Range3>>
merge(_ExecutionPolicy&& __exec, _Range1&& __rng1, _Range2&& __rng2, _Range3&& __rng3, _Compare __comp)
{
    return oneapi::dpl::__internal::__ranges::__pattern_merge(
        ::std::forward<_ExecutionPolicy>(__exec), views::all_read(::std::forward<_Range1>(__rng1)),
        views::all_read(::std::forward<_Range2>(__rng2)), views::all_write(::std::forward<_Range3>(__rng3)), __comp);
}

template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _Range3>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy,
                                                      oneapi::dpl::__internal::__difference_t<_Range3>>
merge(_ExecutionPolicy&& __exec, _Range1&& __rng1, _Range2&& __rng2, _Range3&& __rng3)
{
    return merge(::std::forward<_ExecutionPolicy>(__exec), ::std::forward<_Range1>(__rng1),
                 ::std::forward<_Range2>(__rng2), ::std::forward<_Range3>(__rng3),
                 oneapi::dpl::__internal::__pstl_less());
}

// [alg.min.max]

template <typename _ExecutionPolicy, typename _Range, typename _Compare>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, oneapi::dpl::__internal::__difference_t<_Range>>
min_element(_ExecutionPolicy&& __exec, _Range&& __rng, _Compare __comp)
{
    return oneapi::dpl::__internal::__ranges::__pattern_min_element(
        ::std::forward<_ExecutionPolicy>(__exec), views::all_read(::std::forward<_Range>(__rng)), __comp);
}

template <typename _ExecutionPolicy, typename _Range>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, oneapi::dpl::__internal::__difference_t<_Range>>
min_element(_ExecutionPolicy&& __exec, _Range&& __rng)
{
    return min_element(::std::forward<_ExecutionPolicy>(__exec), ::std::forward<_Range>(__rng),
                       oneapi::dpl::__internal::__pstl_less());
}

template <typename _ExecutionPolicy, typename _Range, typename _Compare>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, oneapi::dpl::__internal::__difference_t<_Range>>
max_element(_ExecutionPolicy&& __exec, _Range&& __rng, _Compare __comp)
{
    return min_element(::std::forward<_ExecutionPolicy>(__exec), ::std::forward<_Range>(__rng),
                       oneapi::dpl::__internal::__reorder_pred<_Compare>(__comp));
}

template <typename _ExecutionPolicy, typename _Range>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, oneapi::dpl::__internal::__difference_t<_Range>>
max_element(_ExecutionPolicy&& __exec, _Range&& __rng)
{
    return min_element(::std::forward<_ExecutionPolicy>(__exec), ::std::forward<_Range>(__rng),
                       oneapi::dpl::__internal::__reorder_pred<oneapi::dpl::__internal::__pstl_less>(
                           oneapi::dpl::__internal::__pstl_less()));
}

template <typename _ExecutionPolicy, typename _Range, typename _Compare>
oneapi::dpl::__internal::__enable_if_execution_policy<
    _ExecutionPolicy,
    ::std::pair<oneapi::dpl::__internal::__difference_t<_Range>, oneapi::dpl::__internal::__difference_t<_Range>>>
minmax_element(_ExecutionPolicy&& __exec, _Range&& __rng, _Compare __comp)
{
    return oneapi::dpl::__internal::__ranges::__pattern_minmax_element(
        ::std::forward<_ExecutionPolicy>(__exec), views::all_read(::std::forward<_Range>(__rng)), __comp);
}

template <typename _ExecutionPolicy, typename _Range>
oneapi::dpl::__internal::__enable_if_execution_policy<
    _ExecutionPolicy,
    ::std::pair<oneapi::dpl::__internal::__difference_t<_Range>, oneapi::dpl::__internal::__difference_t<_Range>>>
minmax_element(_ExecutionPolicy&& __exec, _Range&& __rng)
{
    return minmax_element(::std::forward<_ExecutionPolicy>(__exec), ::std::forward<_Range>(__rng),
                          oneapi::dpl::__internal::__pstl_less());
}

template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _Range3, typename _Range4,
          typename _BinaryPredicate, typename _BinaryOperator>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy,
                                                      oneapi::dpl::__internal::__difference_t<_Range3>>
reduce_by_segment(_ExecutionPolicy&& __exec, _Range1&& __keys, _Range2&& __values, _Range3&& __out_keys,
                  _Range4&& __out_values, _BinaryPredicate __binary_pred, _BinaryOperator __binary_op)
{
    return oneapi::dpl::__internal::__ranges::__pattern_reduce_by_segment(
        ::std::forward<_ExecutionPolicy>(__exec), views::all_read(::std::forward<_Range1>(__keys)),
        views::all_read(::std::forward<_Range2>(__values)), views::all_write(::std::forward<_Range3>(__out_keys)),
        views::all_write(::std::forward<_Range4>(__out_values)), __binary_pred, __binary_op);
}

template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _Range3, typename _Range4,
          typename _BinaryPredicate>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy,
                                                      oneapi::dpl::__internal::__difference_t<_Range3>>
reduce_by_segment(_ExecutionPolicy&& __exec, _Range1&& __keys, _Range2&& __values, _Range3&& __out_keys,
                  _Range4&& __out_values, _BinaryPredicate __binary_pred)
{
    return reduce_by_segment(::std::forward<_ExecutionPolicy>(__exec), ::std::forward<_Range1>(__keys),
                             ::std::forward<_Range2>(__values), ::std::forward<_Range3>(__out_keys),
                             ::std::forward<_Range4>(__out_values), __binary_pred,
                             oneapi::dpl::__internal::__pstl_plus());
}

template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _Range3, typename _Range4>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy,
                                                      oneapi::dpl::__internal::__difference_t<_Range3>>
reduce_by_segment(_ExecutionPolicy&& __exec, _Range1&& __keys, _Range2&& __values, _Range3&& __out_keys,
                  _Range4&& __out_values)
{
    return reduce_by_segment(::std::forward<_ExecutionPolicy>(__exec), ::std::forward<_Range1>(__keys),
                             ::std::forward<_Range2>(__values), ::std::forward<_Range3>(__out_keys),
                             ::std::forward<_Range4>(__out_values), oneapi::dpl::__internal::__pstl_equal());
}

} // namespace ranges
} // namespace experimental
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_GLUE_ALGORITHM_RANGES_IMPL_H
