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

#ifndef _ONEDPL_PSTL_OFFLOAD_INTERNAL_ALGORITHM_IMPL_H
#define _ONEDPL_PSTL_OFFLOAD_INTERNAL_ALGORITHM_IMPL_H

#ifndef __SYCL_PSTL_OFFLOAD__
#error "__SYCL_PSTL_OFFLOAD__ macro should be defined to include this header"
#endif

#include <execution>
#include <utility>

#include <oneapi/dpl/algorithm>

#include "usm_memory_replacement.h"

namespace std {

template <class _ForwardIterator, class _Predicate>
bool any_of(const execution::parallel_unsequenced_policy&, _ForwardIterator __first, _ForwardIterator __last, _Predicate __pred)
{
    return oneapi::dpl::any_of(::__pstl_offload::__offload_policy_holder.__get_policy(), __first, __last, __pred);
}

template <class _ForwardIterator, class _Predicate>
bool all_of(const execution::parallel_unsequenced_policy&, _ForwardIterator __first, _ForwardIterator __last, _Predicate __pred)
{
    return oneapi::dpl::all_of(::__pstl_offload::__offload_policy_holder.__get_policy(), __first, __last, __pred);
}

template <class _ForwardIterator, class _Predicate>
bool none_of(const execution::parallel_unsequenced_policy&, _ForwardIterator __first, _ForwardIterator __last, _Predicate __pred)
{
    return oneapi::dpl::none_of(::__pstl_offload::__offload_policy_holder.__get_policy(), __first, __last, __pred);
}

template <class _ForwardIterator, class _Function>
void for_each(const execution::parallel_unsequenced_policy&, _ForwardIterator __first, _ForwardIterator __last, _Function __f)
{
    oneapi::dpl::for_each(::__pstl_offload::__offload_policy_holder.__get_policy(), __first, __last, __f);
}

template <class _ForwardIterator, class _Size, class _Function>
void for_each_n(const execution::parallel_unsequenced_policy&, _ForwardIterator __first, _Size __n, _Function __f)
{
    oneapi::dpl::for_each_n(::__pstl_offload::__offload_policy_holder.__get_policy(), __first, __n, __f);
}

template <class _ForwardIterator, class _Predicate>
_ForwardIterator find_if(const execution::parallel_unsequenced_policy&, _ForwardIterator __first, _ForwardIterator __last, _Predicate __pred)
{
    return oneapi::dpl::find_if(::__pstl_offload::__offload_policy_holder.__get_policy(), __first, __last, __pred);
}

template <class _ForwardIterator, class _Predicate>
_ForwardIterator find_if_not(const execution::parallel_unsequenced_policy&, _ForwardIterator __first, _ForwardIterator __last, _Predicate __pred)
{
    return oneapi::dpl::find_if_not(::__pstl_offload::__offload_policy_holder.__get_policy(), __first, __last, __pred);
}

template <class _ForwardIterator, class _Tp>
_ForwardIterator find(const execution::parallel_unsequenced_policy&, _ForwardIterator __first, _ForwardIterator __last, const _Tp& __value)
{
    return oneapi::dpl::find(::__pstl_offload::__offload_policy_holder.__get_policy(), __first, __last, __value);
}

template <class _ForwardIterator1, class _ForwardIterator2, class _BinaryPredicate>
_ForwardIterator1 find_end(const execution::parallel_unsequenced_policy&, _ForwardIterator1 __first, _ForwardIterator1 __last, _ForwardIterator2 __s_first,
                           _ForwardIterator2 __s_last, _BinaryPredicate __pred)
{
    return oneapi::dpl::find_end(::__pstl_offload::__offload_policy_holder.__get_policy(), __first, __last, __s_first, __s_last, __pred);
}

template <class _ForwardIterator1, class _ForwardIterator2>
_ForwardIterator1 find_end(const execution::parallel_unsequenced_policy&, _ForwardIterator1 __first, _ForwardIterator1 __last, _ForwardIterator2 __s_first,
                           _ForwardIterator2 __s_last)
{
    return oneapi::dpl::find_end(::__pstl_offload::__offload_policy_holder.__get_policy(), __first, __last, __s_first, __s_last);
}

template <class _ForwardIterator1, class _ForwardIterator2, class _BinaryPredicate>
_ForwardIterator1 find_first_of(const execution::parallel_unsequenced_policy&, _ForwardIterator1 __first, _ForwardIterator1 __last,
                                _ForwardIterator2 __s_first, _ForwardIterator2 __s_last, _BinaryPredicate __pred)
{
    return oneapi::dpl::find_first_of(::__pstl_offload::__offload_policy_holder.__get_policy(), __first, __last, __s_first, __s_last, __pred);
}

template <class _ForwardIterator1, class _ForwardIterator2>
_ForwardIterator1 find_first_of(const execution::parallel_unsequenced_policy&, _ForwardIterator1 __first, _ForwardIterator1 __last,
                                _ForwardIterator2 __s_first, _ForwardIterator2 __s_last)
{
    return oneapi::dpl::find_first_of(::__pstl_offload::__offload_policy_holder.__get_policy(), __first, __last, __s_first, __s_last);
}

template <class _ForwardIterator>
_ForwardIterator adjacent_find(const execution::parallel_unsequenced_policy&, _ForwardIterator __first, _ForwardIterator __last)
{
    return oneapi::dpl::adjacent_find(::__pstl_offload::__offload_policy_holder.__get_policy(), __first, __last);
}

template <class _ForwardIterator, class _BinaryPredicate>
_ForwardIterator adjacent_find(const execution::parallel_unsequenced_policy&, _ForwardIterator __first, _ForwardIterator __last, _BinaryPredicate __pred)
{
    return oneapi::dpl::adjacent_find(::__pstl_offload::__offload_policy_holder.__get_policy(), __first, __last, __pred);
}

template <class _ForwardIterator, class _Tp>
typename iterator_traits<_ForwardIterator>::difference_type
count(const execution::parallel_unsequenced_policy&, _ForwardIterator __first, _ForwardIterator __last, const _Tp& __value)
{
    return oneapi::dpl::count(::__pstl_offload::__offload_policy_holder.__get_policy(), __first, __last, __value);
}

template <class _ForwardIterator, class _Predicate>
typename iterator_traits<_ForwardIterator>::difference_type
count_if(const execution::parallel_unsequenced_policy&, _ForwardIterator __first, _ForwardIterator __last, _Predicate __pred)
{
    return oneapi::dpl::count_if(::__pstl_offload::__offload_policy_holder.__get_policy(), __first, __last, __pred);
}

template <class _ForwardIterator1, class _ForwardIterator2, class _BinaryPredicate>
_ForwardIterator1 search(const execution::parallel_unsequenced_policy&, _ForwardIterator1 __first, _ForwardIterator1 __last,
                         _ForwardIterator2 __s_first, _ForwardIterator2 __s_last, _BinaryPredicate __pred)
{
    return oneapi::dpl::search(::__pstl_offload::__offload_policy_holder.__get_policy(), __first, __last, __s_first, __s_last, __pred);
}

template <class _ForwardIterator1, class _ForwardIterator2>
_ForwardIterator1 search(const execution::parallel_unsequenced_policy&, _ForwardIterator1 __first, _ForwardIterator1 __last,
                         _ForwardIterator2 __s_first, _ForwardIterator2 __s_last)
{
    return oneapi::dpl::search(::__pstl_offload::__offload_policy_holder.__get_policy(), __first, __last, __s_first, __s_last);
}

template <class _ForwardIterator, class _Size, class _Tp, class _BinaryPredicate>
_ForwardIterator search_n(const execution::parallel_unsequenced_policy&, _ForwardIterator __first, _ForwardIterator __last, _Size __count,
                          const _Tp& __value, _BinaryPredicate __pred)
{
    return oneapi::dpl::search_n(::__pstl_offload::__offload_policy_holder.__get_policy(), __first, __last, __count, __value, __pred);
}

template <class _ForwardIterator, class _Size, class _Tp>
_ForwardIterator search_n(const execution::parallel_unsequenced_policy&, _ForwardIterator __first, _ForwardIterator __last, _Size __count,
                          const _Tp& __value)
{
    return oneapi::dpl::search_n(::__pstl_offload::__offload_policy_holder.__get_policy(), __first, __last, __count, __value);
}

template <class _ForwardIterator1, class _ForwardIterator2>
_ForwardIterator2 copy(const execution::parallel_unsequenced_policy&, _ForwardIterator1 __first, _ForwardIterator1 __last, _ForwardIterator2 __result)
{
    return oneapi::dpl::copy(::__pstl_offload::__offload_policy_holder.__get_policy(), __first, __last, __result);
}

template <class _ForwardIterator1, class _Size, class _ForwardIterator2>
_ForwardIterator2 copy_n(const execution::parallel_unsequenced_policy&, _ForwardIterator1 __first, _Size __n, _ForwardIterator2 __result)
{
    return oneapi::dpl::copy_n(::__pstl_offload::__offload_policy_holder.__get_policy(), __first, __n, __result);
}

template <class _ForwardIterator1, class _ForwardIterator2, class _Predicate>
_ForwardIterator2 copy_if(const execution::parallel_unsequenced_policy&, _ForwardIterator1 __first, _ForwardIterator1 __last, _ForwardIterator2 __result,
                          _Predicate __pred)
{
    return oneapi::dpl::copy_if(::__pstl_offload::__offload_policy_holder.__get_policy(), __first, __last, __result, __pred);
}

template <class _ForwardIterator1, class _ForwardIterator2>
_ForwardIterator2 swap_ranges(const execution::parallel_unsequenced_policy&, _ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2)
{
    return oneapi::dpl::swap_ranges(::__pstl_offload::__offload_policy_holder.__get_policy(), __first1, __last1, __first2);
}

template <class _ForwardIterator1, class _ForwardIterator2, class _ForwardIterator, class _BinaryOperation>
_ForwardIterator transform(const execution::parallel_unsequenced_policy&, _ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2,
                           _ForwardIterator __result, _BinaryOperation __op)
{
    return oneapi::dpl::transform(::__pstl_offload::__offload_policy_holder.__get_policy(), __first1, __last1, __first2, __result, __op);
}

template <class _ForwardIterator1, class _ForwardIterator2, class _UnaryOperation>
_ForwardIterator2 transform(const execution::parallel_unsequenced_policy&, _ForwardIterator1 __first, _ForwardIterator1 __last, _ForwardIterator2 __result,
                           _UnaryOperation __op)
{
    return oneapi::dpl::transform(::__pstl_offload::__offload_policy_holder.__get_policy(), __first, __last, __result, __op);
}

template <class _ForwardIterator, class _UnaryPredicate, class _Tp>
void replace_if(const execution::parallel_unsequenced_policy&, _ForwardIterator __first, _ForwardIterator __last, _UnaryPredicate __pred,
                const _Tp& __new_value)
{
    oneapi::dpl::replace_if(::__pstl_offload::__offload_policy_holder.__get_policy(), __first, __last, __pred, __new_value);
}

template <class _ForwardIterator, class _Tp>
void replace(const execution::parallel_unsequenced_policy&, _ForwardIterator __first, _ForwardIterator __last, const _Tp& __old_value,
             const _Tp& __new_value)
{
    oneapi::dpl::replace(::__pstl_offload::__offload_policy_holder.__get_policy(), __first, __last, __old_value, __new_value);
}

template <class _ForwardIterator1, class _ForwardIterator2, class _UnaryPredicate, class _Tp>
_ForwardIterator2 replace_copy_if(const execution::parallel_unsequenced_policy&, _ForwardIterator1 __first, _ForwardIterator1 __last,
                                  _ForwardIterator2 __result, _UnaryPredicate __pred, const _Tp& __new_value)
{
    return oneapi::dpl::replace_copy_if(::__pstl_offload::__offload_policy_holder.__get_policy(), __first, __last, __result, __pred, __new_value);
}

template <class _ForwardIterator1, class _ForwardIterator2, class _Tp>
_ForwardIterator2 replace_copy(const execution::parallel_unsequenced_policy&, _ForwardIterator1 __first, _ForwardIterator1 __last,
                               _ForwardIterator2 __result, const _Tp& __old_value, const _Tp& __new_value)
{
    return oneapi::dpl::replace_copy(::__pstl_offload::__offload_policy_holder.__get_policy(), __first, __last, __result, __old_value, __new_value);
}

template <class _ForwardIterator, class _Tp>
void fill(const execution::parallel_unsequenced_policy&, _ForwardIterator __first, _ForwardIterator __last, const _Tp& __value)
{
    oneapi::dpl::fill(::__pstl_offload::__offload_policy_holder.__get_policy(), __first, __last, __value);
}

template <class _ForwardIterator, class _Size, class _Tp>
_ForwardIterator fill_n(const execution::parallel_unsequenced_policy&, _ForwardIterator __first, _Size __count, const _Tp& __value)
{
    return oneapi::dpl::fill_n(::__pstl_offload::__offload_policy_holder.__get_policy(), __first, __count, __value);
}

template <class _ForwardIterator, class _Generator>
void generate(const execution::parallel_unsequenced_policy&, _ForwardIterator __first, _ForwardIterator __last, _Generator __g)
{
    oneapi::dpl::generate(::__pstl_offload::__offload_policy_holder.__get_policy(), __first, __last, __g);
}

template <class _ForwardIterator, class _Size, class _Generator>
_ForwardIterator generate_n(const execution::parallel_unsequenced_policy&, _ForwardIterator __first, _Size __count, _Generator __g)
{
    return oneapi::dpl::generate_n(::__pstl_offload::__offload_policy_holder.__get_policy(), __first, __count, __g);
}

template <class _ForwardIterator1, class _ForwardIterator2, class _Predicate>
_ForwardIterator2 remove_copy_if(const execution::parallel_unsequenced_policy&, _ForwardIterator1 __first, _ForwardIterator1 __last,
                                 _ForwardIterator2 __result, _Predicate __pred)
{
    return oneapi::dpl::remove_copy_if(::__pstl_offload::__offload_policy_holder.__get_policy(), __first, __last, __result, __pred);
}

template <class _ForwardIterator1, class _ForwardIterator2, class _Tp>
_ForwardIterator2 remove_copy(const execution::parallel_unsequenced_policy&, _ForwardIterator1 __first, _ForwardIterator1 __last,
                              _ForwardIterator2 __result, const _Tp& __value)
{
    return oneapi::dpl::remove_copy(::__pstl_offload::__offload_policy_holder.__get_policy(), __first, __last, __result, __value);
}

template <class _ForwardIterator, class _UnaryPredicate>
_ForwardIterator remove_if(const execution::parallel_unsequenced_policy&, _ForwardIterator __first, _ForwardIterator __last, _UnaryPredicate __pred)
{
    return oneapi::dpl::remove_if(::__pstl_offload::__offload_policy_holder.__get_policy(), __first, __last, __pred);
}

template <class _ForwardIterator, class _Tp>
_ForwardIterator remove(const execution::parallel_unsequenced_policy&, _ForwardIterator __first, _ForwardIterator __last, const _Tp& __value)
{
    return oneapi::dpl::remove(::__pstl_offload::__offload_policy_holder.__get_policy(), __first, __last, __value);
}

template <class _ForwardIterator, class _BinaryPredicate>
_ForwardIterator unique(const execution::parallel_unsequenced_policy&, _ForwardIterator __first, _ForwardIterator __last, _BinaryPredicate __pred)
{
    return oneapi::dpl::unique(::__pstl_offload::__offload_policy_holder.__get_policy(), __first, __last, __pred);
}

template <class _ForwardIterator>
_ForwardIterator unique(const execution::parallel_unsequenced_policy&, _ForwardIterator __first, _ForwardIterator __last)
{
    return oneapi::dpl::unique(::__pstl_offload::__offload_policy_holder.__get_policy(), __first, __last);
}

template <class _ForwardIterator1, class _ForwardIterator2, class _BinaryPredicate>
_ForwardIterator2 unique_copy(const execution::parallel_unsequenced_policy&, _ForwardIterator1 __first, _ForwardIterator1 __last, _ForwardIterator2 __result,
                              _BinaryPredicate __pred)
{
    return oneapi::dpl::unique_copy(::__pstl_offload::__offload_policy_holder.__get_policy(), __first, __last, __result, __pred);
}

template <class _ForwardIterator1, class _ForwardIterator2>
_ForwardIterator2 unique_copy(const execution::parallel_unsequenced_policy&, _ForwardIterator1 __first, _ForwardIterator1 __last, _ForwardIterator2 __result)
{
    return oneapi::dpl::unique_copy(::__pstl_offload::__offload_policy_holder.__get_policy(), __first, __last, __result);
}

template <class _BidirectionalIterator>
void reverse(const execution::parallel_unsequenced_policy&, _BidirectionalIterator __first, _BidirectionalIterator __last)
{
    return oneapi::dpl::reverse(::__pstl_offload::__offload_policy_holder.__get_policy(), __first, __last);
}

template <class _BidirectionalIterator, class _ForwardIterator>
_ForwardIterator reverse_copy(const execution::parallel_unsequenced_policy&, _BidirectionalIterator __first, _BidirectionalIterator __last,
                              _ForwardIterator __d_first)
{
    return oneapi::dpl::reverse_copy(::__pstl_offload::__offload_policy_holder.__get_policy(), __first, __last, __d_first);
}

template <class _ForwardIterator>
_ForwardIterator rotate(const execution::parallel_unsequenced_policy&, _ForwardIterator __first, _ForwardIterator __middle, _ForwardIterator __last)
{
    return oneapi::dpl::rotate(::__pstl_offload::__offload_policy_holder.__get_policy(), __first, __middle, __last);
}

template <class _ForwardIterator1, class _ForwardIterator2>
_ForwardIterator2 rotate_copy(const execution::parallel_unsequenced_policy&, _ForwardIterator1 __first, _ForwardIterator1 __middle, _ForwardIterator1 __last,
                              _ForwardIterator2 __result)
{
    return oneapi::dpl::rotate_copy(::__pstl_offload::__offload_policy_holder.__get_policy(), __first, __middle, __last, __result);
}

template <class _ForwardIterator, class _UnaryPredicate>
bool is_partitioned(const execution::parallel_unsequenced_policy&, _ForwardIterator __first, _ForwardIterator __last, _UnaryPredicate __pred)
{
    return oneapi::dpl::is_partitioned(::__pstl_offload::__offload_policy_holder.__get_policy(), __first, __last, __pred);
}

template <class _ForwardIterator, class _UnaryPredicate>
_ForwardIterator partition(const execution::parallel_unsequenced_policy&, _ForwardIterator __first, _ForwardIterator __last, _UnaryPredicate __pred)
{
    return oneapi::dpl::partition(::__pstl_offload::__offload_policy_holder.__get_policy(), __first, __last, __pred);
}

template <class _BidirectionalIterator, class _UnaryPredicate>
_BidirectionalIterator stable_partition(const execution::parallel_unsequenced_policy&, _BidirectionalIterator __first, _BidirectionalIterator __last,
                                        _UnaryPredicate __pred)
{
    return oneapi::dpl::stable_partition(::__pstl_offload::__offload_policy_holder.__get_policy(), __first, __last, __pred);
}

template <class _ForwardIterator, class _ForwardIterator1, class _ForwardIterator2, class _UnaryPredicate>
pair<_ForwardIterator1, _ForwardIterator2>
partition_copy(const execution::parallel_unsequenced_policy&, _ForwardIterator __first, _ForwardIterator __last,
               _ForwardIterator1 __out_true, _ForwardIterator2 __out_false, _UnaryPredicate __pred)
{
    return oneapi::dpl::partition_copy(::__pstl_offload::__offload_policy_holder.__get_policy(), __first, __last, __out_true, __out_false, __pred);
}

template <class _RandomAccessIterator, class _Compare>
void sort(const execution::parallel_unsequenced_policy&, _RandomAccessIterator __first, _RandomAccessIterator __last, _Compare __comp)
{
    oneapi::dpl::sort(::__pstl_offload::__offload_policy_holder.__get_policy(), __first, __last, __comp);
}

template <class _RandomAccessIterator>
void sort(const execution::parallel_unsequenced_policy&, _RandomAccessIterator __first, _RandomAccessIterator __last)
{
    oneapi::dpl::sort(::__pstl_offload::__offload_policy_holder.__get_policy(), __first, __last);
}

template <class _RandomAccessIterator, class _Compare>
void stable_sort(const execution::parallel_unsequenced_policy&, _RandomAccessIterator __first, _RandomAccessIterator __last, _Compare __comp)
{
    oneapi::dpl::stable_sort(::__pstl_offload::__offload_policy_holder.__get_policy(), __first, __last, __comp);
}

template <class _RandomAccessIterator>
void stable_sort(const execution::parallel_unsequenced_policy&, _RandomAccessIterator __first, _RandomAccessIterator __last)
{
    oneapi::dpl::stable_sort(::__pstl_offload::__offload_policy_holder.__get_policy(), __first, __last);
}

template <class _RandomAccessIterator1, class _RandomAccessIterator2, class _Compare>
void sort_by_key(const execution::parallel_unsequenced_policy&, _RandomAccessIterator1 __keys_first, _RandomAccessIterator1 __keys_last,
                 _RandomAccessIterator2 __values_first, _Compare __comp)
{
    oneapi::dpl::sort_by_key(::__pstl_offload::__offload_policy_holder.__get_policy(), __keys_first, __keys_last, __values_first, __comp);
}

template <class _RandomAccessIterator1, class _RandomAccessIterator2>
void sort_by_key(const execution::parallel_unsequenced_policy&, _RandomAccessIterator1 __keys_first, _RandomAccessIterator1 __keys_last,
                 _RandomAccessIterator2 __values_first)
{
    oneapi::dpl::sort_by_key(::__pstl_offload::__offload_policy_holder.__get_policy(), __keys_first, __keys_last, __values_first);
}

template <class _ForwardIterator1, class _ForwardIterator2, class _BinaryPredicate>
pair<_ForwardIterator1, _ForwardIterator2>
mismatch(const execution::parallel_unsequenced_policy&, _ForwardIterator1 __first1, _ForwardIterator1 __last1,
         _ForwardIterator2 __first2, _ForwardIterator2 __last2, _BinaryPredicate __pred)
{
    return oneapi::dpl::mismatch(::__pstl_offload::__offload_policy_holder.__get_policy(), __first1, __last1, __first2, __last2, __pred);
}

template <class _ForwardIterator1, class _ForwardIterator2, class _BinaryPredicate>
pair<_ForwardIterator1, _ForwardIterator2>
mismatch(const execution::parallel_unsequenced_policy&, _ForwardIterator1 __first1, _ForwardIterator1 __last1,
         _ForwardIterator2 __first2, _BinaryPredicate __pred)
{
    return oneapi::dpl::mismatch(::__pstl_offload::__offload_policy_holder.__get_policy(), __first1, __last1, __first2, __pred);
}

template <class _ForwardIterator1, class _ForwardIterator2>
pair<_ForwardIterator1, _ForwardIterator2>
mismatch(const execution::parallel_unsequenced_policy&, _ForwardIterator1 __first1, _ForwardIterator1 __last1,
         _ForwardIterator2 __first2, _ForwardIterator2 __last2)
{
    return oneapi::dpl::mismatch(::__pstl_offload::__offload_policy_holder.__get_policy(), __first1, __last1, __first2, __last2);
}

template <class _ForwardIterator1, class _ForwardIterator2>
pair<_ForwardIterator1, _ForwardIterator2>
mismatch(const execution::parallel_unsequenced_policy&, _ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2)
{
    return oneapi::dpl::mismatch(::__pstl_offload::__offload_policy_holder.__get_policy(), __first1, __last1, __first2);
}

template <class _ForwardIterator1, class _ForwardIterator2, class _BinaryPredicate>
bool equal(const execution::parallel_unsequenced_policy&, _ForwardIterator1 __first1, _ForwardIterator1 __last1,
           _ForwardIterator2 __first2, _BinaryPredicate __pred)
{
    return oneapi::dpl::equal(::__pstl_offload::__offload_policy_holder.__get_policy(), __first1, __last1, __first2, __pred);
}

template <class _ForwardIterator1, class _ForwardIterator2>
bool equal(const execution::parallel_unsequenced_policy&, _ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2)
{
    return oneapi::dpl::equal(::__pstl_offload::__offload_policy_holder.__get_policy(), __first1, __last1, __first2);
}

template <class _ForwardIterator1, class _ForwardIterator2, class _BinaryPredicate>
bool equal(const execution::parallel_unsequenced_policy&, _ForwardIterator1 __first1, _ForwardIterator1 __last1,
           _ForwardIterator2 __first2, _ForwardIterator2 __last2, _BinaryPredicate __pred)
{
    return oneapi::dpl::equal(::__pstl_offload::__offload_policy_holder.__get_policy(), __first1, __last1, __first2, __last2, __pred);
}

template <class _ForwardIterator1, class _ForwardIterator2>
bool equal(const execution::parallel_unsequenced_policy&, _ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2, _ForwardIterator2 __last2)
{
    return oneapi::dpl::equal(::__pstl_offload::__offload_policy_holder.__get_policy(), __first1, __last1, __first2, __last2);
}

template <class _ForwardIterator1, class _ForwardIterator2>
_ForwardIterator2 move(const execution::parallel_unsequenced_policy&, _ForwardIterator1 __first, _ForwardIterator1 __last, _ForwardIterator2 __d_first)
{
    return oneapi::dpl::move(::__pstl_offload::__offload_policy_holder.__get_policy(), __first, __last, __d_first);
}

template <class _RandomAccessIterator, class _Compare>
void partial_sort(const execution::parallel_unsequenced_policy&, _RandomAccessIterator __first, _RandomAccessIterator __middle, _RandomAccessIterator __last,
                  _Compare __comp)
{
    oneapi::dpl::partial_sort(::__pstl_offload::__offload_policy_holder.__get_policy(), __first, __middle, __last, __comp);
}

template <class _RandomAccessIterator>
void partial_sort(const execution::parallel_unsequenced_policy&, _RandomAccessIterator __first, _RandomAccessIterator __middle, _RandomAccessIterator __last)
{
    oneapi::dpl::partial_sort(::__pstl_offload::__offload_policy_holder.__get_policy(), __first, __middle, __last);
}

template <class _ForwardIterator, class _RandomAccessIterator, class _Compare>
_RandomAccessIterator partial_sort_copy(const execution::parallel_unsequenced_policy&, _ForwardIterator __first, _ForwardIterator __last,
                                        _RandomAccessIterator __d_first, _RandomAccessIterator __d_last, _Compare __comp)
{
    return oneapi::dpl::partial_sort_copy(::__pstl_offload::__offload_policy_holder.__get_policy(), __first, __last, __d_first, __d_last, __comp);
}

template <class _ForwardIterator, class _RandomAccessIterator>
_RandomAccessIterator partial_sort_copy(const execution::parallel_unsequenced_policy&, _ForwardIterator __first, _ForwardIterator __last,
                                        _RandomAccessIterator __d_first, _RandomAccessIterator __d_last)
{
    return oneapi::dpl::partial_sort_copy(::__pstl_offload::__offload_policy_holder.__get_policy(), __first, __last, __d_first, __d_last);
}

template <class _ForwardIterator, class _Compare>
_ForwardIterator is_sorted_until(const execution::parallel_unsequenced_policy&, _ForwardIterator __first, _ForwardIterator __last, _Compare __comp)
{
    return oneapi::dpl::is_sorted_until(::__pstl_offload::__offload_policy_holder.__get_policy(), __first, __last, __comp);
}

template <class _ForwardIterator>
_ForwardIterator is_sorted_until(const execution::parallel_unsequenced_policy&, _ForwardIterator __first, _ForwardIterator __last)
{
    return oneapi::dpl::is_sorted_until(::__pstl_offload::__offload_policy_holder.__get_policy(), __first, __last);
}

template <class _ForwardIterator, class _Compare>
bool is_sorted(const execution::parallel_unsequenced_policy&, _ForwardIterator __first, _ForwardIterator __last, _Compare __comp)
{
    return oneapi::dpl::is_sorted(::__pstl_offload::__offload_policy_holder.__get_policy(), __first, __last, __comp);
}

template <class _ForwardIterator>
bool is_sorted(const execution::parallel_unsequenced_policy&, _ForwardIterator __first, _ForwardIterator __last)
{
    return oneapi::dpl::is_sorted(::__pstl_offload::__offload_policy_holder.__get_policy(), __first, __last);
}

template <class _ForwardIterator1, class _ForwardIterator2, class _ForwardIterator, class _Compare>
_ForwardIterator merge(const execution::parallel_unsequenced_policy&, _ForwardIterator1 __first1, _ForwardIterator1 __last1,
                       _ForwardIterator2 __first2, _ForwardIterator2 __last2, _ForwardIterator __d_first, _Compare __comp)
{
    return oneapi::dpl::merge(::__pstl_offload::__offload_policy_holder.__get_policy(), __first1, __last1, __first2, __last2, __d_first, __comp);
}

template <class _ForwardIterator1, class _ForwardIterator2, class _ForwardIterator>
_ForwardIterator merge(const execution::parallel_unsequenced_policy&, _ForwardIterator1 __first1, _ForwardIterator1 __last1,
                       _ForwardIterator2 __first2, _ForwardIterator2 __last2, _ForwardIterator __d_first)
{
    return oneapi::dpl::merge(::__pstl_offload::__offload_policy_holder.__get_policy(), __first1, __last1, __first2, __last2, __d_first);
}

template <class _BidirectionalIterator, class _Compare>
void inplace_merge(const execution::parallel_unsequenced_policy&, _BidirectionalIterator __first, _BidirectionalIterator __middle,
                   _BidirectionalIterator __last, _Compare __comp)
{
    oneapi::dpl::inplace_merge(::__pstl_offload::__offload_policy_holder.__get_policy(), __first, __middle, __last, __comp);
}

template <class _BidirectionalIterator>
void inplace_merge(const execution::parallel_unsequenced_policy&, _BidirectionalIterator __first, _BidirectionalIterator __middle,
                   _BidirectionalIterator __last)
{
    oneapi::dpl::inplace_merge(::__pstl_offload::__offload_policy_holder.__get_policy(), __first, __middle, __last);
}

template <class _ForwardIterator1, class _ForwardIterator2, class _Compare>
bool includes(const execution::parallel_unsequenced_policy&, _ForwardIterator1 __first1, _ForwardIterator1 __last1,
              _ForwardIterator2 __first2, _ForwardIterator2 __last2, _Compare __comp)
{
    return oneapi::dpl::includes(::__pstl_offload::__offload_policy_holder.__get_policy(), __first1, __last1, __first2, __last2, __comp);
}

template <class _ForwardIterator1, class _ForwardIterator2>
bool includes(const execution::parallel_unsequenced_policy&, _ForwardIterator1 __first1, _ForwardIterator1 __last1,
              _ForwardIterator2 __first2, _ForwardIterator2 __last2)
{
    return oneapi::dpl::includes(::__pstl_offload::__offload_policy_holder.__get_policy(), __first1, __last1, __first2, __last2);
}

template <class _ForwardIterator1, class _ForwardIterator2, class _ForwardIterator, class _Compare>
_ForwardIterator set_union(const execution::parallel_unsequenced_policy&, _ForwardIterator1 __first1, _ForwardIterator1 __last1,
                           _ForwardIterator2 __first2, _ForwardIterator2 __last2, _ForwardIterator __result, _Compare __comp)
{
    return oneapi::dpl::set_union(::__pstl_offload::__offload_policy_holder.__get_policy(), __first1, __last1, __first2, __last2, __result, __comp);
}

template <class _ForwardIterator1, class _ForwardIterator2, class _ForwardIterator>
_ForwardIterator set_union(const execution::parallel_unsequenced_policy&, _ForwardIterator1 __first1, _ForwardIterator1 __last1,
                           _ForwardIterator2 __first2, _ForwardIterator2 __last2, _ForwardIterator __result)
{
    return oneapi::dpl::set_union(::__pstl_offload::__offload_policy_holder.__get_policy(), __first1, __last1, __first2, __last2, __result);
}

template <class _ForwardIterator1, class _ForwardIterator2, class _ForwardIterator, class _Compare>
_ForwardIterator set_intersection(const execution::parallel_unsequenced_policy&, _ForwardIterator1 __first1, _ForwardIterator1 __last1,
                                  _ForwardIterator2 __first2, _ForwardIterator2 __last2, _ForwardIterator __result, _Compare __comp)
{
    return oneapi::dpl::set_intersection(::__pstl_offload::__offload_policy_holder.__get_policy(), __first1, __last1, __first2, __last2, __result, __comp);
}

template <class _ForwardIterator1, class _ForwardIterator2, class _ForwardIterator>
_ForwardIterator set_intersection(const execution::parallel_unsequenced_policy&, _ForwardIterator1 __first1, _ForwardIterator1 __last1,
                                  _ForwardIterator2 __first2, _ForwardIterator2 __last2, _ForwardIterator __result)
{
    return oneapi::dpl::set_intersection(::__pstl_offload::__offload_policy_holder.__get_policy(), __first1, __last1, __first2, __last2, __result);
}

template <class _ForwardIterator1, class _ForwardIterator2, class _ForwardIterator, class _Compare>
_ForwardIterator set_difference(const execution::parallel_unsequenced_policy&, _ForwardIterator1 __first1, _ForwardIterator1 __last1,
                                _ForwardIterator2 __first2, _ForwardIterator2 __last2, _ForwardIterator __result, _Compare __comp)
{
    return oneapi::dpl::set_difference(::__pstl_offload::__offload_policy_holder.__get_policy(), __first1, __last1, __first2, __last2, __result, __comp);
}

template <class _ForwardIterator1, class _ForwardIterator2, class _ForwardIterator>
_ForwardIterator set_difference(const execution::parallel_unsequenced_policy&, _ForwardIterator1 __first1, _ForwardIterator1 __last1,
                                _ForwardIterator2 __first2, _ForwardIterator2 __last2, _ForwardIterator __result)
{
    return oneapi::dpl::set_difference(::__pstl_offload::__offload_policy_holder.__get_policy(), __first1, __last1, __first2, __last2, __result);
}

template <class _ForwardIterator1, class _ForwardIterator2, class _ForwardIterator, class _Compare>
_ForwardIterator set_symmetric_difference(const execution::parallel_unsequenced_policy&, _ForwardIterator1 __first1, _ForwardIterator1 __last1,
                                          _ForwardIterator2 __first2, _ForwardIterator2 __last2, _ForwardIterator __result, _Compare __comp)
{
    return oneapi::dpl::set_symmetric_difference(::__pstl_offload::__offload_policy_holder.__get_policy(), __first1, __last1, __first2, __last2, __result, __comp);
}

template <class _ForwardIterator1, class _ForwardIterator2, class _ForwardIterator>
_ForwardIterator set_symmetric_difference(const execution::parallel_unsequenced_policy&, _ForwardIterator1 __first1, _ForwardIterator1 __last1,
                                          _ForwardIterator2 __first2, _ForwardIterator2 __last2, _ForwardIterator __result)
{
    return oneapi::dpl::set_symmetric_difference(::__pstl_offload::__offload_policy_holder.__get_policy(), __first1, __last1, __first2, __last2, __result);
}

template <class _RandomAccessIterator, class _Compare>
_RandomAccessIterator is_heap_until(const execution::parallel_unsequenced_policy&, _RandomAccessIterator __first, _RandomAccessIterator __last, _Compare __comp)
{
    return oneapi::dpl::is_heap_until(::__pstl_offload::__offload_policy_holder.__get_policy(), __first, __last, __comp);
}

template <class _RandomAccessIterator>
_RandomAccessIterator is_heap_until(const execution::parallel_unsequenced_policy&, _RandomAccessIterator __first, _RandomAccessIterator __last)
{
    return oneapi::dpl::is_heap_until(::__pstl_offload::__offload_policy_holder.__get_policy(), __first, __last);
}

template <class _RandomAccessIterator, class _Compare>
bool is_heap(const execution::parallel_unsequenced_policy&, _RandomAccessIterator __first, _RandomAccessIterator __last, _Compare __comp)
{
    return oneapi::dpl::is_heap(::__pstl_offload::__offload_policy_holder.__get_policy(), __first, __last, __comp);
}

template <class _RandomAccessIterator>
bool is_heap(const execution::parallel_unsequenced_policy&, _RandomAccessIterator __first, _RandomAccessIterator __last)
{
    return oneapi::dpl::is_heap(::__pstl_offload::__offload_policy_holder.__get_policy(), __first, __last);
}

template <class _ForwardIterator, class _Compare>
_ForwardIterator min_element(const execution::parallel_unsequenced_policy&, _ForwardIterator __first, _ForwardIterator __last, _Compare __comp)
{
    return oneapi::dpl::min_element(::__pstl_offload::__offload_policy_holder.__get_policy(), __first, __last, __comp);
}

template <class _ForwardIterator>
_ForwardIterator min_element(const execution::parallel_unsequenced_policy&, _ForwardIterator __first, _ForwardIterator __last)
{
    return oneapi::dpl::min_element(::__pstl_offload::__offload_policy_holder.__get_policy(), __first, __last);
}

template <class _ForwardIterator, class _Compare>
_ForwardIterator max_element(const execution::parallel_unsequenced_policy&, _ForwardIterator __first, _ForwardIterator __last, _Compare __comp)
{
    return oneapi::dpl::max_element(::__pstl_offload::__offload_policy_holder.__get_policy(), __first, __last, __comp);
}

template <class _ForwardIterator>
_ForwardIterator max_element(const execution::parallel_unsequenced_policy&, _ForwardIterator __first, _ForwardIterator __last)
{
    return oneapi::dpl::max_element(::__pstl_offload::__offload_policy_holder.__get_policy(), __first, __last);
}

template <class _ForwardIterator, class _Compare>
pair<_ForwardIterator, _ForwardIterator> minmax_element(const execution::parallel_unsequenced_policy&, _ForwardIterator __first, _ForwardIterator __last, _Compare __comp)
{
    return oneapi::dpl::minmax_element(::__pstl_offload::__offload_policy_holder.__get_policy(), __first, __last, __comp);
}

template <class _ForwardIterator>
pair<_ForwardIterator, _ForwardIterator> minmax_element(const execution::parallel_unsequenced_policy&, _ForwardIterator __first, _ForwardIterator __last)
{
    return oneapi::dpl::minmax_element(::__pstl_offload::__offload_policy_holder.__get_policy(), __first, __last);
}

template <class _RandomAccessIterator, class _Compare>
void nth_element(const execution::parallel_unsequenced_policy&, _RandomAccessIterator __first, _RandomAccessIterator __nth, _RandomAccessIterator __last, _Compare __comp)
{
    return oneapi::dpl::nth_element(::__pstl_offload::__offload_policy_holder.__get_policy(), __first, __nth, __last, __comp);
}

template <class _RandomAccessIterator>
void nth_element(const execution::parallel_unsequenced_policy&, _RandomAccessIterator __first, _RandomAccessIterator __nth, _RandomAccessIterator __last)
{
    return oneapi::dpl::nth_element(::__pstl_offload::__offload_policy_holder.__get_policy(), __first, __nth, __last);
}

template <class _ForwardIterator1, class _ForwardIterator2, class _Compare>
bool lexicographical_compare(const execution::parallel_unsequenced_policy&, _ForwardIterator1 __first1, _ForwardIterator1 __last1,
                             _ForwardIterator2 __first2, _ForwardIterator2 __last2, _Compare __comp)
{
    return oneapi::dpl::lexicographical_compare(::__pstl_offload::__offload_policy_holder.__get_policy(), __first1, __last1, __first2, __last2, __comp);
}

template <class _ForwardIterator1, class _ForwardIterator2>
bool lexicographical_compare(const execution::parallel_unsequenced_policy&, _ForwardIterator1 __first1, _ForwardIterator1 __last1,
                             _ForwardIterator2 __first2, _ForwardIterator2 __last2)
{
    return oneapi::dpl::lexicographical_compare(::__pstl_offload::__offload_policy_holder.__get_policy(), __first1, __last1, __first2, __last2);
}

template <class _ForwardIterator>
_ForwardIterator shift_left(const execution::parallel_unsequenced_policy&, _ForwardIterator __first, _ForwardIterator __last,
                            typename iterator_traits<_ForwardIterator>::difference_type __n)
{
    return oneapi::dpl::shift_left(::__pstl_offload::__offload_policy_holder.__get_policy(), __first, __last, __n);
}

template <class _ForwardIterator>
_ForwardIterator shift_right(const execution::parallel_unsequenced_policy&, _ForwardIterator __first, _ForwardIterator __last,
                             typename iterator_traits<_ForwardIterator>::difference_type __n)
{
    return oneapi::dpl::shift_right(::__pstl_offload::__offload_policy_holder.__get_policy(), __first, __last, __n);
}

} // namespace std

#endif // _ONEDPL_PSTL_OFFLOAD_INTERNAL_ALGORITHM_IMPL_H
