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

#ifndef _ONEDPL_NUMERIC_FWD_H
#define _ONEDPL_NUMERIC_FWD_H

#include <type_traits>
#include <utility>

#include "execution_defs.h"

namespace oneapi
{
namespace dpl
{
namespace __internal
{

//------------------------------------------------------------------------
// transform_reduce (version with two binary functions, according to draft N4659)
//------------------------------------------------------------------------

template <class _RandomAccessIterator1, class _RandomAccessIterator2, class _Tp, class _BinaryOperation1,
          class _BinaryOperation2>
_Tp __brick_transform_reduce(_RandomAccessIterator1, _RandomAccessIterator1, _RandomAccessIterator2, _Tp,
                             _BinaryOperation1, _BinaryOperation2,
                             /*__is_vector=*/::std::true_type) noexcept;

template <class _ForwardIterator1, class _ForwardIterator2, class _Tp, class _BinaryOperation1, class _BinaryOperation2>
_Tp __brick_transform_reduce(_ForwardIterator1, _ForwardIterator1, _ForwardIterator2, _Tp, _BinaryOperation1,
                             _BinaryOperation2,
                             /*__is_vector=*/::std::false_type) noexcept;

template <class _Tag, class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2, class _Tp,
          class _BinaryOperation1, class _BinaryOperation2>
_Tp
__pattern_transform_reduce(_Tag, _ExecutionPolicy&&, _ForwardIterator1, _ForwardIterator1, _ForwardIterator2, _Tp,
                           _BinaryOperation1, _BinaryOperation2) noexcept;

template <class _IsVector, class _ExecutionPolicy, class _RandomAccessIterator1, class _RandomAccessIterator2,
          class _Tp, class _BinaryOperation1, class _BinaryOperation2>
_Tp
__pattern_transform_reduce(__parallel_tag<_IsVector>, _ExecutionPolicy&&, _RandomAccessIterator1,
                           _RandomAccessIterator1, _RandomAccessIterator2, _Tp, _BinaryOperation1, _BinaryOperation2);

//------------------------------------------------------------------------
// transform_reduce (version with unary and binary functions)
//------------------------------------------------------------------------

template <class _RandomAccessIterator, class _Tp, class _UnaryOperation, class _BinaryOperation>
_Tp __brick_transform_reduce(_RandomAccessIterator, _RandomAccessIterator, _Tp, _BinaryOperation, _UnaryOperation,
                             /*is_vector=*/::std::true_type) noexcept;

template <class _ForwardIterator, class _Tp, class _BinaryOperation, class _UnaryOperation>
_Tp __brick_transform_reduce(_ForwardIterator, _ForwardIterator, _Tp, _BinaryOperation, _UnaryOperation,
                             /*is_vector=*/::std::false_type) noexcept;

template <class _Tag, class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2, class _Tp,
          class _BinaryOperation1, class _BinaryOperation2>
_Tp
__pattern_transform_reduce(_Tag, _ExecutionPolicy&&, _ForwardIterator1, _ForwardIterator1, _ForwardIterator2, _Tp,
                           _BinaryOperation1, _BinaryOperation2 __bnary_op2) noexcept;

template <class _IsVector, class _ExecutionPolicy, class _RandomAccessIterator1, class _RandomAccessIterator2,
          class _Tp, class _BinaryOperation1, class _BinaryOperation2>
_Tp
__pattern_transform_reduce(__parallel_tag<_IsVector>, _ExecutionPolicy&&, _RandomAccessIterator1,
                           _RandomAccessIterator1, _RandomAccessIterator2, _Tp, _BinaryOperation1, _BinaryOperation2);

template <class _Tag, class _ExecutionPolicy, class _ForwardIterator, class _Tp, class _BinaryOperation,
          class _UnaryOperation>
_Tp
__pattern_transform_reduce(_Tag, _ExecutionPolicy&&, _ForwardIterator, _ForwardIterator, _Tp, _BinaryOperation,
                           _UnaryOperation) noexcept;

template <class _IsVector, class _ExecutionPolicy, class _RandomAccessIterator, class _Tp, class _BinaryOperation,
          class _UnaryOperation>
_Tp
__pattern_transform_reduce(__parallel_tag<_IsVector>, _ExecutionPolicy&&, _RandomAccessIterator, _RandomAccessIterator,
                           _Tp, _BinaryOperation, _UnaryOperation);

//------------------------------------------------------------------------
// transform_exclusive_scan
//
// walk3 evaluates f(x,y,z) for (x,y,z) drawn from [first1,last1), [first2,...), [first3,...)
//------------------------------------------------------------------------

template <class _ForwardIterator, class _OutputIterator, class _UnaryOperation, class _Tp, class _BinaryOperation>
::std::pair<_OutputIterator, _Tp> __brick_transform_scan(_ForwardIterator, _ForwardIterator, _OutputIterator,
                                                         _UnaryOperation, _Tp, _BinaryOperation,
                                                         /*Inclusive*/ ::std::false_type) noexcept;

template <class _RandomAccessIterator, class _OutputIterator, class _UnaryOperation, class _Tp, class _BinaryOperation>
::std::pair<_OutputIterator, _Tp> __brick_transform_scan(_RandomAccessIterator, _RandomAccessIterator, _OutputIterator,
                                                         _UnaryOperation, _Tp, _BinaryOperation,
                                                         /*Inclusive*/ ::std::true_type) noexcept;

template <class _Tag, class _ExecutionPolicy, class _ForwardIterator, class _OutputIterator, class _UnaryOperation,
          class _Tp, class _BinaryOperation, class _Inclusive>
_OutputIterator
__pattern_transform_scan(_Tag, _ExecutionPolicy&&, _ForwardIterator, _ForwardIterator, _OutputIterator, _UnaryOperation,
                         _Tp, _BinaryOperation, _Inclusive) noexcept;

template <class _IsVector, class _ExecutionPolicy, class _RandomAccessIterator, class _OutputIterator,
          class _UnaryOperation, class _Tp, class _BinaryOperation, class _Inclusive>
::std::enable_if_t<!::std::is_floating_point_v<_Tp>, _OutputIterator>
__pattern_transform_scan(__parallel_tag<_IsVector>, _ExecutionPolicy&&, _RandomAccessIterator, _RandomAccessIterator,
                         _OutputIterator, _UnaryOperation, _Tp, _BinaryOperation, _Inclusive);

template <class _IsVector, class _ExecutionPolicy, class _RandomAccessIterator, class _OutputIterator,
          class _UnaryOperation, class _Tp, class _BinaryOperation, class _Inclusive>
::std::enable_if_t<::std::is_floating_point_v<_Tp>, _OutputIterator>
__pattern_transform_scan(__parallel_tag<_IsVector>, _ExecutionPolicy&&, _RandomAccessIterator, _RandomAccessIterator,
                         _OutputIterator, _UnaryOperation, _Tp, _BinaryOperation, _Inclusive);

// transform_scan without initial element
template <class _Tag, class _ExecutionPolicy, class _ForwardIterator, class _OutputIterator, class _UnaryOperation,
          class _BinaryOperation, class _Inclusive>
_OutputIterator
__pattern_transform_scan(_Tag, _ExecutionPolicy&& __exec, _ForwardIterator, _ForwardIterator, _OutputIterator,
                         _UnaryOperation, _BinaryOperation, _Inclusive);

//------------------------------------------------------------------------
// adjacent_difference
//------------------------------------------------------------------------

template <class _ForwardIterator, class _OutputIterator, class _BinaryOperation>
_OutputIterator __brick_adjacent_difference(_ForwardIterator, _ForwardIterator, _OutputIterator, _BinaryOperation,
                                            /*is_vector*/ ::std::false_type) noexcept;

template <class _RandomAccessIterator, class _OutputIterator, class _BinaryOperation>
_OutputIterator __brick_adjacent_difference(_RandomAccessIterator, _RandomAccessIterator, _OutputIterator,
                                            _BinaryOperation,
                                            /*is_vector*/ ::std::true_type) noexcept;

template <class _Tag, class _ExecutionPolicy, class _ForwardIterator, class _OutputIterator, class _BinaryOperation>
_OutputIterator
__pattern_adjacent_difference(_Tag, _ExecutionPolicy&&, _ForwardIterator, _ForwardIterator, _OutputIterator,
                              _BinaryOperation) noexcept;

template <class _IsVector, class _ExecutionPolicy, class _RandomAccessIterator1, class _RandomAccessIterator2,
          class _BinaryOperation>
_RandomAccessIterator2
__pattern_adjacent_difference(__parallel_tag<_IsVector>, _ExecutionPolicy&&, _RandomAccessIterator1,
                              _RandomAccessIterator1, _RandomAccessIterator2, _BinaryOperation);

} // namespace __internal
} // namespace dpl
} // namespace oneapi
#endif // _ONEDPL_NUMERIC_FWD_H
