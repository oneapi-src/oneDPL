// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2017-2020 Intel Corporation
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

#ifndef _PSTL_MEMORY_IMPL_H
#define _PSTL_MEMORY_IMPL_H

#include <iterator>

#include "memory_fwd.h"
#include "unseq_backend_simd.h"

namespace pstl
{
namespace __internal
{

//------------------------------------------------------------------------
// uninitialized_move
//------------------------------------------------------------------------

template <typename _ForwardIterator, typename _OutputIterator>
_OutputIterator
__brick_uninitialized_move(_ForwardIterator __first, _ForwardIterator __last, _OutputIterator __result,
                           /*vector=*/std::false_type) noexcept
{
    using _ValueType = typename std::iterator_traits<_OutputIterator>::value_type;
    for (; __first != __last; ++__first, ++__result)
    {
        ::new (std::addressof(*__result)) _ValueType(std::move(*__first));
    }
    return __result;
}

template <typename _ForwardIterator, typename _OutputIterator>
_OutputIterator
__brick_uninitialized_move(_ForwardIterator __first, _ForwardIterator __last, _OutputIterator __result,
                           /*vector=*/std::true_type) noexcept
{
    using __ValueType = typename std::iterator_traits<_OutputIterator>::value_type;
    using _ReferenceType1 = typename std::iterator_traits<_ForwardIterator>::reference;
    using _ReferenceType2 = typename std::iterator_traits<_OutputIterator>::reference;

    return __unseq_backend::__simd_walk_2(
        __first, __last - __first, __result,
        [](_ReferenceType1 __x, _ReferenceType2 __y) { ::new (std::addressof(__y)) __ValueType(std::move(__x)); });
}

template <typename _Iterator>
void
__brick_destroy(_Iterator __first, _Iterator __last, /*vector*/ std::false_type) noexcept
{
    using _ValueType = typename std::iterator_traits<_Iterator>::value_type;

    for (; __first != __last; ++__first)
        __first->~_ValueType();
}

template <typename _Iterator>
void
__brick_destroy(_Iterator __first, _Iterator __last, /*vector*/ std::true_type) noexcept
{
    using _ValueType = typename std::iterator_traits<_Iterator>::value_type;
    using _ReferenceType = typename std::iterator_traits<_Iterator>::reference;

    __unseq_backend::__simd_walk_1(__first, __last - __first, [](_ReferenceType __x) { __x.~_ValueType(); });
}

//------------------------------------------------------------------------
// uninitialized copy
//------------------------------------------------------------------------

template <typename _ForwardIterator, typename _OutputIterator>
_OutputIterator
__brick_uninitialized_copy(_ForwardIterator __first, _ForwardIterator __last, _OutputIterator __result,
                           /*vector=*/std::false_type) noexcept
{
    using _ValueType = typename std::iterator_traits<_OutputIterator>::value_type;
    for (; __first != __last; ++__first, ++__result)
    {
        ::new (std::addressof(*__result)) _ValueType(*__first);
    }
    return __result;
}

template <typename _ForwardIterator, typename _OutputIterator>
_OutputIterator
__brick_uninitialized_copy(_ForwardIterator __first, _ForwardIterator __last, _OutputIterator __result,
                           /*vector=*/std::true_type) noexcept
{
    using __ValueType = typename std::iterator_traits<_OutputIterator>::value_type;
    using _ReferenceType1 = typename std::iterator_traits<_ForwardIterator>::reference;
    using _ReferenceType2 = typename std::iterator_traits<_OutputIterator>::reference;

    return __unseq_backend::__simd_walk_2(
        __first, __last - __first, __result,
        [](_ReferenceType1 __x, _ReferenceType2 __y) { ::new (std::addressof(__y)) __ValueType(__x); });
}

template <typename _ExecutionPolicy>
struct __op_uninitialized_copy<_ExecutionPolicy>
{
    template <typename _SourceT, typename _TargetT>
    void
    operator()(_SourceT&& __source, _TargetT& __target)
    {
        using _TargetValueType = typename std::decay<_TargetT>::type;

        ::new (std::addressof(__target)) _TargetValueType(std::forward<_SourceT>(__source));
    }
};

//------------------------------------------------------------------------
// uninitialized move
//------------------------------------------------------------------------

template <typename _ExecutionPolicy>
struct __op_uninitialized_move<_ExecutionPolicy>
{
    template <typename _SourceT, typename _TargetT>
    void
    operator()(_SourceT&& __source, _TargetT& __target)
    {
        using _TargetValueType = typename std::decay<_TargetT>::type;

        ::new (std::addressof(__target)) _TargetValueType(std::move(__source));
    }
};

//------------------------------------------------------------------------
// uninitialized fill
//------------------------------------------------------------------------

template <typename _SourceT, typename _ExecutionPolicy>
struct __op_uninitialized_fill<_SourceT, _ExecutionPolicy>
{
    __ref_or_copy<_ExecutionPolicy, _SourceT> __source;

    template <typename _TargetT>
    void
    operator()(_TargetT& __target)
    {
        using _TargetValueType = typename std::decay<_TargetT>::type;

        ::new (std::addressof(__target)) _TargetValueType(__source);
    }
};

//------------------------------------------------------------------------
// destroy
//------------------------------------------------------------------------

template <typename _ExecutionPolicy>
struct __op_destroy<_ExecutionPolicy>
{
    template <typename _TargetT>
    void
    operator()(_TargetT& __target)
    {
        using _TargetValueType = typename std::decay<_TargetT>::type;
        __target.~_TargetValueType();
    }
};

//------------------------------------------------------------------------
// uninitialized default_construct
//------------------------------------------------------------------------

template <typename _ExecutionPolicy>
struct __op_uninitialized_default_construct<_ExecutionPolicy>
{
    template <typename _TargetT>
    void
    operator()(_TargetT& __target)
    {
        using _TargetValueType = typename std::decay<_TargetT>::type;

        ::new (std::addressof(__target)) _TargetValueType;
    }
};

//------------------------------------------------------------------------
// uninitialized value_construct
//------------------------------------------------------------------------

template <typename _ExecutionPolicy>
struct __op_uninitialized_value_construct<_ExecutionPolicy>
{
    template <typename _TargetT>
    void
    operator()(_TargetT& __target)
    {
        using _TargetValueType = typename std::decay<_TargetT>::type;

        ::new (std::addressof(__target)) _TargetValueType();
    }
};

} // namespace __internal
} // namespace pstl

#endif /* _PSTL_MEMORY_IMPL_H */
