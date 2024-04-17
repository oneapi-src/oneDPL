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

// This header guard is used to check inclusion of OpenMP backend.
// Changing this macro may result in broken tests.
#ifndef _ONEDPL_PARALLEL_BACKEND_OMP_H
#define _ONEDPL_PARALLEL_BACKEND_OMP_H

#include "execution_defs.h"

namespace oneapi
{
namespace dpl
{
namespace __backend
{

template <>
struct __backend_impl<oneapi::dpl::__internal::__omp_backend_tag>
{
    template <typename _ExecutionPolicy, typename _Tp>
    using __buffer = oneapi::dpl::__utils::__buffer_impl<std::decay_t<_ExecutionPolicy>, _Tp, std::allocator>;

    static void
    __cancel_execution()
    {
        // TODO: Figure out how to make cancellation work.
    }

    template <class _ExecutionPolicy, class _Index, class _Fp>
    static void
    __parallel_for(_ExecutionPolicy&&, _Index __first, _Index __last, _Fp __f);

    template <class _ExecutionPolicy, class _ForwardIterator, class _Fp>
    static void
    __parallel_for_each(_ExecutionPolicy&&, _ForwardIterator __first, _ForwardIterator __last, _Fp __f);

    template <class _ExecutionPolicy, typename _F1, typename _F2>
    static void
    __parallel_invoke(_ExecutionPolicy&&, _F1&& __f1, _F2&& __f2);

    template <class _ExecutionPolicy, typename _RandomAccessIterator1, typename _RandomAccessIterator2,
              typename _RandomAccessIterator3, typename _Compare, typename _LeafMerge>
    static void
    __parallel_merge(_ExecutionPolicy&& /*__exec*/, _RandomAccessIterator1 __xs, _RandomAccessIterator1 __xe,
                     _RandomAccessIterator2 __ys, _RandomAccessIterator2 __ye, _RandomAccessIterator3 __zs,
                     _Compare __comp, _LeafMerge __leaf_merge);

    template <class _ExecutionPolicy, class _RandomAccessIterator, class _Value, typename _RealBody,
              typename _Reduction>
    static _Value
    __parallel_reduce(_ExecutionPolicy&&, _RandomAccessIterator __first, _RandomAccessIterator __last,
                      _Value __identity, _RealBody __real_body, _Reduction __reduction);

    template <class _ExecutionPolicy, typename _Index, typename _Tp, typename _Rp, typename _Cp, typename _Sp,
              typename _Ap>
    static void
    __parallel_strict_scan(_ExecutionPolicy&& __exec, _Index __n, _Tp __initial, _Rp __reduce, _Cp __combine,
                           _Sp __scan, _Ap __apex);

    template <class _ExecutionPolicy, typename _RandomAccessIterator, typename _Compare, typename _LeafSort>
    static void
    __parallel_stable_sort(_ExecutionPolicy&& /*__exec*/, _RandomAccessIterator __xs, _RandomAccessIterator __xe,
                           _Compare __comp, _LeafSort __leaf_sort, std::size_t __nsort = 0);

    template <class _ExecutionPolicy, class _RandomAccessIterator, class _UnaryOp, class _Value, class _Combiner,
              class _Reduction>
    static _Value
    __parallel_transform_reduce(_ExecutionPolicy&&, _RandomAccessIterator __first, _RandomAccessIterator __last,
                                _UnaryOp __unary_op, _Value __init, _Combiner __combiner, _Reduction __reduction);

    template <class _ExecutionPolicy, class _Index, class _Up, class _Tp, class _Cp, class _Rp, class _Sp>
    static _Tp
    __parallel_transform_scan(_ExecutionPolicy&&, _Index __n, _Up /* __u */, _Tp __init, _Cp /* __combine */,
                              _Rp /* __brick_reduce */, _Sp __scan);
};

} // namespace __backend
} // namespace dpl
} // namespace oneapi

//------------------------------------------------------------------------
// parallel_invoke
//------------------------------------------------------------------------

#include "./omp/parallel_invoke.h"

//------------------------------------------------------------------------
// parallel_for
//------------------------------------------------------------------------

#include "./omp/parallel_for.h"

//------------------------------------------------------------------------
// parallel_for_each
//------------------------------------------------------------------------

#include "./omp/parallel_for_each.h"

//------------------------------------------------------------------------
// parallel_reduce
//------------------------------------------------------------------------

#include "./omp/parallel_reduce.h"
#include "./omp/parallel_transform_reduce.h"

//------------------------------------------------------------------------
// parallel_scan
//------------------------------------------------------------------------

#include "./omp/parallel_scan.h"
#include "./omp/parallel_transform_scan.h"

//------------------------------------------------------------------------
// parallel_stable_sort
//------------------------------------------------------------------------

#include "./omp/parallel_stable_partial_sort.h"
#include "./omp/parallel_stable_sort.h"

//------------------------------------------------------------------------
// parallel_merge
//------------------------------------------------------------------------
#include "./omp/parallel_merge.h"

#endif //_ONEDPL_PARALLEL_BACKEND_OMP_H
