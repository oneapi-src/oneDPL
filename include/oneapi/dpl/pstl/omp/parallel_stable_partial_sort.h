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

#ifndef _ONEDPL_INTERNAL_OMP_PARALLEL_STABLE_PARTIAL_SORT_H
#define _ONEDPL_INTERNAL_OMP_PARALLEL_STABLE_PARTIAL_SORT_H

#include "util.h"

namespace oneapi
{
namespace dpl
{
namespace __omp_backend
{

template <typename _RandomAccessIterator, typename _Compare, typename _LeafSort>
void
__parallel_stable_partial_sort(_RandomAccessIterator __xs, _RandomAccessIterator __xe, _Compare __comp,
                               _LeafSort __leaf_sort, std::size_t /* __nsort */)
{
    // TODO: "Parallel partial sort needs to be implemented.");
    __leaf_sort(__xs, __xe, __comp);
}

} // namespace __omp_backend
} // namespace dpl
} // namespace oneapi
#endif // _ONEDPL_INTERNAL_OMP_PARALLEL_STABLE_PARTIAL_SORT_H
