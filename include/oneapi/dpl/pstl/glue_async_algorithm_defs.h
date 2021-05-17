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

#ifndef _ONEDPL_GLUE_ALGORITHM_DEFS_H
#define _ONEDPL_GLUE_ALGORITHM_DEFS_H

#include <functional>
#include <iterator>

#include "execution_defs.h"

namespace oneapi
{
namespace dpl
{
namespace experimental
{

// [alg.foreach]

namespace ranges
{

template <typename _ExecutionPolicy, typename _Range, typename _Function,
          typename = oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, void>, typename... _Events>
auto for_each_async(_ExecutionPolicy&& __exec, _Range&& __rng, _Function __f, _Events&&... __dependencies);

} //namespace ranges

template <typename _ExecutionPolicy, typename _ForwardIterator, typename _Function,
          typename = oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, void>, typename... _Events>
auto for_each_async(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last, _Function __f,
                    _Events&&... __dependencies);

} // namespace experimental
} // namespace dpl
} // namespace oneapi
#endif /* _ONEDPL_GLUE_ALGORITHM_DEFS_H */
