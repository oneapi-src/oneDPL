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

#ifndef _PSTL_GLUE_ALGORITHM_RANGES_IMPL_H
#define _PSTL_GLUE_ALGORITHM_RANGES_IMPL_H

#include "execution_defs.h"
#include "glue_algorithm_defs.h"

#if _PSTL_HETERO_BACKEND
#    include "hetero/algorithm_ranges_impl_hetero.h"
#    include "hetero/algorithm_impl_hetero.h" //TODO: for __brick_copy
#endif

namespace dpstd
{

namespace ranges
{

// [alg.foreach]

template <typename _ExecutionPolicy, typename _Range, typename _Function>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, void>
for_each(_ExecutionPolicy&& __exec, _Range&& __rng, _Function __f)
{
    oneapi::dpl::__internal::__ranges::__pattern_walk1(std::forward<_ExecutionPolicy>(__exec),
                                                       std::forward<_Range>(__rng), __f);
}

// [alg.copy]

template <typename _ExecutionPolicy, typename _Range1, typename _Range2>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, void>
copy(_ExecutionPolicy&& __exec, _Range1&& __rng, _Range2&& __result)
{
    oneapi::dpl::__internal::__ranges::__pattern_walk2(std::forward<_ExecutionPolicy>(__exec),
                                                       std::forward<_Range1>(__rng), std::forward<_Range2>(__result),
                                                       oneapi::dpl::__internal::__brick_copy<_ExecutionPolicy>{});
}

// [alg.transform]

template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _UnaryOperation>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, void>
transform(_ExecutionPolicy&& __exec, _Range1&& __rng, _Range2&& __result, _UnaryOperation __op)
{
    oneapi::dpl::__internal::__ranges::__pattern_walk2(std::forward<_ExecutionPolicy>(__exec),
                                                       std::forward<_Range1>(__rng), std::forward<_Range2>(__result),
                                                       [__op](auto x, auto& z) mutable { z = __op(x); });
}

template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _Range3, typename _BinaryOperation>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, void>
transform(_ExecutionPolicy&& __exec, _Range1&& __rng1, _Range2&& __rng2, _Range3&& __result, _BinaryOperation __op)
{
    oneapi::dpl::__internal::__ranges::__pattern_walk3(
        std::forward<_ExecutionPolicy>(__exec), std::forward<_Range1>(__rng1), std::forward<_Range2>(__rng2),
        std::forward<_Range3>(__result), [__op](auto x, auto y, auto& z) mutable { z = __op(x, y); });
}

} // namespace ranges
} // namespace dpstd

#endif /* _PSTL_GLUE_ALGORITHM_RANGES_IMPL_H */
