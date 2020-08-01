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

#ifndef _PSTL_algorithm_ranges_impl_hetero_H
#define _PSTL_algorithm_ranges_impl_hetero_H

#include "../algorithm_fwd.h"
#include "../parallel_backend.h"

#if _PSTL_BACKEND_SYCL
#    include "dpcpp/utils_ranges_sycl.h"
#    include "dpcpp/unseq_backend_sycl.h"
#endif

namespace oneapi
{
namespace dpl
{
namespace __internal
{
namespace __ranges
{

//------------------------------------------------------------------------
// walk1
//------------------------------------------------------------------------

template <typename _ExecutionPolicy, typename _Range, typename _Function>
oneapi::dpl::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy, void>
__pattern_walk1(_ExecutionPolicy&& __exec, _Range&& __rng, _Function __f)
{
    if (!__rng.empty())
        oneapi::dpl::__par_backend_hetero::__ranges::__parallel_for(
            std::forward<_ExecutionPolicy>(__exec), unseq_backend::walk_n<_ExecutionPolicy, _Function>{__f},
            __rng.size(), std::forward<_Range>(__rng));
}

//------------------------------------------------------------------------
// walk2
//------------------------------------------------------------------------

template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _Function>
oneapi::dpl::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy, void>
__pattern_walk2(_ExecutionPolicy&& __exec, _Range1&& __rng1, _Range2&& __rng2, _Function __f)
{
    if (!__rng1.empty() && !__rng2.empty())
        oneapi::dpl::__par_backend_hetero::__ranges::__parallel_for(
            std::forward<_ExecutionPolicy>(__exec), unseq_backend::walk_n<_ExecutionPolicy, _Function>{__f},
            __rng1.size(), std::forward<_Range1>(__rng1), std::forward<_Range2>(__rng2));
}

//------------------------------------------------------------------------
// walk3
//------------------------------------------------------------------------

template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _Range3, typename _Function>
oneapi::dpl::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy, void>
__pattern_walk3(_ExecutionPolicy&& __exec, _Range1&& __rng1, _Range2&& __rng2, _Range3&& __rng3, _Function __f)
{
    if (!__rng1.empty() && !__rng2.empty() && !__rng3.empty())
        oneapi::dpl::__par_backend_hetero::__ranges::__parallel_for(
            std::forward<_ExecutionPolicy>(__exec), unseq_backend::walk_n<_ExecutionPolicy, _Function>{__f},
            __rng1.size(), std::forward<_Range1>(__rng1), std::forward<_Range2>(__rng2), std::forward<_Range3>(__rng3));
}

} // namespace __ranges
} // namespace __internal
} // namespace dpl
} // namespace oneapi

#endif /* _PSTL_algorithm_ranges_impl_hetero_H */
