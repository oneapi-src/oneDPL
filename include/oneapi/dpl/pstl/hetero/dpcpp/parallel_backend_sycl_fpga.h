// -*- C++ -*-
//===-- parallel_backend_sycl_fpga.h --------------------------------------===//
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

//!!! NOTE: This file should be included under the macro _ONEDPL_BACKEND_SYCL

// This header guard is used to check inclusion of DPC++ backend for the FPGA.
// Changing this macro may result in broken tests.
#ifndef _ONEDPL_PARALLEL_BACKEND_SYCL_FPGA_H
#define _ONEDPL_PARALLEL_BACKEND_SYCL_FPGA_H

#include <cassert>
#include <algorithm>
#include <type_traits>

#include "sycl_defs.h"
#include "parallel_backend_sycl_utils.h"
// workaround until we implement more performant optimization for patterns
#include "parallel_backend_sycl.h"
#include "parallel_backend_sycl_histogram.h"
#include "../../execution_impl.h"
#include "execution_sycl_defs.h"
#include "../../iterator_impl.h"
#include "sycl_iterator.h"

#include "sycl_traits.h" //SYCL traits specialization for some oneDPL types.

namespace oneapi
{
namespace dpl
{
namespace __par_backend_hetero
{
//------------------------------------------------------------------------
// parallel_for
//------------------------------------------------------------------------
//General version of parallel_for, one additional parameter - __count of iterations of loop __cgh.parallel_for,
//for some algorithms happens that size of processing range is n, but amount of iterations is n/2.

// Please see the comment for __parallel_for_submitter for optional kernel name explanation
template <typename _Name>
struct __parallel_for_fpga_submitter;

template <typename... _Name>
struct __parallel_for_fpga_submitter<__internal::__optional_kernel_name<_Name...>>
{
    template <typename _ExecutionPolicy, typename _Fp, typename _Index, typename... _Ranges>
    auto
    operator()(_ExecutionPolicy&& __exec, _Fp __brick, _Index __count, _Ranges&&... __rngs) const
    {
        auto __n = oneapi::dpl::__ranges::__get_first_range_size(__rngs...);
        assert(__n > 0);

        _PRINT_INFO_IN_DEBUG_MODE(__exec);
        auto __event = __exec.queue().submit([&__rngs..., &__brick, __count](sycl::handler& __cgh) {
            //get an access to data under SYCL buffer:
            oneapi::dpl::__ranges::__require_access(__cgh, __rngs...);

            __cgh.single_task<_Name...>([=]() {
#pragma unroll(::std::decay <_ExecutionPolicy>::type::unroll_factor)
                for (auto __idx = 0; __idx < __count; ++__idx)
                {
                    __brick(__idx, __rngs...);
                }
            });
        });
        return __future(__event);
    }
};

template <typename _ExecutionPolicy, typename _Fp, typename _Index, typename... _Ranges>
auto
__parallel_for(oneapi::dpl::__internal::__fpga_backend_tag, _ExecutionPolicy&& __exec, _Fp __brick, _Index __count,
               _Ranges&&... __rngs)
{
    using _CustomName = oneapi::dpl::__internal::__policy_kernel_name<_ExecutionPolicy>;
    using __parallel_for_name = __internal::__kernel_name_provider<_CustomName>;

    return __parallel_for_fpga_submitter<__parallel_for_name>()(std::forward<_ExecutionPolicy>(__exec), __brick,
                                                                __count, std::forward<_Ranges>(__rngs)...);
}

//------------------------------------------------------------------------
// parallel_histogram
//-----------------------------------------------------------------------

// TODO: check if it makes sense to move these wrappers out of backend to a common place
template <typename _ExecutionPolicy, typename _Event, typename _Range1, typename _Range2, typename _BinHashMgr>
auto
__parallel_histogram(oneapi::dpl::__internal::__fpga_backend_tag, _ExecutionPolicy&& __exec, const _Event& __init_event,
                     _Range1&& __input, _Range2&& __bins, const _BinHashMgr& __binhash_manager)
{
    static_assert(sizeof(oneapi::dpl::__internal::__value_t<_Range2>) <= sizeof(::std::uint32_t),
                  "histogram is not supported on FPGA devices with output types greater than 32 bits");

    // workaround until we implement more performant version for patterns
    return oneapi::dpl::__par_backend_hetero::__parallel_histogram(
        oneapi::dpl::__internal::__device_backend_tag{}, __exec, __init_event, ::std::forward<_Range1>(__input),
        ::std::forward<_Range2>(__bins), __binhash_manager);
}

} // namespace __par_backend_hetero
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_PARALLEL_BACKEND_SYCL_FPGA_H
