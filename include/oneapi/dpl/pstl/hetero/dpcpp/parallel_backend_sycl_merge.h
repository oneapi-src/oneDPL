// -*- C++ -*-
//===-- parallel_backend_sycl_merge.h --------------------------------===//
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

#ifndef _ONEDPL_PARALLEL_BACKEND_SYCL_MERGE_H
#define _ONEDPL_PARALLEL_BACKEND_SYCL_MERGE_H

#include <limits>    // std::numeric_limits
#include <cassert>   // assert
#include <cstdint>   // std::uint8_t, ...
#include <utility>   // std::make_pair, std::forward
#include <algorithm> // std::min, std::lower_bound

#include "sycl_defs.h"
#include "parallel_backend_sycl_utils.h"

namespace oneapi
{
namespace dpl
{
namespace __par_backend_hetero
{

//Searching for an intersection of a merge matrix (n1, n2) diagonal with the Merge Path to define sub-ranges
//to serial merge. For example, a merge matrix for [0,1,1,2,3] and [0,0,2,3] is shown below:
//     0   1  1  2   3
//    ------------------
//   |--->
// 0 | 0 | 1  1  1   1
//   |   |
// 0 | 0 | 1  1  1   1
//   |   ---------->
// 2 | 0   0  0  0 | 1
//   |             ---->
// 3 | 0   0  0  0   0 |
template <typename _Rng1, typename _Rng2, typename _Index, typename _Index1, typename _Index2, typename _Compare>
auto
__find_start_point(const _Rng1& __rng1, const _Rng2& __rng2, _Index __i_elem, _Index1 __n1, _Index2 __n2,
                   _Compare __comp)
{
    _Index1 __start1 = 0;
    _Index2 __start2 = 0;
    if (__i_elem < __n2) //a condition to specify upper or lower part of the merge matrix to be processed
    {
        auto __q = __i_elem;                          //diagonal index
        auto __n_diag = std::min<_Index2>(__q, __n1); //diagonal size

        //searching for the first '1', a lower bound for a diagonal [0, 0,..., 0, 1, 1,.... 1, 1]
        oneapi::dpl::counting_iterator<_Index> __diag_it(0);
        auto __res =
            std::lower_bound(__diag_it, __diag_it + __n_diag, 1 /*value to find*/,
                             [&__rng2, &__rng1, __q, __comp](const auto& __i_diag, const auto& __value) mutable {
                                 auto __zero_or_one = __comp(__rng2[__q - __i_diag - 1], __rng1[__i_diag]);
                                 return __zero_or_one < __value;
                             });
        __start1 = *__res;
        __start2 = __q - *__res;
    }
    else
    {
        auto __q = __i_elem - __n2;                          //diagonal index
        auto __n_diag = std::min<_Index1>(__n1 - __q, __n2); //diagonal size

        //searching for the first '1', a lower bound for a diagonal [0, 0,..., 0, 1, 1,.... 1, 1]
        oneapi::dpl::counting_iterator<_Index> __diag_it(0);
        auto __res =
            std::lower_bound(__diag_it, __diag_it + __n_diag, 1 /*value to find*/,
                             [&__rng2, &__rng1, __n2, __q, __comp](const auto& __i_diag, const auto& __value) mutable {
                                 auto __zero_or_one = __comp(__rng2[__n2 - __i_diag - 1], __rng1[__q + __i_diag]);
                                 return __zero_or_one < __value;
                             });

        __start1 = __q + *__res;
        __start2 = __n2 - *__res;
    }
    return std::make_pair(__start1, __start2);
}

// Do serial merge of the data from rng1 (starting from start1) and rng2 (starting from start2) and writing
// to rng3 (starting from start3) in 'chunk' steps, but do not exceed the total size of the sequences (n1 and n2)
template <typename _Rng1, typename _Rng2, typename _Rng3, typename _Index1, typename _Index2, typename _Index3,
          typename _Size1, typename _Size2, typename _Compare>
void
__serial_merge(const _Rng1& __rng1, const _Rng2& __rng2, _Rng3& __rng3, _Index1 __start1, _Index2 __start2,
               _Index3 __start3, std::uint8_t __chunk, _Size1 __n1, _Size2 __n2, _Compare __comp)
{
    if (__start1 >= __n1)
    {
        //copying a residual of the second seq
        const auto __n = std::min<_Index2>(__n2 - __start2, __chunk);
        for (std::uint8_t __i = 0; __i < __n; ++__i)
            __rng3[__start3 + __i] = __rng2[__start2 + __i];
    }
    else if (__start2 >= __n2)
    {
        //copying a residual of the first seq
        const auto __n = std::min<_Index1>(__n1 - __start1, __chunk);
        for (std::uint8_t __i = 0; __i < __n; ++__i)
            __rng3[__start3 + __i] = __rng1[__start1 + __i];
    }
    else
    {
        std::uint8_t __n = __chunk;
        for (std::uint8_t __i = 0; __i < __n && __start1 < __n1 && __start2 < __n2; ++__i)
        {
            const auto& __val1 = __rng1[__start1];
            const auto& __val2 = __rng2[__start2];
            if (__comp(__val2, __val1))
            {
                __rng3[__start3 + __i] = __val2;
                if (++__start2 == __n2)
                {
                    //copying a residual of the first seq
                    for (++__i; __i < __n && __start1 < __n1; ++__i, ++__start1)
                        __rng3[__start3 + __i] = __rng1[__start1];
                }
            }
            else
            {
                __rng3[__start3 + __i] = __val1;
                if (++__start1 == __n1)
                {
                    //copying a residual of the second seq
                    for (++__i; __i < __n && __start2 < __n2; ++__i, ++__start2)
                        __rng3[__start3 + __i] = __rng2[__start2];
                }
            }
        }
    }
}

// Please see the comment for __parallel_for_submitter for optional kernel name explanation
template <typename _IdType, typename _Name>
struct __parallel_merge_submitter;

template <typename _IdType, typename... _Name>
struct __parallel_merge_submitter<_IdType, __internal::__optional_kernel_name<_Name...>>
{
    template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _Range3, typename _Compare>
    auto
    operator()(_ExecutionPolicy&& __exec, _Range1&& __rng1, _Range2&& __rng2, _Range3&& __rng3, _Compare __comp) const
    {
        using _Size = oneapi::dpl::__internal::__difference_t<_Range3>;

        auto __n1 = __rng1.size();
        auto __n2 = __rng2.size();

        assert(__n1 > 0 || __n2 > 0);

        _PRINT_INFO_IN_DEBUG_MODE(__exec);

        const std::uint8_t __chunk = __exec.queue().get_device().is_cpu() ? 128 : 4;
        const _Size __n = __n1 + __n2;
        const _Size __steps = oneapi::dpl::__internal::__dpl_ceiling_div(__n, __chunk);

        auto __event = __exec.queue().submit([&](sycl::handler& __cgh) {
            oneapi::dpl::__ranges::__require_access(__cgh, __rng1, __rng2, __rng3);
            __cgh.parallel_for<_Name...>(sycl::range</*dim=*/1>(__steps), [=](sycl::item</*dim=*/1> __item_id) {
                const _IdType __i_elem = __item_id.get_linear_id() * __chunk;
                const auto __start = __find_start_point(__rng1, __rng2, __i_elem, __n1, __n2, __comp);
                __serial_merge(__rng1, __rng2, __rng3, __start.first, __start.second, __i_elem, __chunk, __n1, __n2,
                               __comp);
            });
        });
        return __future(__event);
    }
};

template <typename... _Name>
class __merge_kernel_name;

template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _Range3, typename _Compare>
auto
__parallel_merge(oneapi::dpl::__internal::__device_backend_tag, _ExecutionPolicy&& __exec, _Range1&& __rng1,
                 _Range2&& __rng2, _Range3&& __rng3, _Compare __comp)
{
    using _CustomName = oneapi::dpl::__internal::__policy_kernel_name<_ExecutionPolicy>;

    const auto __n = __rng1.size() + __rng2.size();
    if (__n <= std::numeric_limits<std::uint32_t>::max())
    {
        using _wi_index_type = std::uint32_t;
        using _MergeKernel = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
            __merge_kernel_name<_CustomName, _wi_index_type>>;
        return __parallel_merge_submitter<_wi_index_type, _MergeKernel>()(
            std::forward<_ExecutionPolicy>(__exec), std::forward<_Range1>(__rng1), std::forward<_Range2>(__rng2),
            std::forward<_Range3>(__rng3), __comp);
    }
    else
    {
        using _wi_index_type = std::uint64_t;
        using _MergeKernel = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
            __merge_kernel_name<_CustomName, _wi_index_type>>;
        return __parallel_merge_submitter<_wi_index_type, _MergeKernel>()(
            std::forward<_ExecutionPolicy>(__exec), std::forward<_Range1>(__rng1), std::forward<_Range2>(__rng2),
            std::forward<_Range3>(__rng3), __comp);
    }
}

} // namespace __par_backend_hetero
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_PARALLEL_BACKEND_SYCL_MERGE_H
