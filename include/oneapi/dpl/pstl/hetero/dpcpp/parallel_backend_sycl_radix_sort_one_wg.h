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

#ifndef _ONEDPL_parallel_backend_sycl_radix_sort_one_wg_H
#define _ONEDPL_parallel_backend_sycl_radix_sort_one_wg_H

//#include <limits>

//#include "sycl_defs.h"
//#include "parallel_backend_sycl_utils.h"
//#include "execution_sycl_defs.h"

//namespace oneapi
//{
//namespace dpl
//{
//namespace __par_backend_hetero
//{

template <int __block_size, int __radix, bool __is_asc, typename _Group, typename _Src, typename _Dst, typename _BinPos>
void
__sort_iter(_Group __wg, _Src& __src, _Dst& __dst, ::std::size_t __data_size, _BinPos& __bin_position,
            int __radix_offset)
{
    constexpr int __bin_count = 1 << __radix;
    int __histogram[__bin_count] = {0};
    int __wi_block_pos[__bin_count];
    int __element_pos[__block_size];

    auto __wi = __wg.get_local_linear_id();

    for (auto __i = 0; __i < __block_size; ++__i)
    {
        const auto __idx = __wi * __block_size + __i;
        if (__idx < __data_size)
        {
            const int __bin =
                __get_bucket<(1 << __radix) - 1>(__order_preserving_cast<__is_asc>(__src[__idx]), __radix_offset);
            // trivial exclusive scan within the WI block: for each element store the previous counter value
            __element_pos[__i] = __histogram[__bin];
            // and then increment the counter
            ++__histogram[__bin];
        }
    }

    int __bin_pos = 0; // only matters for the last WI in the group
    for (auto __i = 0; __i < __bin_count; ++__i)
    {
        __wi_block_pos[__i] = sycl::exclusive_scan_over_group(__wg, __histogram[__i], sycl::plus<>());
        if (__wi == __wg.get_local_linear_range() - 1)
        {
            // the last WI can iteratively do exclusive scan for bins
            __bin_position[__i] = __bin_pos;
            __bin_pos += __wi_block_pos[__i] + __histogram[__i]; // the reduction over all private histograms
        }
    }
    sycl::group_barrier(__wg);

    for (auto __i = 0; __i < __block_size; ++__i)
    {
        const auto __idx = __wi * __block_size + __i;
        if (__idx < __data_size)
        {
            const auto __val = __src[__idx];
            const int __bin =
                __get_bucket<(1 << __radix) - 1>(__order_preserving_cast<__is_asc>(__val), __radix_offset);
            const int __new_idx = __bin_position[__bin] + __wi_block_pos[__bin] + __element_pos[__i];
            __dst[__new_idx] = __val;
        }
    }
    sycl::group_barrier(__wg);
}

template<typename _KernelName, int __block_size = 16/*from 8 to 64*/, ::std::uint32_t __radix = 4, bool __is_asc = true,
         typename _RangeIn, typename _RangeOut>
auto
__group_radix_sort(sycl::queue __q, _RangeIn&& __src, _RangeOut&& __res, int __max_wg_size)
{
    using _T = oneapi::dpl::__internal::__value_t<_RangeIn>;
    constexpr int __bin_count = 1 << __radix;
    constexpr int __iter_count = (sizeof(_T) * std::numeric_limits<unsigned char>::digits) / __radix;
    static_assert(__iter_count % 2 == 0, "Number of radix iterations must be even");

    const ::std::size_t __data_size = __src.size();
    const ::std::size_t __wg_size = (__data_size - 1) / __block_size + 1;
    assert(__wg_size <= __max_wg_size);

    auto __bundle = sycl::get_kernel_bundle<sycl::bundle_state::executable>(__q.get_context(),
                                                                            {sycl::get_kernel_id<_KernelName>()});

    auto __event = __q.submit([&](sycl::handler& __cgh){
        oneapi::dpl::__ranges::__require_access(__cgh, __src, __res);
        auto __bin_position = 
            sycl::accessor<int, 1, sycl::access_mode::read_write, sycl::target::local>(__bin_count, __cgh);

        __cgh.use_kernel_bundle(__bundle);
        __cgh.parallel_for<_KernelName>(sycl::nd_range{sycl::range{__wg_size}, sycl::range{__wg_size}},
            [=](sycl::nd_item<1> __it)
            { // kernel code
                auto __wg = __it.get_group();
                assert( __it.get_local_range(0) == __wg_size );
                assert( __wg.get_local_linear_id() == __it.get_global_linear_id() );

                for (auto __i = 0; __i < __iter_count; __i += 2)
                {
                    __sort_iter<__block_size, __radix, __is_asc>(__wg, __src, __res, __data_size,
                                                                 __bin_position, __i * __radix);
                    __sort_iter<__block_size, __radix, __is_asc>(__wg, __res, __src, __data_size,
                                                                 __bin_position, (__i + 1) * __radix);
                }
            });
    });

    return __event;
}


//} // namespace __par_backend_hetero
//} // namespace dpl
//} // namespace oneapi

#endif /* _ONEDPL_parallel_backend_sycl_radix_sort_one_wg_H */
