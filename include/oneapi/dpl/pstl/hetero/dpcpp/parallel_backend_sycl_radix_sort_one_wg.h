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

template <int __block_size, int __radix, bool __is_asc, typename ScanSum, typename Src, typename Dst,
          typename IdxDst, typename ScanBins>
void
sort_iter(sycl::nd_item<1> it, Src& src, Dst& res, ::std::size_t N, ScanSum& scan_sum,
          IdxDst& res_idx, ScanBins& scan_bins_lacc, int iter)
{
    auto wg = it.get_group();
    auto wi = wg.get_local_linear_id();
    assert( wi == it.get_global_linear_id() );

    constexpr int bin_count = 1 << __radix;
    int bins[bin_count] = {0};
    for (auto i = 0; i < __block_size; ++i)
    {
        const auto idx = wi * __block_size + i;
        if (idx < N)
        {
            // (src[idx] & mask) >> mask_shift;
            const int bin = __get_bucket<(1 << __radix) - 1, __is_asc>(__to_ordered(src[idx]), iter * __radix);

            //trivial exclusive scan within the WI block: for each element store the previous bin counter
            res_idx[i] = bins[bin];

            //and then increment the counter
            ++bins[bin];
        }
    }

    int init = 0;
    for (auto i = 0; i < bin_count; ++i)
    {
        const int a = bins[i];
        scan_sum[i] = sycl::exclusive_scan_over_group(wg, a, sycl::plus<>());
        if (wi == wg.get_local_linear_range() - 1)
        {
            // the last WI can iteratively do exclusive scan for bins
            scan_bins_lacc[i] = init;
            init += scan_sum[i] + bins[i];
        }
    }
    it.barrier(sycl::access::fence_space::local_space);

    for (auto i = 0; i < __block_size; ++i)
    {
        const auto idx = wi * __block_size + i;
        if (idx < N)
        {
            const auto val = src[idx];
            //(val & mask) >> mask_shift;
            const int bin = __get_bucket<(1 << __radix) - 1, __is_asc>(__to_ordered(val), iter * __radix);

            auto idx_d = scan_bins_lacc[bin] + scan_sum[bin] + res_idx[i];

            res[idx_d] = val;
        }
   }
   it.barrier(sycl::access::fence_space::local_space);
}

template<typename _KernelName, int __block_size = 16/*from 8 to 64*/, ::std::uint32_t __radix = 4, bool __is_asc = true,
         typename _RangeIn, typename _RangeOut>
auto
__group_radix_sort(sycl::queue q, _RangeIn&& src, _RangeOut&& res, int max_wg_size)
{
    const ::std::size_t __data_size = src.size();
    const auto wgSize = (__data_size - 1) / __block_size + 1;
    assert(wgSize <= max_wg_size);

    using _T = oneapi::dpl::__internal::__value_t<_RangeIn>;

    constexpr int bin_count = 1 << __radix;
    constexpr int iter_count = (sizeof(_T) * std::numeric_limits<unsigned char>::digits) / __radix;
    assert("Number of iterations must be even" && iter_count % 2 == 0);

    auto context = q.get_context();
    sycl::kernel_id kernelId1 = sycl::get_kernel_id<_KernelName>();
    auto bundle = sycl::get_kernel_bundle<sycl::bundle_state::executable>(context, {kernelId1});

    auto e = q.submit([&](sycl::handler& cgh) {
        oneapi::dpl::__ranges::__require_access(cgh, src, res);
        auto scan_bins_lacc = sycl::accessor<int, 1, access_mode::read_write, sycl::target::local>(bin_count, cgh);
        //auto res = sycl::accessor<int, 1, access_mode::read_write, sycl::target::local>(N, cgh);

        cgh.use_kernel_bundle(bundle);
        cgh.parallel_for<_KernelName>(sycl::nd_range{sycl::range{wgSize}, sycl::range{wgSize}},
            [=](sycl::nd_item<1> it) { // kernel code
                assert( it.get_local_range(0)==wgSize );
                int res_idx[__block_size];
                int scan_sum[bin_count];
                for (auto iter = 0; iter < iter_count; iter += 2)
                {
                    sort_iter<__block_size, __radix, __is_asc>(it, src, res, __data_size, scan_sum, res_idx,
                                                               scan_bins_lacc, iter);
                    sort_iter<__block_size, __radix, __is_asc>(it, res, src, __data_size, scan_sum, res_idx,
                                                               scan_bins_lacc, iter + 1);
                }
            });
    });

    return e;
}


//} // namespace __par_backend_hetero
//} // namespace dpl
//} // namespace oneapi

#endif /* _ONEDPL_parallel_backend_sycl_radix_sort_one_wg_H */
