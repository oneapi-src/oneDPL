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

#include <limits>

#include "sycl_defs.h"
#include "parallel_backend_sycl_utils.h"
#include "execution_sycl_defs.h"

namespace oneapi
{
namespace dpl
{
namespace __par_backend_hetero
{

template<int radix, bool __is_asc, typename _T>
int get_bin(_T val, int iter)
{
    return oneapi::dpl::__par_backend_hetero::__get_bucket_value<radix, __is_asc>(oneapi::dpl::__par_backend_hetero::__convert_to_ordered(val), iter);
}

template<int radix = 4, bool __is_asc = true, typename Bins, typename ScanSum, typename Src, typename Dst, typename IdxDst, typename ScanBins>
void sort_iter(Bins bins, ScanSum scan_sum, sycl::nd_item<1> it, Src& src, Dst& res, IdxDst& res_idx, ScanBins& scan_bins_lacc, int bin_count, int block_size, int wgSize, int N, int iter)
{
    auto wi = it.get_global_linear_id();
    auto wg_i = it.get_local_linear_id();
      
    auto wg = it.get_group();
    auto wg_id = wg.get_group_linear_id();

    for(auto i = 0; i < block_size; ++i)
    {
        const auto idx = wi*block_size + i;
        if(idx < N)
        {
          const int bin = get_bin<radix, __is_asc>(src[idx], iter);// (src[idx] & mask) >> mask_shift;

          //trivial serial scan, one sum
          res_idx[i] = bins[bin];

          //count
          ++bins[bin];

        }
    }

    for(auto i = 0; i < bin_count; ++i)
    {
      const int a = bins[i];
      scan_sum[i] = sycl::inclusive_scan_over_group(wg, a, sycl::plus<>());
    }
    if(wg_i == wgSize - 1)
    {
        //exclusive scan for bins
        int init = 0;
        for(auto i = 0; i < bin_count; ++i)
        {
            auto tmp = init;
            init += scan_sum[i];
            scan_bins_lacc[i] = tmp;
        }
    }
    it.barrier(sycl::access::fence_space::local_space);

    for(auto i = 0; i < block_size; ++i)
    {
      const auto idx = wi*block_size + i;
      if(idx < N)
      {
            const auto val = src[idx];
            const int bin = get_bin<radix, __is_asc>(val, iter);//(val & mask) >> mask_shift;
            auto idx_d = res_idx[i];
            idx_d += scan_bins_lacc[bin] + scan_sum[bin] - bins[bin];

            res[idx_d] = val;
        }
   }
   //sync
   it.barrier(sycl::access::fence_space::local_space);
}

template<typename KernelName, int block_size = 16/*from 8 to 64*/, bool __is_comp_asc = true, int radix = 4,
         typename _RangeIn, typename _RangeOut>
auto __group_radix_sort(sycl::queue q, _RangeIn&& src, _RangeOut&& res, int max_wg_size)
{
    const int N = src.size();
    const auto wgSize = (N - 1) / block_size + 1;
    if(wgSize > max_wg_size)
        return sycl::event{};

    using _T = oneapi::dpl::__internal::__value_t<_RangeIn>;

    constexpr int bin_count = 1 << radix;
    constexpr int iter_count = (sizeof(_T) * std::numeric_limits<unsigned char>::digits) / radix;

    auto n_wi = ((N -1) / wgSize + 1)*wgSize;
    auto wg_count = (n_wi / block_size) / wgSize;

    auto context = q.get_context();
    sycl::kernel_id kernelId1 = sycl::get_kernel_id<KernelName>();
    auto bundle = sycl::get_kernel_bundle<sycl::bundle_state::executable>(context, {kernelId1});

    assert(wg_count == 1);

    auto e = q.submit([&](sycl::handler& cgh) {
        oneapi::dpl::__ranges::__require_access(cgh, src, res);
        auto scan_bins_lacc = sycl::accessor<int, 1, access_mode::read_write, sycl::target::local>(bin_count, cgh);
        //auto res = sycl::accessor<int, 1, access_mode::read_write, sycl::target::local>(N, cgh);

        cgh.use_kernel_bundle(bundle);
        cgh.parallel_for<KernelName>(sycl::nd_range{sycl::range{n_wi / block_size}, sycl::range{wgSize}}, ([=](sycl::nd_item<1> it) {

            // kernel code
            int res_idx[block_size];
            for(auto iter = 0; iter < iter_count; ++iter)
            {
                int bins[bin_count] = {0};
                int scan_sum[bin_count] = {0};

                if(iter % 2 == 0)
                    sort_iter(bins, scan_sum, it, src, res, res_idx, scan_bins_lacc, bin_count, block_size, wgSize, N, iter);
                else
                    sort_iter(bins, scan_sum, it, res, src, res_idx, scan_bins_lacc, bin_count, block_size, wgSize, N, iter);
            }//for
        }
      ));
    });

  return e;
}


} // namespace __par_backend_hetero
} // namespace dpl
} // namespace oneapi

#endif /* _ONEDPL_parallel_backend_sycl_radix_sort_one_wg_H */
