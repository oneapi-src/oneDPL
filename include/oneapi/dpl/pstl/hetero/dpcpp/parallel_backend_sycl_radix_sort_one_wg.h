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

//namespace oneapi
//{
//namespace dpl
//{
//namespace __par_backend_hetero
//{

template <int __block_size, typename KeyT, typename _Wi, typename _Src, typename _Keys>
void
__block_load(const _Wi __wi, const _Src& __src, _Keys& __keys, const uint32_t __n)
{
    constexpr KeyT __default_key = KeyT{};

    #pragma unroll
    for (auto i = 0; i < __block_size; i++)
    {
        const uint32_t __offset = __wi*__block_size + i;
        //boundary check is slow but nessecary
        if (__offset < __n)
            __keys[i] = __src[__offset];
        else
            __keys[i] = __default_key;
    }
}

template <int __block_size, typename _Item, typename _Wi, typename _Lacc, typename _Keys, typename _Ranks>
void
__to_blocked(_Item __it, const _Wi __wi, _Lacc& __exchange_lacc, _Keys& __keys, const _Ranks& __ranks)
{
    for (int i = 0; i < __block_size; i++)
        __exchange_lacc[__ranks[i]] = __keys[i];

    __dpl_sycl::__group_barrier(__it);

    for (int i = 0; i<__block_size; i++)
        __keys[i] = __exchange_lacc[__wi*__block_size + i];
}

template<typename _KernelName, int __wg_size = 256/*work group size*/, int __block_size = 16,
         ::std::uint32_t __radix = 4, bool __is_asc = true, typename _RangeIn,
         int req_sub_group_size = (__block_size < 4 ? 32 : 16)>
auto __subgroup_radix_sort(sycl::queue __q, _RangeIn&& __src)
{
    constexpr unsigned int __bin_count = 1 << __radix;

    size_t __n = __src.size();
    assert(__n <= __block_size*__wg_size);
  
# if _ONEDPL_KERNEL_BUNDLE_PRESENT
    auto __kernel_id = sycl::get_kernel_id<_KernelName>();
    auto __bundle = sycl::get_kernel_bundle<sycl::bundle_state::executable>(__q.get_context(), {__kernel_id});
# endif
  
    using KeyT = oneapi::dpl::__internal::__value_t<_RangeIn>;
    sycl::nd_range myRange {sycl::range{__wg_size}, sycl::range{__wg_size}};
    auto __event = __q.submit([&](sycl::handler& cgh) {
        oneapi::dpl::__ranges::__require_access(cgh, __src);
  
        auto exchange_lacc = sycl::local_accessor<KeyT, 1>(__block_size*__wg_size, cgh);//exchange key, size is __block_size*__wg_size KeyT
        auto counter_lacc = sycl::local_accessor<uint32_t, 1>(__wg_size * __bin_count, cgh);//counter, could be private but use slm here
  
# if _ONEDPL_KERNEL_BUNDLE_PRESENT
        cgh.use_kernel_bundle(__bundle);
# endif
        cgh.parallel_for<_KernelName>(
            myRange, ([=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(req_sub_group_size)]]
        {
  
            KeyT keys[__block_size];
            auto wi_x = it.get_local_linear_id();
            uint32_t begin_bit = 0;
            constexpr uint32_t end_bit = sizeof(KeyT) * 8; 
  
            __block_load<__block_size, KeyT>(wi_x, __src, keys, __n);
  
            __dpl_sycl::__group_barrier(it);
            while (true)
            {
                uint16_t ranks[__block_size];
                {
                    uint16_t thread_prefixes[__block_size];
                    uint32_t* digit_counters[__block_size];
                    //ResetCounters();
                    auto pcounter = counter_lacc.get_pointer()+wi_x;
                    #pragma unroll
                    for (int LANE = 0; LANE < __bin_count; LANE++)
                        pcounter[LANE*__wg_size] = 0;
  
                    #pragma unroll
                    for (uint16_t ITEM = 0; ITEM < __block_size; ++ITEM)
                    {
                        const int digit =
                            __get_bucket</*mask*/__bin_count - 1>(__order_preserving_cast<__is_asc>(keys[ITEM]), begin_bit);
  
                        digit_counters[ITEM] = &pcounter[digit*__wg_size];
                        thread_prefixes[ITEM] = *digit_counters[ITEM];
                        *digit_counters[ITEM] = thread_prefixes[ITEM] + 1;
                    }
                    __dpl_sycl::__group_barrier(it);
  
                    // Scan shared memory counters
                    {
                        //access pattern might be further optimized
  
                        //scan contiguous numbers
                        uint16_t bin_sum[__bin_count];
                        bin_sum[0] = counter_lacc[wi_x * __bin_count];
                        for (uint16_t i = 1; i < __bin_count; i++)
                            bin_sum[i] = bin_sum[i-1] + counter_lacc[wi_x * __bin_count + i];
  
                        __dpl_sycl::__group_barrier(it);
                        //exclusive scan local sum
                        uint16_t sum_scan = __dpl_sycl::__exclusive_scan_over_group(it.get_group(), bin_sum[__bin_count-1], sycl::plus<uint16_t>());
                        //add to local sum, generate exclusive scan result
                        for (uint16_t i = 0; i < __bin_count; i++)
                            counter_lacc[wi_x * __bin_count + i + 1] = sum_scan + bin_sum[i];
  
                        if (wi_x == 0)
                            counter_lacc[0] = 0;
                        __dpl_sycl::__group_barrier(it);
                    }
  
                    // Extract the local ranks of each key
                    #pragma unroll
                    for (uint16_t ITEM = 0; ITEM < __block_size; ++ITEM)
                    {
                        // Add in thread block exclusive prefix
                        ranks[ITEM] = thread_prefixes[ITEM] + *digit_counters[ITEM];
                    }
                }
  
                begin_bit += __radix;
  
                __dpl_sycl::__group_barrier(it);
                if (begin_bit >= end_bit)
                {
                    // end of iteration, write out result
                    for (uint16_t i = 0; i<__block_size; i++)
                    {
                        //boundary check is slow but nessecary
                        if (ranks[i] < N)
                            __src[ranks[i]] = keys[i];
                    }
                    return;
                }
                __to_blocked<__block_size>(it, wi_x, exchange_lacc, keys, ranks);
                __dpl_sycl::__group_barrier(it);
            }
        }));
     });
   return __event;
}


//} // namespace __par_backend_hetero
//} // namespace dpl
//} // namespace oneapi

#endif /* _ONEDPL_parallel_backend_sycl_radix_sort_one_wg_H */
