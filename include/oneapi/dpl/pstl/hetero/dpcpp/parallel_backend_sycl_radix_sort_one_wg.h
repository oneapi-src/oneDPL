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


template <int ITEMS_PER_THREAD, typename KeyT, typename _Wi, typename _Src, typename _Keys>
void
__block_load(const _Wi __wi, const _Src& __src, _Keys& __keys, const uint32_t __n)
{
    constexpr KeyT __default_key = 0xffffffff;

    #pragma unroll
    for (auto i = 0; i < ITEMS_PER_THREAD; i++)
    {
        const uint32_t __offset = __wi*ITEMS_PER_THREAD + i;
        //boundary check is slow but nessecary
        if (__offset < __n)
            __keys[i] = __src[__offset];
        else
            __keys[i] = __default_key;
    }
}

template <int ITEMS_PER_THREAD, typename _Item, typename _Wi, typename _Lacc, typename _Keys, typename _Ranks>
void
__to_blocked(_Item __it, const _Wi __wi, _Lacc& __exchange_lacc, _Keys& __keys, const _Ranks& __ranks)
{
    for (int i = 0; i < ITEMS_PER_THREAD; i++)
        __exchange_lacc[__ranks[i]] = __keys[i];

    __dpl_sycl::__group_barrier(__it);

    for (int i = 0; i<ITEMS_PER_THREAD; i++)
        __keys[i] = __exchange_lacc[__wi*ITEMS_PER_THREAD + i];
}

template<typename _KernelName, int BLOCK_THREADS = 256/*work group size*/, int __block_size = 16/*from 8 to 64*/,
         ::std::uint32_t __radix = 4, bool __is_asc = true, typename _RangeIn>
auto __subgroup_radix_sort(sycl::queue __q, _RangeIn&& __src)
{
   constexpr auto req_sub_group_size = (__block_size < 4 ? 32 : 16);
   constexpr unsigned int RADIX_BITS = __radix;//radix 6bits need BLOCK_THREADS*BIN_COUNT to be at most 128*16
   constexpr unsigned int BIN_COUNT = 1 << RADIX_BITS;
   constexpr unsigned int mask = BIN_COUNT - 1;

   constexpr int ITEMS_PER_THREAD = __block_size;

   size_t N = __src.size();
   assert(N <= __block_size*BLOCK_THREADS);

    auto __bundle = sycl::get_kernel_bundle<sycl::bundle_state::executable>(__q.get_context(),
                                                                            {sycl::get_kernel_id<_KernelName>()});

   sycl::nd_range myRange {sycl::range{BLOCK_THREADS}, sycl::range{BLOCK_THREADS}};

   auto __event = __q.submit([&](sycl::handler& cgh) {
       oneapi::dpl::__ranges::__require_access(cgh, __src);
       uint32_t slm_size = std::max<uint32_t>(ITEMS_PER_THREAD*BLOCK_THREADS, BIN_COUNT * BLOCK_THREADS+16);
       auto lacc = sycl::local_accessor<uint32_t, 1>(slm_size, cgh);
       auto exchange_lacc = lacc; //exchange key, size is ITEMS_PER_THREAD*BLOCK_THREADS KeyT
       auto counter_lacc = lacc; //counter, could be private but use slm here, size is BLOCK_THREADS * BIN_COUNT

       cgh.use_kernel_bundle(__bundle);
       cgh.parallel_for<_KernelName>(myRange, ([=](sycl::nd_item<1> it) [[intel::reqd_sub_group_size(req_sub_group_size)]] {

           using KeyT = uint32_t;
           KeyT keys[ITEMS_PER_THREAD];
           auto wi_x = it.get_local_linear_id();
           uint32_t begin_bit = 0;
           constexpr uint32_t end_bit = sizeof(KeyT) * 8; 

           __block_load<ITEMS_PER_THREAD, KeyT>(wi_x, __src, keys, N);

           __dpl_sycl::__group_barrier(it);
           while (true)
           {
               int ranks[ITEMS_PER_THREAD];
               {
                   uint32_t thread_prefixes[ITEMS_PER_THREAD];
                   uint32_t* digit_counters[ITEMS_PER_THREAD];
                   //ResetCounters();
                   auto pcounter = counter_lacc.get_pointer()+wi_x;
                   #pragma unroll
                   for (int LANE = 0; LANE < BIN_COUNT; LANE++)
                       pcounter[LANE*BLOCK_THREADS] = 0;

                   #pragma unroll
                   for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
                   {
                       const int digit =  __get_bucket<mask>(__order_preserving_cast<__is_asc>(keys[ITEM]), begin_bit);

                       digit_counters[ITEM] = &pcounter[digit*BLOCK_THREADS];
                       thread_prefixes[ITEM] = *digit_counters[ITEM];
                       *digit_counters[ITEM] = thread_prefixes[ITEM] + 1;
                   }
                   __dpl_sycl::__group_barrier(it);
                   // Scan shared memory counters
                   {
                       //access pattern might be further optimized

                       //scan contiguous numbers
                       uint32_t bin_sum[BIN_COUNT];
                       bin_sum[0] = counter_lacc[wi_x * BIN_COUNT];
                       for (int i = 1; i < BIN_COUNT; i++)
                           bin_sum[i] = bin_sum[i-1] + counter_lacc[wi_x * BIN_COUNT + i];

                       __dpl_sycl::__group_barrier(it);
                       //exclusive scan local sum
                       uint32_t sum_scan = __dpl_sycl::__exclusive_scan_over_group(it.get_group(), bin_sum[BIN_COUNT-1], sycl::plus<>());
                       //add to local sum, generate exclusive scan result
                       for (int i = 0; i < BIN_COUNT; i++)
                           counter_lacc[wi_x * BIN_COUNT + i + 1] = sum_scan + bin_sum[i];

                       if (wi_x == 0)
                           counter_lacc[0] = 0;
                       __dpl_sycl::__group_barrier(it);
                   }

                   // Extract the local ranks of each key
                   #pragma unroll
                   for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
                   {
                       // Add in thread block exclusive prefix
                       ranks[ITEM] = thread_prefixes[ITEM] + *digit_counters[ITEM];
                   }
               }

               begin_bit += RADIX_BITS;

               __dpl_sycl::__group_barrier(it);
               if (begin_bit >= end_bit)
               {
                   // end of iteration, write out result
                   for (int i = 0; i<ITEMS_PER_THREAD; i++)
                   {
                       //boundary check is slow but nessecary
                       if (ranks[i] < N)
                           __src[ranks[i]] = keys[i];
                   }
                   return;
               }
               __to_blocked<ITEMS_PER_THREAD>(it, wi_x, exchange_lacc, keys, ranks);
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
