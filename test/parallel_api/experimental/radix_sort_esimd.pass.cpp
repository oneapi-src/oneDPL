// -*- C++ -*-
//===-- rasix_sort_esimd.pass.cpp -----------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <ext/intel/esimd.hpp>

#include <sycl/sycl.hpp>

#include <iostream>

//#define REPRODUCE_ERROR_1 1
//#define REPRODUCE_ERROR_2 1

// rounded up result of (__number / __divisor)
template <typename _T1, typename _T2>
constexpr auto
__dpl_ceiling_div(_T1 __number, _T2 __divisor)
{
    return (__number - 1) / __divisor + 1;
}

// get bits value (bucket) in a certain radix position
template <::std::uint16_t __radix_mask, typename _T, int _N>
sycl::ext::intel::esimd::simd<::std::uint16_t, _N>
__get_bucket(sycl::ext::intel::esimd::simd<_T, _N> __value, ::std::uint32_t __radix_offset)
{
    return sycl::ext::intel::esimd::simd<::std::uint16_t, _N>(__value >> __radix_offset) & __radix_mask;
}

template <typename KeyT, typename InputT, uint32_t RADIX_BITS, uint32_t TG_COUNT, uint32_t THREAD_PER_TG, bool IsAscending>
void global_histogram(sycl::nd_item<1> idx, size_t __n, const InputT& input, uint32_t *p_global_offset, uint32_t *p_sync_buffer)
{
    using bin_t = uint16_t;
    using hist_t = uint32_t;
    using global_hist_t = uint32_t;

    using namespace sycl;
    using namespace __ESIMD_NS;
    using namespace __ESIMD_ENS;

    using device_addr_t = uint32_t;

    slm_init(16384);
    constexpr uint32_t BINCOUNT = 1 << RADIX_BITS;
    constexpr uint32_t NBITS =  sizeof(KeyT) * 8;
    constexpr uint32_t STAGES = __dpl_ceiling_div(NBITS, RADIX_BITS);
    constexpr uint32_t PROCESS_SIZE = 128;
    constexpr uint32_t addr_step = TG_COUNT * THREAD_PER_TG * PROCESS_SIZE;

    uint32_t local_tid = idx.get_local_linear_id();
    uint32_t tid = idx.get_global_linear_id();

    if ((tid - local_tid) * PROCESS_SIZE > __n) {
        //no work for this tg;
        return;
    }

    // cooperative fill 0
    {
        constexpr uint32_t BUFFER_SIZE = STAGES * BINCOUNT;
        constexpr uint32_t THREAD_SIZE = BUFFER_SIZE / THREAD_PER_TG;
        slm_block_store<global_hist_t, THREAD_SIZE>(local_tid*THREAD_SIZE*sizeof(global_hist_t), 0);
    }
    barrier();


    simd<KeyT, PROCESS_SIZE> keys;
    simd<bin_t, PROCESS_SIZE> bins;
    simd<global_hist_t, BINCOUNT * STAGES> state_hist_grf(0);
    constexpr bin_t MASK = BINCOUNT - 1;

    device_addr_t read_addr;
    for (read_addr = tid * PROCESS_SIZE; read_addr < __n; read_addr += addr_step) {
        if (read_addr+PROCESS_SIZE < __n) {
#ifdef REPRODUCE_ERROR_2
            keys.copy_from(input + read_addr);
#endif
        }
        else
        {
            simd<uint32_t, 16> lane_id(0, 1);
            #pragma unroll
            for (uint32_t s = 0; s<PROCESS_SIZE; s+=16) {
                simd_mask<16> m = (s+lane_id)<(__n-read_addr);

                sycl::ext::intel::esimd::simd offset((read_addr + s + lane_id)*sizeof(KeyT));
                simd<KeyT, 16> source = lsc_gather<KeyT, 1, lsc_data_size::default_size, cache_hint::cached, cache_hint::cached, 16>(input, offset, m);

                keys.template select<16, 1>(s) = merge(source, simd<KeyT, 16>(0), m);
            }
        }
        #pragma unroll
        for (uint32_t stage = 0; stage < STAGES; stage++)
        {
            bins = __get_bucket<MASK>(keys, stage * RADIX_BITS);

            #pragma unroll
            for (uint32_t s = 0; s < PROCESS_SIZE; s++)
            {
#ifdef REPRODUCE_ERROR_1
                state_hist_grf[stage * BINCOUNT + bins[s]]++;// 256K * 4 * 1.25 = 1310720 instr for grf indirect addr
#endif // REPRODUCE_ERROR_1
            }
        }
    }

    //atomic add to the state counter in slm
    #pragma unroll
    for (uint32_t s = 0; s < BINCOUNT * STAGES; s+=16) {
        simd<uint32_t, 16> offset(0, sizeof(global_hist_t));
        lsc_slm_atomic_update<atomic_op::add, global_hist_t, 16>(s*sizeof(global_hist_t)+offset, state_hist_grf.template select<16, 1>(s), 1);
    }

    barrier();

    {
        // bin count 256, 4 stages, 1K uint32_t, by 64 threads, happen to by 16-wide each thread. will not work for other config.
        constexpr uint32_t BUFFER_SIZE = STAGES * BINCOUNT;
        constexpr uint32_t THREAD_SIZE = BUFFER_SIZE / THREAD_PER_TG;
        simd<global_hist_t, THREAD_SIZE> group_hist = slm_block_load<global_hist_t, THREAD_SIZE>(local_tid*THREAD_SIZE*sizeof(global_hist_t));
        simd<uint32_t, THREAD_SIZE> offset(0, 4);
        lsc_atomic_update<atomic_op::add>(p_global_offset + local_tid*THREAD_SIZE, offset, group_hist, simd_mask<THREAD_SIZE>(1));
    }
}

template <typename KeyT, std::uint16_t WorkGroupSize, std::uint16_t DataPerWorkItem, std::uint32_t RADIX_BITS,
          typename _Iterator>
void
radix_sort(sycl::queue q, _Iterator __first, _Iterator __last)
{
    using namespace sycl;
    using namespace __ESIMD_NS;

    static_assert(sizeof(KeyT) <= sizeof(uint32_t) || sizeof(KeyT) == sizeof(uint64_t),
                  "Tested and supported only types of specified size");

    const ::std::size_t __n = __last - __first;
    assert(__n > 1);

    constexpr ::std::uint32_t PROCESS_SIZE = 416;

    using global_hist_t = uint32_t;
    constexpr uint32_t BINCOUNT = 1 << RADIX_BITS;
    constexpr uint32_t HW_TG_COUNT = 64;
    constexpr uint32_t THREAD_PER_TG = 64;
    constexpr uint32_t SWEEP_PROCESSING_SIZE = PROCESS_SIZE;
    const uint32_t sweep_tg_count = __dpl_ceiling_div(__n, THREAD_PER_TG * SWEEP_PROCESSING_SIZE);
    constexpr uint32_t NBITS = sizeof(KeyT) * 8;
    constexpr uint32_t STAGES = __dpl_ceiling_div(NBITS, RADIX_BITS);

    //types are messy. now all are uint32_t
    const uint32_t SYNC_BUFFER_SIZE = sweep_tg_count * BINCOUNT * STAGES * sizeof(global_hist_t); //bytes
    constexpr uint32_t GLOBAL_OFFSET_SIZE = BINCOUNT * STAGES * sizeof(global_hist_t);
    size_t temp_buffer_size = GLOBAL_OFFSET_SIZE + SYNC_BUFFER_SIZE;

    uint8_t* tmp_buffer = sycl::malloc_device<uint8_t>(temp_buffer_size, q);
    auto p_global_offset = reinterpret_cast<uint32_t*>(tmp_buffer);
    auto p_sync_buffer = reinterpret_cast<uint32_t*>(tmp_buffer + GLOBAL_OFFSET_SIZE);

    sycl::event event_chain = q.memset(tmp_buffer, 0, temp_buffer_size);

    sycl::nd_range<1> __nd_range(HW_TG_COUNT * THREAD_PER_TG, THREAD_PER_TG);
    q.submit(
        [&](sycl::handler& __cgh)
        {
            __cgh.depends_on(event_chain);

            __cgh.parallel_for(
                __nd_range, [=](sycl::nd_item<1> __nd_item) [[intel::sycl_explicit_simd]] {
                    global_histogram<KeyT, _Iterator, RADIX_BITS, HW_TG_COUNT, THREAD_PER_TG, true>(
                        __nd_item, __n, __first, p_global_offset, p_sync_buffer);
                });
        });

    sycl::free(tmp_buffer, q);
}

int main()
{
    try
    {
        std::size_t size = (1 << 18) + 1;

        sycl::queue q;
        uint64_t* pData = sycl::malloc_shared<uint64_t>(size, q);
        radix_sort<uint64_t /* KeyT */, 256 /* WorkGroupSize */, 16 /* DataPerWorkItem */, 8 /* RADIX_BITS */>(q, pData,pData + size);

    }
    catch (const ::std::exception& exc)
    {
        std::cout << "Exception: " << exc.what() << std::endl;
        return -1;
    }

    return 0;
}
