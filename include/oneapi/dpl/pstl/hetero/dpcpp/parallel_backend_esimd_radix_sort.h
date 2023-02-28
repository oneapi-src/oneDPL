// -*- C++ -*-
//===-- parallel_backend_esimd_radix_sort.h --------------------------------===//
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

#ifndef _ONEDPL_parallel_backend_esimd_radix_sort_H
#define _ONEDPL_parallel_backend_esimd_radix_sort_H

#include <ext/intel/esimd.hpp>
#include "sycl_defs.h"

namespace oneapi::dpl::experimental::esimd::impl
{

template<typename SIMD, typename Input, std::enable_if_t<std::is_pointer_v<Input>, bool> = true>
inline void
copy_from(SIMD& simd, const Input& input, uint32_t offset)
{
    simd.copy_from(input + offset);
}

template<typename SIMD, typename Input, std::enable_if_t<!std::is_pointer_v<Input>, bool> = true>
inline void
copy_from(SIMD& simd, const Input& input, uint32_t offset)
{
    simd.copy_from(input, offset);
}

template<typename SIMD, typename Output, std::enable_if_t<std::is_pointer_v<Output>, bool> = true>
inline void
copy_to(const SIMD& simd, Output& output, uint32_t offset)
{
    simd.copy_to(output + offset);
}

template<typename SIMD, typename Output, std::enable_if_t<!std::is_pointer_v<Output>, bool> = true>
inline void
copy_to(const SIMD& simd, Output& output, uint32_t offset)
{
    simd.copy_to(output, offset);
}

template <typename RT, typename T>
inline sycl::ext::intel::esimd::simd<RT, 32> scan(sycl::ext::intel::esimd::simd<T, 32> src) {
    sycl::ext::intel::esimd::simd<RT, 32> result;
    result.template select<8, 4>(0) = src.template select<8, 4>(0);
    result.template select<8, 4>(1) = src.template select<8, 4>(1) + src.template select<8, 4>(0);
    result.template select<8, 4>(2) = src.template select<8, 4>(2) + result.template select<8, 4>(1);
    result.template select<8, 4>(3) = src.template select<8, 4>(3) + result.template select<8, 4>(2);
    result.template select<4, 1>(4) = result.template select<4, 1>(4) + result[3];
    result.template select<4, 1>(8) = result.template select<4, 1>(8) + result[7];
    result.template select<4, 1>(12) = result.template select<4, 1>(12) + result[11];
    result.template select<4, 1>(16) = result.template select<4, 1>(16) + result[15];
    result.template select<4, 1>(20) = result.template select<4, 1>(20) + result[19];
    result.template select<4, 1>(24) = result.template select<4, 1>(24) + result[23];
    result.template select<4, 1>(28) = result.template select<4, 1>(28) + result[27];
    return result;
}

template <typename RT, typename T>
inline sycl::ext::intel::esimd::simd<RT, 16> scan(sycl::ext::intel::esimd::simd<T, 16> src) {
    sycl::ext::intel::esimd::simd<RT, 16> result;
    result.template select<4, 4>(0) = src.template select<4, 4>(0);
    result.template select<4, 4>(1) = src.template select<4, 4>(1) + src.template select<4, 4>(0);
    result.template select<4, 4>(2) = src.template select<4, 4>(2) + result.template select<4, 4>(1);
    result.template select<4, 4>(3) = src.template select<4, 4>(3) + result.template select<4, 4>(2);
    result.template select<4, 1>(4) = result.template select<4, 1>(4) + result[3];
    result.template select<4, 1>(8) = result.template select<4, 1>(8) + result[7];
    result.template select<4, 1>(12) = result.template select<4, 1>(12) + result[11];
    return result;
}

constexpr auto
div_up(auto a, auto b)
{
    return (a + b - 1) / b;
}

template <typename InT, typename KeyT, uint32_t RADIX_BITS, uint32_t TG_COUNT, uint32_t THREAD_PER_TG>
void
global_histogram(auto idx, size_t n, InT in, uint32_t *p_global_offset, uint32_t *p_sync_buffer) {
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
    constexpr uint32_t STAGES = div_up(NBITS, RADIX_BITS);
    constexpr uint32_t PROCESS_SIZE = 128;
    constexpr uint32_t addr_step = TG_COUNT * THREAD_PER_TG * PROCESS_SIZE;

    uint32_t local_tid = idx.get_local_linear_id();
    uint32_t tid = idx.get_global_linear_id();

    constexpr uint32_t SYNC_SEGMENT_COUNT = 64;
    constexpr uint32_t SYNC_SEGMENT_SIZE_DW = 128;
    if (tid < SYNC_SEGMENT_COUNT) {
        simd<uint32_t, SYNC_SEGMENT_SIZE_DW> sync_init = 0;
        sync_init.copy_to(p_sync_buffer + SYNC_SEGMENT_SIZE_DW * tid);
    }

    if ((tid - local_tid) * PROCESS_SIZE > n) {
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
    bin_t MASK = BINCOUNT-1;

    device_addr_t read_addr;
    for (read_addr = tid * PROCESS_SIZE; read_addr < n; read_addr += addr_step) {
        if (read_addr+PROCESS_SIZE < n) {
            // keys.copy_from(in+read_addr);
            copy_from(keys, in, read_addr);
        }
        else
        {
            simd<uint32_t, 16> lane_id(0, 1);
            #pragma unroll
            for (uint32_t s = 0; s<PROCESS_SIZE; s+=16) {
                simd_mask<16> m = (s+lane_id)<(n-read_addr);
                simd<KeyT, 16> source = lsc_gather<KeyT, 1,
                        // lsc_data_size::default_size, cache_hint::cached, cache_hint::cached, 16>(in+read_addr+s, lane_id*sizeof(KeyT), m);
                        lsc_data_size::default_size, cache_hint::cached, cache_hint::cached, 16>(in, sycl::ext::intel::esimd::simd((lane_id + read_addr+s)*sizeof(KeyT)), m);

                keys.template select<16, 1>(s) = merge(source, simd<KeyT, 16>(-1), m);
            }
        }
        #pragma unroll
        for (uint32_t i = 0; i < STAGES; i++) //4*3 = 12 instr
        {
            bins = (keys >> (i * RADIX_BITS))&MASK;
            #pragma unroll
            for (uint32_t s = 0; s < PROCESS_SIZE; s++)
            {
                state_hist_grf[i * BINCOUNT + bins[s]]++;// 256K * 4 * 1.25 = 1310720 instr for grf indirect addr
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

template <typename T, uint32_t R, uint32_t C>
class simd2d:public sycl::ext::intel::esimd::simd<T, R*C> {
    public:
        auto row(uint16_t r) {return this->template bit_cast_view<T, R, C>().row(r);}
        template <int SizeY, int StrideY, int SizeX, int StrideX>
        auto select(uint16_t OffsetY = 0, uint16_t OffsetX = 0) {
            return this->template bit_cast_view<T, R, C>().template select<SizeY, StrideY, SizeX, StrideX>(OffsetY, OffsetX);
        }
};

void inline init_global_sync(uint32_t * psync, uint32_t tg_id) {
    using namespace __ESIMD_NS;
    using namespace __ESIMD_ENS;
    simd<uint32_t, 16> lane_id(0, 4);
    simd<uint32_t, 16> old_value = lsc_atomic_update<atomic_op::load, uint32_t, 16>(psync, lane_id, 1);
    if (tg_id == 0) {
        if (!(old_value==1).all()) {
            lsc_atomic_update<atomic_op::store, uint32_t, 16>(psync, lane_id, 1, 1);
        }
    } else {
        uint32_t try_count = 0;
        while (!(old_value==1).all()) {
            old_value = lsc_atomic_update<atomic_op::load, uint32_t, 16>(psync, lane_id, 1);
            if (try_count++ > 10240) break;
        }
    }
}

void inline global_sync(uint32_t *psync, uint32_t sync_id, uint32_t count, uint32_t gid, uint32_t tid) {
    using namespace __ESIMD_NS;
    using namespace __ESIMD_ENS;
    //assume initial is 1, do inc, then repeat load until count is met, then the first one atomic reduce by count to reset to 1, do not use store to 1, because second round might started.
    psync += sync_id;
    // uint32_t prev = lsc_atomic_update<atomic_op::inc, uint32_t, 1>(psync, 0, 1)[0];
    uint32_t prev = lsc_atomic_update<atomic_op::inc, uint32_t, 1>(psync, simd<uint32_t, 1>(0), 1)[0];
    uint32_t current;
    current = -1;
    uint32_t try_count = 0;
    while (current != count+1) {
        // current=lsc_atomic_update<atomic_op::load, uint32_t, 1>(psync, 0, 1)[0];
        current=lsc_atomic_update<atomic_op::load, uint32_t, 1>(psync, simd<uint32_t, 1>(0), 1)[0];
        if (try_count++ > 5120) break;
    }
}

void inline global_wait(uint32_t *psync, uint32_t sync_id, uint32_t count, uint32_t gid, uint32_t tid) {
    using namespace __ESIMD_NS;
    using namespace __ESIMD_ENS;
    //assume initial is 1, do inc, then repeat load until count is met, then the first one atomic reduce by count to reset to 1, do not use store to 1, because second round might started.
    psync += sync_id;
    uint32_t current = -1;
    uint32_t try_count = 0;
    while (current != count) {
        // current=lsc_atomic_update<atomic_op::load, uint32_t, 1>(psync, 0, 1)[0];
        current=lsc_atomic_update<atomic_op::load, uint32_t, 1>(psync, simd<uint32_t, 1>(0), 1)[0];
        if (try_count++ > 5120) break;
    }
}

template <typename Acc, typename KeyT, uint32_t RADIX_BITS, uint32_t PROCESS_SIZE>
void
one_workgroup_kernel(const Acc& acc, auto idx, uint32_t n, uint32_t THREAD_PER_TG)
{
    using namespace sycl;
    using namespace __ESIMD_NS;
    using namespace __ESIMD_ENS;

    using bin_t = uint16_t;
    using hist_t = uint16_t;
    using device_addr_t = uint32_t;

    uint32_t local_tid = idx.get_local_linear_id();
    constexpr uint32_t BIN_COUNT = 1 << RADIX_BITS;
    constexpr uint32_t NBITS = sizeof(KeyT) * 8;
    constexpr uint32_t STAGES = div_up(NBITS, RADIX_BITS);
    constexpr uint32_t MAX_THREAD_PER_TG = 64;
    constexpr bin_t MASK = BIN_COUNT - 1;

    constexpr uint32_t REORDER_SLM_SIZE = PROCESS_SIZE * sizeof(KeyT) * MAX_THREAD_PER_TG; // reorder buffer
    constexpr uint32_t BIN_HIST_SLM_SIZE = BIN_COUNT * sizeof(hist_t) * MAX_THREAD_PER_TG; // bin hist working buffer
    constexpr uint32_t INCOMING_OFFSET_SLM_SIZE = (BIN_COUNT + 1) * sizeof(hist_t);        // incoming offset buffer

    // max SLM is 256 * 4 * 64 + 256 * 2 * 64 + 257*2, 97KB, when  PROCESS_SIZE = 256, BIN_COUNT = 256
    // to support 512 processing size, we can use all SLM as reorder buffer with cost of more barrier
    slm_init(std::max(REORDER_SLM_SIZE, BIN_HIST_SLM_SIZE + INCOMING_OFFSET_SLM_SIZE));
    uint32_t slm_reorder_start = 0;
    uint32_t slm_bin_hist_start = 0;
    uint32_t slm_incoming_offset = slm_bin_hist_start + BIN_HIST_SLM_SIZE;

    uint32_t slm_reorder = slm_reorder_start + local_tid * PROCESS_SIZE * sizeof(KeyT);
    uint32_t slm_bin_hist_this_thread = slm_bin_hist_start + local_tid * BIN_COUNT * sizeof(hist_t);

    simd<hist_t, BIN_COUNT> bin_offset;
    simd<device_addr_t, PROCESS_SIZE> write_addr;
    simd<KeyT, PROCESS_SIZE> keys;
    simd<bin_t, PROCESS_SIZE> bins;
    simd<device_addr_t, 16> lane_id(0, 1);

    device_addr_t io_offset = PROCESS_SIZE * local_tid;

#pragma unroll
    for (uint32_t s = 0; s < PROCESS_SIZE; s += 16)
    {
        simd_mask<16> m = (io_offset + lane_id + s) < n;
        keys.template select<16, 1>(s) =
            // merge(gather(p_acc + io_offset + s, lane_id * uint32_t(sizeof(KeyT))),
            //       simd<KeyT, 16>(-1), m);
            merge(gather<KeyT>(acc, lane_id * uint32_t(sizeof(KeyT)), (io_offset + s) * sizeof(KeyT)),
                  simd<KeyT, 16>(-1), m);
    }

    for (uint32_t stage = 0; stage < STAGES; stage++)
    {
        bins = (keys >> (stage * RADIX_BITS)) & MASK;
        bin_offset = 0;
#pragma unroll
        for (uint32_t s = 0; s < PROCESS_SIZE; s += 1)
        {
            write_addr[s] = bin_offset[bins[s]];
            bin_offset[bins[s]] += 1;
        }

            /*
            first write to slm,
            then do column scan by group, each thread to 32c*8r,
            then last row do exclusive scan as incoming offset
            then every thread add local sum with sum of previous group and incoming offset
            */
        {
            barrier();
            constexpr uint32_t HIST_STRIDE = sizeof(hist_t) * BIN_COUNT;
#pragma unroll
            for (uint32_t s = 0; s < BIN_COUNT; s += 128)
            {
                lsc_slm_block_store<uint32_t, 64>(
                    slm_bin_hist_this_thread + s * sizeof(hist_t),
                    bin_offset.template select<128, 1>(s).template bit_cast_view<uint32_t>());
            }
            barrier();
            constexpr uint32_t BIN_SUMMARY_GROUP_SIZE = 8;
            if (local_tid < BIN_SUMMARY_GROUP_SIZE)
            {
                constexpr uint32_t BIN_WIDTH = BIN_COUNT / BIN_SUMMARY_GROUP_SIZE;
                constexpr uint32_t BIN_WIDTH_UD = BIN_WIDTH * sizeof(hist_t) / sizeof(uint32_t);
                uint32_t slm_bin_hist_summary_offset = slm_bin_hist_start + local_tid * BIN_WIDTH * sizeof(hist_t);
                simd<hist_t, BIN_WIDTH> thread_grf_hist_summary;
                simd<uint32_t, BIN_WIDTH_UD> tmp;

                thread_grf_hist_summary.template bit_cast_view<uint32_t>() =
                    lsc_slm_block_load<uint32_t, BIN_WIDTH_UD>(slm_bin_hist_summary_offset);
                slm_bin_hist_summary_offset += HIST_STRIDE;
                for (uint32_t s = 1; s < THREAD_PER_TG - 1; s++)
                {
                    tmp = lsc_slm_block_load<uint32_t, BIN_WIDTH_UD>(slm_bin_hist_summary_offset);
                    thread_grf_hist_summary += tmp.template bit_cast_view<hist_t>();
                    lsc_slm_block_store<uint32_t, BIN_WIDTH_UD>(
                        slm_bin_hist_summary_offset, thread_grf_hist_summary.template bit_cast_view<uint32_t>());
                    slm_bin_hist_summary_offset += HIST_STRIDE;
                }
                tmp = lsc_slm_block_load<uint32_t, BIN_WIDTH_UD>(slm_bin_hist_summary_offset);
                thread_grf_hist_summary += tmp.template bit_cast_view<hist_t>();
                thread_grf_hist_summary = scan<hist_t, hist_t>(thread_grf_hist_summary);
                lsc_slm_block_store<uint32_t, BIN_WIDTH_UD>(slm_bin_hist_summary_offset,
                                                            thread_grf_hist_summary.template bit_cast_view<uint32_t>());
            }
            barrier();
            if (local_tid == 0)
            {
                simd<hist_t, BIN_COUNT> grf_hist_summary;
                simd<hist_t, BIN_COUNT + 1> grf_hist_summary_scan;
#pragma unroll
                for (uint32_t s = 0; s < BIN_COUNT; s += 128)
                {
                    grf_hist_summary.template select<128, 1>(s).template bit_cast_view<uint32_t>() =
                        lsc_slm_block_load<uint32_t, 64>(slm_bin_hist_start + (THREAD_PER_TG - 1) * HIST_STRIDE +
                                                         s * sizeof(hist_t));
                }
                grf_hist_summary_scan[0] = 0;
                grf_hist_summary_scan.template select<32, 1>(1) = grf_hist_summary.template select<32, 1>(0);
#pragma unroll
                for (uint32_t i = 32; i < BIN_COUNT; i += 32)
                {
                    grf_hist_summary_scan.template select<32, 1>(i + 1) =
                        grf_hist_summary.template select<32, 1>(i) + grf_hist_summary_scan[i];
                }
#pragma unroll
                for (uint32_t s = 0; s < BIN_COUNT; s += 128)
                {
                    lsc_slm_block_store<uint32_t, 64>(
                        slm_incoming_offset + s * sizeof(hist_t),
                        grf_hist_summary_scan.template select<128, 1>(s).template bit_cast_view<uint32_t>());
                }
            }
            barrier();
            {
#pragma unroll
                for (uint32_t s = 0; s < BIN_COUNT; s += 128)
                {
                    bin_offset.template select<128, 1>(s).template bit_cast_view<uint32_t>() =
                        lsc_slm_block_load<uint32_t, 64>(slm_incoming_offset + s * sizeof(hist_t));
                }
                if (local_tid > 0)
                {
#pragma unroll
                    for (uint32_t s = 0; s < BIN_COUNT; s += 128)
                    {
                        simd<hist_t, 128> group_local_sum;
                        group_local_sum.template bit_cast_view<uint32_t>() = lsc_slm_block_load<uint32_t, 64>(
                            slm_bin_hist_start + (local_tid - 1) * HIST_STRIDE + s * sizeof(hist_t));
                        bin_offset.template select<128, 1>(s) += group_local_sum;
                    }
                }
            }
            barrier();
        }

#pragma unroll
        for (uint32_t s = 0; s < PROCESS_SIZE; s += 16)
        {
            simd<uint16_t, 16> bins_uw = bins.template select<16, 1>(s);
            write_addr.template select<16, 1>(s) += bin_offset.template iselect(bins_uw);
        }

        if (stage != STAGES - 1)
        {
#pragma unroll
            for (uint32_t s = 0; s < PROCESS_SIZE; s += 16)
            {
                slm_scatter<KeyT, 16>(write_addr.template select<16, 1>(s) * sizeof(KeyT) +
                                                      slm_reorder_start,
                                                  keys.template select<16, 1>(s));
            }
            barrier();
#pragma unroll
            for (uint32_t s = 0; s < PROCESS_SIZE; s += 64)
            {
                keys.template select<64, 1>(s) =
                    lsc_slm_block_load<KeyT, 64>(slm_reorder + s * sizeof(KeyT));
            }
        }
    }
#pragma unroll
    for (uint32_t s = 0; s < PROCESS_SIZE; s += 16)
    {
        // scatter<KeyT, 16>(p_acc, write_addr.template select<16, 1>(s) * sizeof(KeyT),
        //                               keys.template select<16, 1>(s), (local_tid * PROCESS_SIZE + lane_id + s) < n);
        scatter<KeyT, 16>(acc, write_addr.template select<16, 1>(s) * sizeof(KeyT),
                                      keys.template select<16, 1>(s), 0, (local_tid * PROCESS_SIZE + lane_id + s) < n);
    }
}

template <typename _Range, typename KeyT, uint32_t RADIX_BITS>
void
one_workgroup(sycl::queue& q, _Range&& rng, std::size_t n)
{
    using namespace sycl;

    constexpr uint32_t BINCOUNT = 1 << RADIX_BITS;
    constexpr uint32_t MAX_TG_COUNT = 64;
    constexpr uint32_t MIN_TG_COUNT = 8;
    uint32_t PROCESS_SIZE = 64;
    if (n < MIN_TG_COUNT * 64)
    {
        PROCESS_SIZE = 64;
    }
    else if (n < MIN_TG_COUNT * 128)
    {
        PROCESS_SIZE = 128;
    }
    else
    {
        PROCESS_SIZE = 256;
    };
    assert((PROCESS_SIZE == 64) || (PROCESS_SIZE == 128) || (PROCESS_SIZE == 256));

    uint32_t TG_COUNT = div_up(n, PROCESS_SIZE);
    TG_COUNT = std::max(TG_COUNT, MIN_TG_COUNT);
    assert((TG_COUNT <= MAX_TG_COUNT) && (TG_COUNT >= MIN_TG_COUNT));

    auto acc = rng.__m_acc;
    sycl::event e;
    {
        nd_range<1> Range((range<1>(TG_COUNT)), (range<1>(TG_COUNT)));
        if (PROCESS_SIZE == 64)
        {
            e = q.submit([&](handler& cgh) {
                oneapi::dpl::__ranges::__require_access(cgh, rng);
                cgh.parallel_for<class kernel_radix_sort_one_tg_64>(Range, [=](nd_item<1> idx) [[intel::sycl_explicit_simd]] {
                    one_workgroup_kernel<decltype(acc), KeyT, RADIX_BITS, 64>(acc, idx, n, TG_COUNT);
                });
            });
        }
        else if (PROCESS_SIZE == 128)
        {
            e = q.submit([&](handler& cgh) {
                oneapi::dpl::__ranges::__require_access(cgh, rng);
                cgh.parallel_for<class kernel_radix_sort_one_tg_128>(Range, [=](nd_item<1> idx) [[intel::sycl_explicit_simd]] {
                    one_workgroup_kernel<decltype(acc), KeyT, RADIX_BITS, 128>(acc, idx, n, TG_COUNT);
                });
            });
        }
        else if (PROCESS_SIZE == 256)
        {
            e = q.submit([&](handler& cgh) {
                oneapi::dpl::__ranges::__require_access(cgh, rng);
                cgh.parallel_for<class kernel_radix_sort_one_tg_256>(Range, [=](nd_item<1> idx) [[intel::sycl_explicit_simd]] {
                    one_workgroup_kernel<decltype(acc), KeyT, RADIX_BITS, 256>(acc, idx, n, TG_COUNT);
                });
            });
        }
    }
    e.wait();
}

template <typename Acc, typename KeyT, uint32_t RADIX_BITS, uint32_t THREAD_PER_TG, uint32_t PROCESS_SIZE>
void cooperative_kernel(const Acc& acc, auto idx, size_t n, uint32_t *p_global_buffer) {
    using namespace sycl;
    using namespace __ESIMD_NS;
    using namespace __ESIMD_ENS;

    using bin_t = uint16_t;
    using hist_t = uint32_t;
    using global_hist_t = uint32_t;
    using device_addr_t = uint32_t;

    uint32_t tg_id = idx.get_group(0);
    uint32_t tg_count = idx.get_group_range(0);

    uint32_t local_tid = idx.get_local_linear_id();
    constexpr uint32_t BIN_COUNT = 1 << RADIX_BITS;
    constexpr uint32_t NBITS =  sizeof(KeyT) * 8;
    constexpr uint32_t STAGES = div_up(NBITS, RADIX_BITS);
    constexpr uint32_t TG_PROCESS_SIZE = PROCESS_SIZE * THREAD_PER_TG;
    constexpr bin_t MASK = BIN_COUNT - 1;

    constexpr uint32_t BIN_HIST_SLM_SIZE = BIN_COUNT * sizeof(hist_t) * THREAD_PER_TG;  // bin hist working buffer, 64K for DW hist
    constexpr uint32_t INCOMING_OFFSET_SLM_SIZE = (BIN_COUNT+16)*sizeof(global_hist_t); // incoming offset buffer
    constexpr uint32_t GLOBAL_SCAN_SIZE = (BIN_COUNT+16)*sizeof(global_hist_t);

    slm_init( BIN_HIST_SLM_SIZE + INCOMING_OFFSET_SLM_SIZE + GLOBAL_SCAN_SIZE);
    uint32_t slm_bin_hist_start = 0;
    uint32_t slm_incoming_start = slm_bin_hist_start + BIN_HIST_SLM_SIZE;
    uint32_t slm_global_scan_start = slm_incoming_start + INCOMING_OFFSET_SLM_SIZE;

    uint32_t slm_bin_hist_this_thread = slm_bin_hist_start + local_tid * BIN_COUNT * sizeof(hist_t);

    uint32_t global_sync_buffer_size = 1024; //1K uint32_t for sync buffer
    uint32_t global_bin_start_buffer_size = BIN_COUNT * sizeof(global_hist_t)+16;
    uint32_t global_bin_hist_size = tg_count * BIN_COUNT * sizeof(global_hist_t);
    uint32_t *p_sync_buffer = p_global_buffer;
    uint32_t *p_global_bin_start_buffer = p_sync_buffer + global_sync_buffer_size;
    uint32_t *p_global_bin_hist = p_global_bin_start_buffer + global_bin_start_buffer_size;
    uint32_t *p_global_bin_hist_tg = p_global_bin_hist + tg_id * BIN_COUNT * sizeof(global_hist_t)/sizeof(uint32_t);

    simd<hist_t, BIN_COUNT> bin_offset;
    simd<device_addr_t, PROCESS_SIZE> write_addr;
    simd<KeyT, PROCESS_SIZE> keys;
    simd<bin_t, PROCESS_SIZE> bins;
    simd<device_addr_t, 16> lane_id(0, 1);

    device_addr_t io_offset = tg_id * TG_PROCESS_SIZE + PROCESS_SIZE * local_tid;

    constexpr uint32_t BIN_GROUPS = 8;
    constexpr uint32_t THREAD_PER_BIN_GROUP = THREAD_PER_TG / BIN_GROUPS;
    constexpr uint32_t BIN_WIDTH = BIN_COUNT / BIN_GROUPS;
    constexpr uint32_t BIN_WIDTH_UD = BIN_COUNT / BIN_GROUPS * sizeof(hist_t) / sizeof(uint32_t);
    constexpr uint32_t BIN_HEIGHT = THREAD_PER_TG / THREAD_PER_BIN_GROUP;

    if (local_tid == 0) init_global_sync(p_sync_buffer, tg_id);
    barrier();

    #pragma unroll
    for (uint32_t s = 0; s<PROCESS_SIZE; s+=16) {
        simd_mask<16> m = (io_offset+lane_id+s)<n;
        simd<KeyT, 16> source = lsc_gather<KeyT, 1,
                // lsc_data_size::default_size, cache_hint::uncached, cache_hint::cached, 16>(acc+io_offset+s, lane_id*uint32_t(sizeof(KeyT)), m);
                lsc_data_size::default_size, cache_hint::uncached, cache_hint::cached, 16>(acc, sycl::ext::intel::esimd::simd<KeyT, 16>((lane_id + io_offset+s)*uint32_t(sizeof(KeyT))), m);
        keys.template select<16, 1>(s) = merge(source, simd<KeyT, 16>(-1), m);
    }

    for (uint32_t stage=0; stage < STAGES; stage++) {
        bins = (keys >> (stage * RADIX_BITS)) & MASK;
        bin_offset = 0;
        #pragma unroll
        for (uint32_t s = 0; s<PROCESS_SIZE; s+=1) {
            write_addr[s] = bin_offset[bins[s]];
            bin_offset[bins[s]] += 1;
        }
        /*
        first write to slm,
        then do column scan by group, each thread to 32c*8r,
        then last row do exclusive scan as incoming offset
        then every thread add local sum with sum of previous group and incoming offset
        */
        {
            barrier();
            // put local hist to slm
            #pragma unroll
            for (uint32_t s = 0; s<BIN_COUNT; s+=64) {
                lsc_slm_block_store<uint32_t, 64>(slm_bin_hist_this_thread + s*sizeof(hist_t), bin_offset.template select<64, 1>(s).template bit_cast_view<uint32_t>());
            }
            barrier();
            // small group sum for local hist
            constexpr uint32_t BIN_GROUPS = 8;
            constexpr uint32_t THREAD_PER_BIN_GROUP = THREAD_PER_TG / BIN_GROUPS;
            constexpr uint32_t BIN_WIDTH = BIN_COUNT / BIN_GROUPS;
            constexpr uint32_t BIN_HEIGHT = THREAD_PER_TG / THREAD_PER_BIN_GROUP;
            constexpr uint32_t HIST_STRIDE = BIN_COUNT * sizeof(hist_t);
            uint32_t THREAD_GID = local_tid / THREAD_PER_BIN_GROUP;
            uint32_t THREAD_LTID = local_tid % THREAD_PER_BIN_GROUP;
            {
                uint32_t slm_bin_hist_ingroup_offset = slm_bin_hist_start + THREAD_GID * BIN_WIDTH * sizeof(hist_t) + THREAD_LTID * BIN_HEIGHT * HIST_STRIDE;
                simd2d<hist_t, BIN_HEIGHT, BIN_WIDTH> thread_grf_hist;
                #pragma unroll
                for (uint32_t s = 0; s<BIN_HEIGHT; s++) {
                    thread_grf_hist.row(s).template bit_cast_view<uint32_t>() = lsc_slm_block_load<uint32_t, BIN_WIDTH_UD>(slm_bin_hist_ingroup_offset + s * HIST_STRIDE);
                }
                #pragma unroll
                for (uint32_t s = 1; s<BIN_HEIGHT; s++) {
                    thread_grf_hist.row(s) += thread_grf_hist.row(s-1);
                }
                #pragma unroll
                for (uint32_t s = 1; s<BIN_HEIGHT; s++) {
                    lsc_slm_block_store<uint32_t, BIN_WIDTH_UD>(slm_bin_hist_ingroup_offset + s * HIST_STRIDE, thread_grf_hist.row(s).template bit_cast_view<uint32_t>());
                }
            }

            barrier();
            // thread group sum for groups
            if (THREAD_LTID == 0) {
                uint32_t slm_bin_hist_group_summary_offset = slm_bin_hist_start + THREAD_GID * BIN_WIDTH * sizeof(hist_t) +  (BIN_HEIGHT-1) * HIST_STRIDE;
                simd2d<hist_t, THREAD_PER_BIN_GROUP, BIN_WIDTH> thread_grf_hist_summary;
                #pragma unroll
                for (uint32_t s = 0; s<THREAD_PER_BIN_GROUP; s++) {
                    thread_grf_hist_summary.row(s).template bit_cast_view<uint32_t>() = lsc_slm_block_load<uint32_t, BIN_WIDTH_UD>(slm_bin_hist_group_summary_offset + s * BIN_HEIGHT * HIST_STRIDE);
                }
                #pragma unroll
                for (uint32_t s = 1; s<THREAD_PER_BIN_GROUP; s++) {
                    thread_grf_hist_summary.row(s) += thread_grf_hist_summary.row(s-1);
                }
                #pragma unroll
                for (uint32_t s = 1; s<THREAD_PER_BIN_GROUP-1; s++) {
                    lsc_slm_block_store<uint32_t, BIN_WIDTH_UD>(slm_bin_hist_group_summary_offset + s * BIN_HEIGHT * HIST_STRIDE, thread_grf_hist_summary.row(s).template bit_cast_view<uint32_t>());
                }
                simd<global_hist_t, BIN_WIDTH> group_hist_sum = thread_grf_hist_summary.row(THREAD_PER_BIN_GROUP-1);
                group_hist_sum.copy_to(p_global_bin_hist_tg + THREAD_GID * BIN_WIDTH * sizeof(global_hist_t)/sizeof(uint32_t));
            }
            lsc_fence<lsc_memory_kind::untyped_global, lsc_fence_op::evict, lsc_scope::gpu>();
            barrier();

            // thread group scan -> change to coopeative group scan
            // first dump result to global hist buffer - done in previous part, last row is copy to global
            // global sync()
            // each tg calculate sum of previous tg
            // last tg then do exclusive scan and save to global scan buffer
            // global sync()
            // each tg read global scan buffer, add with prev tg, then put to slm incoming buffer
            if (local_tid == 0) {
                global_sync(p_sync_buffer, stage*4+0, tg_count, tg_id, local_tid);
            }
            barrier();
            {
                if (tg_id == tg_count-1) {
                    if (local_tid < 16)  {
                        //16 threads cooperative, each for BIN_COUNT/16 element.
                        uint32_t *p_global_hist_sum_per_thread = p_global_bin_hist + local_tid * (BIN_COUNT/16)* sizeof(global_hist_t)/sizeof(uint32_t);

                        simd<global_hist_t, BIN_COUNT/16> global_hist_sum(p_global_hist_sum_per_thread);
                        for (uint32_t i = 1; i<tg_count; i++) {
                            simd<global_hist_t, BIN_COUNT/16> global_hist(p_global_hist_sum_per_thread + i * BIN_COUNT * sizeof(global_hist_t)/sizeof(uint32_t));
                            global_hist_sum += global_hist;
                            global_hist_sum.copy_to(p_global_hist_sum_per_thread + i * BIN_COUNT * sizeof(global_hist_t)/sizeof(uint32_t));
                        }
                        global_hist_sum = scan<global_hist_t, global_hist_t>(global_hist_sum);
                        lsc_slm_block_store<global_hist_t, 16>(slm_global_scan_start + local_tid * (BIN_COUNT/16)* sizeof(global_hist_t), global_hist_sum);
                    }
                    barrier();//TODO: change to smaller named barrier
                    if (local_tid==0) {
                        simd<global_hist_t, BIN_COUNT/16> global_hist_sum;
                        global_hist_t prev(0);
                        #pragma unroll
                        for (uint32_t i = 0; i<BIN_COUNT; i+=16) {
                            global_hist_sum = prev + lsc_slm_block_load<global_hist_t, 16>(slm_global_scan_start + i * sizeof(global_hist_t));
                            prev = global_hist_sum[15];
                            global_hist_sum.copy_to(p_global_bin_start_buffer+1+i);
                        }
                        p_global_bin_start_buffer[0] = 0;
                    }
                    lsc_fence<lsc_memory_kind::untyped_global, lsc_fence_op::evict, lsc_scope::gpu>();
                }
                // if (stage == 0) return;
                if (local_tid == 0)  {
                    global_sync(p_sync_buffer, stage*4+1, tg_count, tg_id, local_tid);
                    {
                        simd<global_hist_t, BIN_COUNT> global_hist_start(p_global_bin_start_buffer);
                        if (tg_id != 0) {
                            global_hist_start += simd<global_hist_t, BIN_COUNT>(p_global_bin_hist_tg - BIN_COUNT * sizeof(global_hist_t)/sizeof(uint32_t));
                        }
                        #pragma unroll
                        for (uint32_t s = 0; s<BIN_COUNT; s+=64) {
                            lsc_slm_block_store<uint32_t, 64>(slm_incoming_start + s * sizeof(global_hist_t), global_hist_start.template select<64, 1>(s));
                        }
                    }
                }
            }
            barrier();
            {
                #pragma unroll
                for (uint32_t s = 0; s<BIN_COUNT; s+=64) {
                    bin_offset.template select<64, 1>(s).template bit_cast_view<uint32_t>() = lsc_slm_block_load<uint32_t, 64>(slm_incoming_start + s*sizeof(hist_t));
                }
                if (local_tid>0) {
                    #pragma unroll
                    for (uint32_t s = 0; s<BIN_COUNT; s+=64) {
                        simd<hist_t, 64> group_local_sum;
                        group_local_sum.template bit_cast_view<uint32_t>() = lsc_slm_block_load<uint32_t, 64>(slm_bin_hist_start + (local_tid-1)*HIST_STRIDE + s*sizeof(hist_t));
                        bin_offset.template select<64, 1>(s) += group_local_sum;
                    }
                }
                if ((local_tid > THREAD_PER_BIN_GROUP) && (local_tid % BIN_HEIGHT != 0)) {
                    uint32_t prev_cum_rowid = (local_tid-1) / BIN_HEIGHT * BIN_HEIGHT - 1;
                    #pragma unroll
                    for (uint32_t s = 0; s<BIN_COUNT; s+=64) {
                        simd<hist_t, 64> group_sum;
                        group_sum.template bit_cast_view<uint32_t>() = lsc_slm_block_load<uint32_t, 64>(slm_bin_hist_start + prev_cum_rowid*HIST_STRIDE + s*sizeof(hist_t));
                        bin_offset.template select<64, 1>(s) += group_sum;
                    }
                }
            }
            barrier();
        }

        #pragma unroll
        for (uint32_t s = 0; s<PROCESS_SIZE; s+=16) {
            simd<uint16_t, 16> bins_uw = bins.template select<16, 1>(s);
            write_addr.template select<16, 1>(s) += bin_offset.template iselect(bins_uw);
        }

        #pragma unroll
        for (uint32_t s = 0; s<PROCESS_SIZE; s+=16) {
            lsc_scatter<KeyT, 1, lsc_data_size::default_size, cache_hint::uncached, cache_hint::write_back, 16>(
                acc,
                write_addr.template select<16, 1>(s)*sizeof(KeyT),
                keys.template select<16, 1>(s), write_addr.template select<16, 1>(s)<n);
        }
        if (stage != STAGES - 1) {
            lsc_fence<lsc_memory_kind::untyped_global, lsc_fence_op::evict, lsc_scope::gpu>();
            barrier();
            if (local_tid == 0) {global_sync(p_sync_buffer, stage*4+2, tg_count, tg_id, local_tid);}
            barrier();
            #pragma unroll
            for (uint32_t s = 0; s<PROCESS_SIZE; s+=16) {
                simd_mask<16> m = (io_offset+lane_id+s)<n;
                simd<KeyT, 16> reordered = lsc_gather<KeyT, 1,
                        lsc_data_size::default_size, cache_hint::uncached, cache_hint::cached, 16>(
                            // acc+io_offset+s, lane_id*uint32_t(sizeof(KeyT)), m);
                            acc, sycl::ext::intel::esimd::simd<KeyT, 16>((lane_id + io_offset+s)*uint32_t(sizeof(KeyT))), m);
                keys.template select<16, 1>(s) = merge(reordered, simd<KeyT, 16>(-1), m);
            }
        }
    }
}

template <typename _Range, typename KeyT, uint32_t RADIX_BITS>
void cooperative(sycl::queue &q, _Range&& rng, std::size_t n) {
    using namespace sycl;
    using namespace __ESIMD_NS;

    constexpr uint32_t BIN_COUNT = 1 << RADIX_BITS;
    constexpr uint32_t THREAD_PER_TG = 64;
    uint32_t MAX_GROUPS = 56; //TODO: get from sycl api
    assert(n <= 256 * THREAD_PER_TG * MAX_GROUPS);

    auto acc = rng.__m_acc;
    auto p_sync = static_cast<uint32_t*>(sycl::malloc_device(1024 + (MAX_GROUPS+2) * BIN_COUNT * sizeof(uint32_t), q));
    sycl::event e;
    {
        if (n<=128 * THREAD_PER_TG * MAX_GROUPS) {
            uint32_t groups = div_up(n, 128 * THREAD_PER_TG);
            nd_range<1> Range( (range<1>(THREAD_PER_TG * groups)), (range<1>(THREAD_PER_TG)));
            e = q.submit([&](handler &cgh) {
                oneapi::dpl::__ranges::__require_access(cgh, rng);
                cgh.parallel_for<class kernel_radix_sort_cooperative_128>(
                        Range, [=](nd_item<1> idx) [[intel::sycl_explicit_simd]] {
                            cooperative_kernel<decltype(acc), KeyT, RADIX_BITS, THREAD_PER_TG, 128> (acc, idx, n, p_sync);
                        });
            });
        } else if (n<=256 * THREAD_PER_TG * MAX_GROUPS) {
            uint32_t groups = div_up(n, 256 * THREAD_PER_TG);
            nd_range<1> Range( (range<1>(THREAD_PER_TG * groups)), (range<1>(THREAD_PER_TG)));
            e = q.submit([&](handler &cgh) {
                oneapi::dpl::__ranges::__require_access(cgh, rng);
                cgh.parallel_for<class kernel_radix_sort_cooperative_256>(
                        Range, [=](nd_item<1> idx) [[intel::sycl_explicit_simd]] {
                            cooperative_kernel<decltype(acc), KeyT, RADIX_BITS, THREAD_PER_TG, 256> (acc, idx, n, p_sync);
                        });
            });
        }
    }
    e.wait();
    sycl::free(p_sync, q);
}

template <typename InT, typename OutT, typename KeyT, uint32_t RADIX_BITS, uint32_t THREAD_PER_TG, uint32_t PROCESS_SIZE>
void onesweep_kernel(auto idx, uint32_t n, uint32_t stage, InT in, OutT out, uint8_t *p_global_buffer) {
    using namespace sycl;
    using namespace __ESIMD_NS;
    using namespace __ESIMD_ENS;

    using bin_t = uint16_t;
    using hist_t = uint32_t;
    using global_hist_t = uint32_t;
    using device_addr_t = uint32_t;

    uint32_t tg_id = idx.get_group(0);
    uint32_t tg_count = idx.get_group_range(0);

    uint32_t local_tid = idx.get_local_linear_id();
    constexpr uint32_t BIN_COUNT = 1 << RADIX_BITS;
    constexpr uint32_t NBITS =  sizeof(KeyT) * 8;
    constexpr uint32_t STAGES = div_up(NBITS, RADIX_BITS);
    constexpr uint32_t TG_PROCESS_SIZE = PROCESS_SIZE * THREAD_PER_TG;
    constexpr bin_t MASK = BIN_COUNT - 1;

    constexpr uint32_t BIN_HIST_SLM_SIZE = BIN_COUNT * sizeof(hist_t) * THREAD_PER_TG;  // bin hist working buffer, 64K for DW hist
    constexpr uint32_t INCOMING_OFFSET_SLM_SIZE = (BIN_COUNT+16)*sizeof(global_hist_t); // incoming offset buffer
    constexpr uint32_t GLOBAL_SCAN_SIZE = (BIN_COUNT+16)*sizeof(global_hist_t);

    slm_init( BIN_HIST_SLM_SIZE + INCOMING_OFFSET_SLM_SIZE + GLOBAL_SCAN_SIZE);
    uint32_t slm_bin_hist_start = 0;
    uint32_t slm_incoming_start = slm_bin_hist_start + BIN_HIST_SLM_SIZE;
    uint32_t slm_global_scan_start = slm_incoming_start + INCOMING_OFFSET_SLM_SIZE;


    uint32_t slm_bin_hist_this_thread = slm_bin_hist_start + local_tid * BIN_COUNT * sizeof(hist_t);

    size_t temp_io_size = n * sizeof(KeyT);

    global_hist_t *p_global_bin_start_buffer_allstages = reinterpret_cast<global_hist_t*>(p_global_buffer);
    global_hist_t *p_global_bin_start_buffer = p_global_bin_start_buffer_allstages + BIN_COUNT * stage;
    uint32_t *p_sync_buffer = reinterpret_cast<uint32_t*>(p_global_bin_start_buffer_allstages + BIN_COUNT * STAGES);

    constexpr uint32_t SYNC_SEGMENT_COUNT = 64;
    constexpr uint32_t SYNC_SEGMENT_SIZE_DW = 128;


    simd<hist_t, BIN_COUNT> bin_offset;
    simd<device_addr_t, 16> lane_id(0, 1);

    device_addr_t io_offset = tg_id * TG_PROCESS_SIZE + PROCESS_SIZE * local_tid;

    constexpr uint32_t BIN_GROUPS = 8;
    constexpr uint32_t THREAD_PER_BIN_GROUP = THREAD_PER_TG / BIN_GROUPS;
    constexpr uint32_t BIN_WIDTH = BIN_COUNT / BIN_GROUPS;
    constexpr uint32_t BIN_WIDTH_UD = BIN_COUNT / BIN_GROUPS * sizeof(hist_t) / sizeof(uint32_t);
    constexpr uint32_t BIN_HEIGHT = THREAD_PER_TG / THREAD_PER_BIN_GROUP;

    {
        simd<KeyT, PROCESS_SIZE> keys;
        simd<bin_t, PROCESS_SIZE> bins;

        if (io_offset+PROCESS_SIZE < n) {
            // keys.copy_from(in+io_offset);
            copy_from(keys, in, io_offset);
        }
        else if (io_offset >= n) {
            keys = -1;
        }
        else {
            #pragma unroll
            for (uint32_t s = 0; s<PROCESS_SIZE; s+=16) {
                simd_mask<16> m = (io_offset+lane_id+s)<n;
                simd<KeyT, 16> source = lsc_gather<KeyT, 1,
                        lsc_data_size::default_size, cache_hint::cached, cache_hint::cached, 16>(in, sycl::ext::intel::esimd::simd((lane_id + io_offset+s)*uint32_t(sizeof(KeyT))), m);
                keys.template select<16, 1>(s) = merge(source, simd<KeyT, 16>(-1), m);
            }
        }
        bins = (keys >> (stage * RADIX_BITS)) & MASK;
        bin_offset = 0;

        #pragma unroll
        for (uint32_t s = 0; s<PROCESS_SIZE; s+=1) {
            bin_offset[bins[s]] += 1;
        }
    }
    {
        /*
        first write to slm,
        then do column scan by group, each thread to 32c*8r,
        then last row do exclusive scan as incoming offset
        then every thread add local sum with sum of previous group and incoming offset
        */
        {
            // put local hist to slm
            #pragma unroll
            for (uint32_t s = 0; s<BIN_COUNT; s+=64) {
                lsc_slm_block_store<uint32_t, 64>(slm_bin_hist_this_thread + s*sizeof(hist_t), bin_offset.template select<64, 1>(s).template bit_cast_view<uint32_t>());
            }
            barrier();
            // small group sum for local hist
            constexpr uint32_t BIN_GROUPS = 8;
            constexpr uint32_t THREAD_PER_BIN_GROUP = THREAD_PER_TG / BIN_GROUPS;
            constexpr uint32_t BIN_WIDTH = BIN_COUNT / BIN_GROUPS;
            constexpr uint32_t BIN_HEIGHT = THREAD_PER_TG / THREAD_PER_BIN_GROUP;
            constexpr uint32_t HIST_STRIDE = BIN_COUNT * sizeof(hist_t);
            uint32_t THREAD_GID = local_tid / THREAD_PER_BIN_GROUP;
            uint32_t THREAD_LTID = local_tid % THREAD_PER_BIN_GROUP;
            {
                uint32_t slm_bin_hist_ingroup_offset = slm_bin_hist_start + THREAD_GID * BIN_WIDTH * sizeof(hist_t) + THREAD_LTID * BIN_HEIGHT * HIST_STRIDE;
                simd2d<hist_t, BIN_HEIGHT, BIN_WIDTH> thread_grf_hist;
                #pragma unroll
                for (uint32_t s = 0; s<BIN_HEIGHT; s++) {
                    thread_grf_hist.row(s).template bit_cast_view<uint32_t>() = lsc_slm_block_load<uint32_t, BIN_WIDTH_UD>(slm_bin_hist_ingroup_offset + s * HIST_STRIDE);
                }
                #pragma unroll
                for (uint32_t s = 1; s<BIN_HEIGHT; s++) {
                    thread_grf_hist.row(s) += thread_grf_hist.row(s-1);
                }
                #pragma unroll
                for (uint32_t s = 1; s<BIN_HEIGHT; s++) {
                    lsc_slm_block_store<uint32_t, BIN_WIDTH_UD>(slm_bin_hist_ingroup_offset + s * HIST_STRIDE, thread_grf_hist.row(s).template bit_cast_view<uint32_t>());
                }
            }

            barrier();
            // thread group sum for groups
            simd<global_hist_t, BIN_WIDTH> group_hist_sum;
            if (THREAD_LTID == 0) {
                uint32_t slm_bin_hist_group_summary_offset = slm_bin_hist_start + THREAD_GID * BIN_WIDTH * sizeof(hist_t) +  (BIN_HEIGHT-1) * HIST_STRIDE;
                simd2d<hist_t, THREAD_PER_BIN_GROUP, BIN_WIDTH> thread_grf_hist_summary;
                #pragma unroll
                for (uint32_t s = 0; s<THREAD_PER_BIN_GROUP; s++) {
                    thread_grf_hist_summary.row(s).template bit_cast_view<uint32_t>() = lsc_slm_block_load<uint32_t, BIN_WIDTH_UD>(slm_bin_hist_group_summary_offset + s * BIN_HEIGHT * HIST_STRIDE);
                }
                #pragma unroll
                for (uint32_t s = 1; s<THREAD_PER_BIN_GROUP; s++) {
                    thread_grf_hist_summary.row(s) += thread_grf_hist_summary.row(s-1);
                }
                #pragma unroll
                for (uint32_t s = 1; s<THREAD_PER_BIN_GROUP-1; s++) {
                    lsc_slm_block_store<uint32_t, BIN_WIDTH_UD>(slm_bin_hist_group_summary_offset + s * BIN_HEIGHT * HIST_STRIDE, thread_grf_hist_summary.row(s).template bit_cast_view<uint32_t>());
                }
                group_hist_sum = thread_grf_hist_summary.row(THREAD_PER_BIN_GROUP-1);
            }
            // wait for previous TG pass over starting point
            // atomic add starting point with group_hist_sum
            // and pass to next
            {
                //sync offset will be distributed to 64 banks to reduce latency
                //for each tg, sync segment id is (tg_id % 64), sync target is ((tg_id+63) / 64) * BIN_GROUPS,
                if (local_tid == 0) {
                    uint32_t sync_segment_id = tg_id % SYNC_SEGMENT_COUNT;
                    uint32_t sync_segment_target = ((tg_id + SYNC_SEGMENT_COUNT-1) / SYNC_SEGMENT_COUNT) * BIN_GROUPS;
                    uint32_t sync_offset = sync_segment_id * SYNC_SEGMENT_SIZE_DW + stage;
                    global_wait(p_sync_buffer, sync_offset, sync_segment_target, tg_id, local_tid);
                }
                barrier();
                if (THREAD_LTID == 0) {
                    uint32_t pass_over_sync_segment_id = (tg_id + 1) % SYNC_SEGMENT_COUNT;
                    uint32_t pass_over_sync_offset = pass_over_sync_segment_id * SYNC_SEGMENT_SIZE_DW + stage;
                    simd<uint32_t, BIN_WIDTH> update_lanes(0, sizeof(global_hist_t));
                    group_hist_sum = lsc_atomic_update<atomic_op::add, uint32_t, BIN_WIDTH>(p_global_bin_start_buffer+THREAD_GID*BIN_WIDTH, update_lanes, group_hist_sum, 1);
                    lsc_slm_block_store<uint32_t, BIN_WIDTH>(slm_incoming_start + THREAD_GID * BIN_WIDTH * sizeof(global_hist_t), group_hist_sum);
                    // lsc_atomic_update<atomic_op::inc, uint32_t, 1>(p_sync_buffer+pass_over_sync_offset, 0, 1);
                    lsc_atomic_update<atomic_op::inc, uint32_t, 1>(p_sync_buffer+pass_over_sync_offset, simd<uint32_t, 1>(0), 1);
                }
            }
            // calculate each thread starting offset
            barrier();
            {
                #pragma unroll
                for (uint32_t s = 0; s<BIN_COUNT; s+=64) {
                    bin_offset.template select<64, 1>(s).template bit_cast_view<uint32_t>() = lsc_slm_block_load<uint32_t, 64>(slm_incoming_start + s*sizeof(hist_t));
                }
                if (local_tid>0) {
                    #pragma unroll
                    for (uint32_t s = 0; s<BIN_COUNT; s+=64) {
                        simd<hist_t, 64> group_local_sum;
                        group_local_sum.template bit_cast_view<uint32_t>() = lsc_slm_block_load<uint32_t, 64>(slm_bin_hist_start + (local_tid-1)*HIST_STRIDE + s*sizeof(hist_t));
                        bin_offset.template select<64, 1>(s) += group_local_sum;
                    }
                }
                if ((local_tid > THREAD_PER_BIN_GROUP) && (local_tid % BIN_HEIGHT != 0)) {
                    uint32_t prev_cum_rowid = (local_tid-1) / BIN_HEIGHT * BIN_HEIGHT - 1;
                    #pragma unroll
                    for (uint32_t s = 0; s<BIN_COUNT; s+=64) {
                        simd<hist_t, 64> group_sum;
                        group_sum.template bit_cast_view<uint32_t>() = lsc_slm_block_load<uint32_t, 64>(slm_bin_hist_start + prev_cum_rowid*HIST_STRIDE + s*sizeof(hist_t));
                        bin_offset.template select<64, 1>(s) += group_sum;
                    }
                }
            }
        }
    }
    {
        constexpr uint32_t BLOCK_SIZE = 32;
        simd<device_addr_t, BLOCK_SIZE> write_addr;
        simd<KeyT, BLOCK_SIZE> keys;
        simd<bin_t, BLOCK_SIZE> bins;
        // auto p_read = in + io_offset;
        if (io_offset < n) {
            for (uint32_t i=0; i<PROCESS_SIZE; i+=BLOCK_SIZE) {
                simd<device_addr_t, BLOCK_SIZE> lane_id_block(0, 1);
                simd_mask<BLOCK_SIZE> m = (io_offset+lane_id_block+i)<n;
                simd<KeyT, BLOCK_SIZE> source = lsc_gather<KeyT, 1,
                        // lsc_data_size::default_size, cache_hint::cached, cache_hint::cached, BLOCK_SIZE>(p_read+i, lane_id_block*uint32_t(sizeof(KeyT)), m);
                        lsc_data_size::default_size, cache_hint::cached, cache_hint::cached, BLOCK_SIZE>(in, (io_offset + i + lane_id_block) * uint32_t(sizeof(KeyT)), m);
                keys = merge(source, simd<KeyT, BLOCK_SIZE>(-1), m);

                bins = (keys >> (stage * RADIX_BITS)) & MASK;
                #pragma unroll
                for (uint32_t s = 0; s<BLOCK_SIZE; s+=1) {
                    write_addr[s] = bin_offset[bins[s]];
                    bin_offset[bins[s]] += 1;
                }
                lsc_scatter<KeyT, 1, lsc_data_size::default_size, cache_hint::write_back, cache_hint::write_back, BLOCK_SIZE>(
                    out,
                    write_addr*sizeof(KeyT),
                    keys, write_addr<n);
            }
        }
    }
}

/* TODO:
    Make sure the data is in rng after the last stage
    Generate unique kernel names
*/
template <typename _Range, typename KeyT, uint32_t RADIX_BITS>
void onesweep(sycl::queue &q, _Range&& rng, size_t n) {
    using namespace sycl;
    using namespace __ESIMD_NS;
    using global_hist_t = uint32_t;

    const uint32_t PROCESS_SIZE = 512;
    constexpr uint32_t BINCOUNT = 1 << RADIX_BITS;
    const uint32_t HW_TG_COUNT = 64;
    constexpr uint32_t THREAD_PER_TG = 64;
    uint32_t SWEEP_PROCESSING_SIZE = PROCESS_SIZE;
    const uint32_t sweep_tg_count = div_up(n, THREAD_PER_TG*SWEEP_PROCESSING_SIZE);
    const uint32_t sweep_threads = sweep_tg_count * THREAD_PER_TG;
    constexpr uint32_t NBITS =  sizeof(KeyT) * 8;
    constexpr uint32_t STAGES = div_up(NBITS, RADIX_BITS);

    assert((SWEEP_PROCESSING_SIZE == 256) || (SWEEP_PROCESSING_SIZE == 512) || (SWEEP_PROCESSING_SIZE == 1024) || (SWEEP_PROCESSING_SIZE == 1536));

    constexpr uint32_t SYNC_SEGMENT_COUNT = 64;
    constexpr uint32_t SYNC_SEGMENT_SIZE_DW = 128;
    constexpr uint32_t SYNC_BUFFER_SIZE = SYNC_SEGMENT_COUNT * SYNC_SEGMENT_SIZE_DW * sizeof(uint32_t); //bytes
    constexpr uint32_t GLOBAL_OFFSET_SIZE = BINCOUNT * STAGES * sizeof(global_hist_t);
    size_t temp_buffer_size = GLOBAL_OFFSET_SIZE + // global offset
                              SYNC_BUFFER_SIZE;  // sync buffer

    uint8_t *tmp_buffer = reinterpret_cast<uint8_t*>(sycl::malloc_device(temp_buffer_size, q));
    auto p_global_offset = reinterpret_cast<uint32_t*>(tmp_buffer);
    auto p_sync_buffer = reinterpret_cast<uint32_t*>(tmp_buffer + GLOBAL_OFFSET_SIZE);
    auto p_output = sycl::malloc_device<KeyT>(n, q);

    auto acc = rng.__m_acc;
    using InT = decltype(acc);
    using OutT = decltype(p_output);
    {
        nd_range<1> Range( (range<1>( HW_TG_COUNT * THREAD_PER_TG)), (range<1>(THREAD_PER_TG)) );
        auto e = q.submit([&](handler &cgh) {
            oneapi::dpl::__ranges::__require_access(cgh, rng);
            cgh.parallel_for<class kernel_global_histogram>(
                    Range, [=](nd_item<1> idx) [[intel::sycl_explicit_simd]] {
                        global_histogram<InT, KeyT, RADIX_BITS, HW_TG_COUNT, THREAD_PER_TG> (idx, n, acc, p_global_offset, p_sync_buffer);
                    });
        });
    }
    {
        auto e = q.submit([&](handler &cgh) {
            cgh.parallel_for<class kernel_global_scan>(
                nd_range<1>({STAGES * BINCOUNT}, {BINCOUNT}), [=](nd_item<1> idx) {
                    uint32_t offset = idx.get_global_id(0);
                    auto g = idx.get_group();
                    uint32_t count = p_global_offset[offset];
                    uint32_t presum = sycl::exclusive_scan_over_group(g, count, sycl::plus<KeyT>());
                    p_global_offset[offset] = presum;
                });
            });
    }
    {
        sycl::event e;
        for (uint32_t stage = 0; stage < STAGES; stage++) {
            nd_range<1> Range( (range<1>( sweep_tg_count * THREAD_PER_TG)), (range<1>(THREAD_PER_TG)) );
            if (SWEEP_PROCESSING_SIZE == 256) {
                e = q.submit([&](handler &cgh) {
                    oneapi::dpl::__ranges::__require_access(cgh, rng);
                    if((stage & 1) == 0)
                    {
                        cgh.parallel_for<class kernel_radix_sort_onesweep_256_0>(
                                Range, [=](nd_item<1> idx) [[intel::sycl_explicit_simd]] {
                                        onesweep_kernel<InT, OutT, KeyT, RADIX_BITS, THREAD_PER_TG, 256> (idx, n, stage, acc, p_output, tmp_buffer);
                                });
                    }
                    else
                    {
                        cgh.parallel_for<class kernel_radix_sort_onesweep_256_1>(
                                Range, [=](nd_item<1> idx) [[intel::sycl_explicit_simd]] {
                                        onesweep_kernel<OutT, InT, KeyT, RADIX_BITS, THREAD_PER_TG, 256> (idx, n, stage, p_output, acc, tmp_buffer);
                                });
                    }
                });
            } else if (SWEEP_PROCESSING_SIZE == 512) {
                e = q.submit([&](handler &cgh) {
                    oneapi::dpl::__ranges::__require_access(cgh, rng);
                    if((stage & 1) == 0)
                    {
                        cgh.parallel_for<class kernel_radix_sort_onesweep_512_0>(
                                Range, [=](nd_item<1> idx) [[intel::sycl_explicit_simd]] {
                                        onesweep_kernel<InT, OutT, KeyT, RADIX_BITS, THREAD_PER_TG, 512> (idx, n, stage, acc, p_output, tmp_buffer);
                                });
                    }
                    else
                    {
                        cgh.parallel_for<class kernel_radix_sort_onesweep_512_1>(
                                Range, [=](nd_item<1> idx) [[intel::sycl_explicit_simd]] {
                                        onesweep_kernel<OutT, InT, KeyT, RADIX_BITS, THREAD_PER_TG, 512> (idx, n, stage, p_output, acc, tmp_buffer);
                                });
                    }
                });
            } else if (SWEEP_PROCESSING_SIZE == 1024) {
                e = q.submit([&](handler &cgh) {
                    oneapi::dpl::__ranges::__require_access(cgh, rng);
                    if((stage & 1) == 0)
                    {
                        cgh.parallel_for<class kernel_radix_sort_onesweep_1024_0>(
                                Range, [=](nd_item<1> idx) [[intel::sycl_explicit_simd]] {
                                    onesweep_kernel<InT, OutT, KeyT, RADIX_BITS, THREAD_PER_TG, 1024> (idx, n, stage, acc, p_output, tmp_buffer);
                                });
                    }
                    else
                    {
                        cgh.parallel_for<class kernel_radix_sort_onesweep_1024_1>(
                                Range, [=](nd_item<1> idx) [[intel::sycl_explicit_simd]] {
                                    onesweep_kernel<OutT, InT, KeyT, RADIX_BITS, THREAD_PER_TG, 1024> (idx, n, stage, p_output, acc, tmp_buffer);
                                });
                    }
                });
            } else if (SWEEP_PROCESSING_SIZE == 1536) {
                e = q.submit([&](handler &cgh) {
                    oneapi::dpl::__ranges::__require_access(cgh, rng);
                    if((stage & 1) == 0)
                    {
                        cgh.parallel_for<class kernel_radix_sort_onesweep_1536_0>(
                                Range, [=](nd_item<1> idx) [[intel::sycl_explicit_simd]] {
                                    onesweep_kernel<InT, OutT, KeyT, RADIX_BITS, THREAD_PER_TG, 1536> (idx, n, stage, acc, p_output, tmp_buffer);
                                });
                    }
                    else
                    {
                        cgh.parallel_for<class kernel_radix_sort_onesweep_1536_1>(
                                Range, [=](nd_item<1> idx) [[intel::sycl_explicit_simd]] {
                                    onesweep_kernel<OutT, InT, KeyT, RADIX_BITS, THREAD_PER_TG, 1536> (idx, n, stage, p_output, acc, tmp_buffer);
                                });
                    }
                });
            }
        }
        e.wait();
    }
    sycl::free(tmp_buffer, q);
    sycl::free(p_output, q);
}

} // namespace oneapi::dpl::experimental::esimd::impl


namespace oneapi::dpl::experimental::esimd
{
/*
    Interface:
        Provide a temporary buffer. May be not necessary for some implementations, e.g. one-work-group implementation.
            Provide a way to show required size of a temporary memory pool.
        Sort only specific range of bits?
            Add a tag for implementation with requiring forward progress
            Allow selection of specific implementation?
*/
/*
     limitations:
         eSIMD operations:
            gather: Element type; can only be a 1,2,4-byte integer, sycl::half or float.
            lsc_gather: limited supported platforms: see https://intel.github.io/llvm-docs/doxygen/group__sycl__esimd__memory__lsc.html#ga250b3c0085f39c236582352fb711aadb)
            one_workgroup:
                RadixBits = 7 or 8: see BIN_WIDTH and scan implementation
*/
// TODO: call it only for all_view (accessor) and guard_view (USM) ranges
template <typename Policy, typename _Range, bool IsAscending = true, std::uint16_t WorkGroupSize = 256,
          std::uint16_t ItemsPerWorkItem = 16, std::uint32_t RadixBits = 8>
void
radix_sort(Policy&& exec, _Range&& rng)
{
    auto q = exec.queue();
    using KeyT = oneapi::dpl::__internal::__value_t<_Range>;

    const ::std::size_t n = rng.size();
    if (n <= 16384)
    {
        oneapi::dpl::experimental::esimd::impl::one_workgroup<_Range, KeyT, RadixBits>(q, std::forward<_Range>(rng), n);
    }
    else if (n <= 262144)
    {
        oneapi::dpl::experimental::esimd::impl::cooperative<_Range, KeyT, RadixBits>(q, std::forward<_Range>(rng), n);
    }
    else
    {
        // TODO: pass processing size basing on input size
        oneapi::dpl::experimental::esimd::impl::onesweep<_Range, KeyT, RadixBits>(q, std::forward<_Range>(rng), n);
    }
}
} // namespace oneapi::dpl::experimental::esimd

#endif // _ONEDPL_parallel_backend_esimd_radix_sort_H