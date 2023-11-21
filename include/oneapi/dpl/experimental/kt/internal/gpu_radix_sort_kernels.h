// -*- C++ -*-
//===-- gpu_radix_sort_kernels.h --------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------------------------------------------------===//

#ifndef _ONEDPL_KT_GPU_RADIX_SORT_KERNELS_H
#define _ONEDPL_KT_GPU_RADIX_SORT_KERNELS_H

#if __has_include(<sycl/sycl.hpp>)
#    include <sycl/sycl.hpp>
#else
#    include <CL/sycl.hpp>
#endif

#include <cstdint>
#include "gpu_radix_sort_utils.h"
#include "../../../pstl/hetero/dpcpp/parallel_backend_sycl_utils.h"

namespace oneapi::dpl::experimental::kt::gpu::__impl
{

template <uint16_t __work_group_size, uint16_t __data_per_work_item, typename keyT, typename _GlobalOffsetData,
          ::std::uint8_t __radix_bits, ::std::uint32_t __stage_count,
          bool __is_ascending> //TODO: hook up __is_ascending
struct RadixSortHistogram
{
    static constexpr ::std::uint32_t __bin_count = 1 << __radix_bits;
    static constexpr ::std::uint32_t __histogram_elements = __bin_count * __stage_count;

    using atomic_local = sycl::atomic_ref<_GlobalOffsetData, sycl::memory_order::relaxed,
        sycl::memory_scope::device, sycl::access::address_space::local_space>;

    FORCE_INLINE RadixSortHistogram(_GlobalOffsetData * const digit_bins_histogram, _GlobalOffsetData * const shared_digit_histogram,
        keyT * const array, uint32_t num_keys)
        : digit_bins_histogram(digit_bins_histogram),
          array(array),
          num_keys_global(num_keys),
          shared_digit_histogram(shared_digit_histogram) {}

    FORCE_INLINE void accumulateFullTile(keyT *array) {
        keyT key[__data_per_work_item];
#pragma unroll
        for (uint32_t i = 0; i < __data_per_work_item; i++) {
            key[i] = array[__work_group_size * i];
        }

#pragma unroll
        for (uint32_t i = 0; i < __data_per_work_item; i++) {
#pragma unroll
            for (uint32_t d = 0; d < __stage_count; d++) {
                uint32_t digit = __utils::getDigit(oneapi::dpl::__par_backend_hetero::__order_preserving_cast<__is_ascending>(key[i]), d, __radix_bits);
                atomic_local(shared_digit_histogram[d * __bin_count + digit])++;
            }
        }
    }

    FORCE_INLINE void accumulatePartialTile(keyT *array, uint32_t eleCnt) {
        for (uint32_t i = 0; i < eleCnt; i++) {
            keyT key = array[i * __work_group_size];
#pragma unroll
            for (uint32_t d = 0; d < __stage_count; d++) {
                uint32_t digit = __utils::getDigit(oneapi::dpl::__par_backend_hetero::__order_preserving_cast<__is_ascending>(key), d, __radix_bits);
                atomic_local(shared_digit_histogram[d * __bin_count + digit])++;
            }
        }
    }

    FORCE_INLINE void accumulateSharedHistogram(keyT *array, uint32_t offset) {
        if (offset + __data_per_work_item * __work_group_size <= num_keys_global)
            accumulateFullTile(&array[offset]);
        else {
            uint32_t eleCnt = (num_keys_global - offset + __work_group_size - 1) / __work_group_size;
            accumulatePartialTile(&array[offset], eleCnt);
        }
    }

    FORCE_INLINE void accumulateGlobalHistogram(uint32_t local_id) {
#pragma unroll
        for (uint32_t idx = local_id; idx < __histogram_elements; idx += __work_group_size) {
            auto atomic_global_counter =
                sycl::atomic_ref<_GlobalOffsetData, sycl::memory_order::relaxed, sycl::memory_scope::device,
                    sycl::access::address_space::global_space>(digit_bins_histogram[idx]);
            atomic_global_counter += shared_digit_histogram[idx];
        }
    }

    FORCE_INLINE void process(sycl::nd_item<1> &id) {
        uint32_t localId = id.get_local_linear_id();
#pragma unroll
        for (uint32_t i = localId; i < __histogram_elements; i += __work_group_size) {
            shared_digit_histogram[i] = 0;
        }
        uint32_t offset = id.get_group_linear_id() * __data_per_work_item * __work_group_size + localId;
        __utils::slmBarrier(id);
        accumulateSharedHistogram(array, offset);
        __utils::slmBarrier(id);
        accumulateGlobalHistogram(localId);
    }

    _GlobalOffsetData * const digit_bins_histogram;

    uint32_t num_keys_global;

    _GlobalOffsetData * const shared_digit_histogram;

    keyT * const array;
};

//TODO: Currently unused. Do we want to use this scan or the version from esimd_radix_sort?
template <uint32_t NUM_BINS>
FORCE_INLINE void globalExclusiveScan(uint32_t *histogram, uint32_t *scan, sycl::nd_item<1> &id) {
    uint32_t offset = id.get_group_linear_id() * NUM_BINS;
    sycl::joint_exclusive_scan(id.get_group(), histogram + offset, histogram + offset + NUM_BINS,
        scan + offset, sycl::plus<uint32_t>());
}

template <uint32_t RADIX_DIGITS, uint32_t GROUP_WARPS, uint32_t ITEMS_PER_THREAD,
    uint32_t GROUP_THREADS, typename keyT>
struct OneSweepSharedData {
    union {
        uint16_t warp_offsets[GROUP_WARPS * RADIX_DIGITS + 1];
        uint16_t warp_counters[GROUP_WARPS * RADIX_DIGITS];
    };
    union {
        keyT shared_keys[GROUP_THREADS * ITEMS_PER_THREAD];
        uint16_t offset_buffer[GROUP_WARPS * 2];
    };
    uint32_t group_offsets[RADIX_DIGITS];
    uint32_t group_id;
};

template <uint32_t RADIX_BITS, uint32_t GROUP_THREADS, uint32_t ITEMS_PER_THREAD, bool USE_DYNAMIC_ID,
          uint32_t WARP_THREADS, typename keyT, typename _InKeysRng, typename _OutKeysRng, bool __is_ascending>
struct OneSweepRadixSort
{
    enum
    {
        RADIX_DIGITS = 1 << RADIX_BITS,
        NUM_DIGITS = sizeof(keyT) * 8 / RADIX_BITS,
        // seperate the threads into parts, each parts share a single counter.
        HISTOGRAM_ELEMENTS = RADIX_DIGITS * NUM_DIGITS,
        BINS_PER_THREAD = (RADIX_DIGITS + GROUP_THREADS - 1) / GROUP_THREADS,
        GROUP_WARPS = (GROUP_THREADS + WARP_THREADS - 1) / WARP_THREADS,
        WARP_ITEMS = ITEMS_PER_THREAD * WARP_THREADS,
        GROUP_ITEMS = ITEMS_PER_THREAD * GROUP_THREADS,
    };

    using Data = OneSweepSharedData<RADIX_DIGITS, GROUP_WARPS, ITEMS_PER_THREAD, GROUP_THREADS, keyT>;
    using atomic_global = sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
        sycl::memory_scope::device, sycl::access::address_space::global_space>;
    using Vector = __utils::Vector<ITEMS_PER_THREAD, keyT>;

    OneSweepRadixSort(uint32_t pass, _InKeysRng array_in, _OutKeysRng array_out, uint32_t *digit_offsets,
        uint32_t *global_offsets, uint32_t *dynamic_id_ptr, uint32_t size, Data &__lmem)
        : array_in(array_in),
          array_out(array_out),
          __lmem(__lmem),                            // in SLM
          digit_offsets(digit_offsets),    // the counts for fronter digits
          global_offsets(global_offsets),  // in ugm
          pass(pass),
          array_size(size),
          dynamic_id_ptr(dynamic_id_ptr) {}

    FORCE_INLINE void initializeSLM(sycl::nd_item<1> &id) {
        uint16_t j = 0;
        uint32_t warp_offset = warp * WARP_THREADS;
#pragma unroll
        for (uint32_t i = 0; i < GROUP_WARPS * RADIX_DIGITS; i += GROUP_THREADS) {
            id.get_sub_group().store(
                sycl::local_ptr<uint16_t>(&__lmem.warp_counters[i + warp_offset]), j);
        }
        if (id.get_group().leader()) {
            __lmem.warp_offsets[GROUP_WARPS * RADIX_DIGITS] = ITEMS_PER_THREAD * GROUP_THREADS;
        }
    }

    FORCE_INLINE void generateDynamicGroupId(sycl::nd_item<1> &id) {
        if (USE_DYNAMIC_ID) {
            if (id.get_group().leader()) {
                __lmem.group_id = atomic_global(dynamic_id_ptr[0])++;
            }
            __utils::slmBarrier(id);
            group_id = __lmem.group_id;
        } else {
            group_id = id.get_group_linear_id();
            __utils::slmBarrier(id);
        }
    }

    FORCE_INLINE void rankSharedKeysMatchAny(Vector &keys, uint16_t *ranks, sycl::nd_item<1> &id) {
        uint16_t *warp_counters = &__lmem.warp_counters[warp];
#pragma unroll
        for (uint32_t i = 0; i < ITEMS_PER_THREAD; i++) {
            uint32_t digit = __utils::getDigit(keys[i], pass, RADIX_BITS);
            uint32_t bin = digit * GROUP_WARPS;
            uint32_t bin_mask = __utils::matchAny<RADIX_BITS>(id, digit);
            bool last = (bin_mask >> (lane + 1)) == 0;

            bin_mask <<= (31 - lane);
            uint16_t popc = (uint16_t)(sycl::popcount(bin_mask));
            uint16_t warp_counter = warp_counters[bin];
            ranks[i] = warp_counter + popc - 1;
            if (last) {
                warp_counters[bin] += popc;
            }
        }
        __utils::slmBarrier(id);
    }

    FORCE_INLINE void scanCounters(sycl::nd_item<1> &id) {
        uint16_t sum = 0, i = id.get_local_linear_id(),
                 counters_offset = i * RADIX_DIGITS / WARP_THREADS;
#pragma unroll
        for (uint16_t j = 0; j < RADIX_DIGITS / WARP_THREADS; j++)
            sum += __lmem.warp_counters[j + counters_offset];
        uint16_t offset = sycl::reduce_over_group(id.get_sub_group(), sum, sycl::plus<uint16_t>());
        if(lane==0)
            __lmem.offset_buffer[warp] = offset;

        __utils::slmBarrier(id);

        if (warp == 0) {
            __lmem.offset_buffer[GROUP_WARPS] = 0;
#pragma unroll
            for (uint32_t i = 0; i < GROUP_WARPS; i += WARP_THREADS) {
                uint16_t inclusive_sum =
                    id.get_sub_group().load(sycl::local_ptr<uint16_t>(&__lmem.offset_buffer[i]));
                inclusive_sum = __lmem.offset_buffer[GROUP_WARPS + i] +
                    sycl::inclusive_scan_over_group(
                        id.get_sub_group(), inclusive_sum, sycl::plus<uint16_t>());

                id.get_sub_group().store(
                    sycl::local_ptr<uint16_t>(&__lmem.offset_buffer[GROUP_WARPS + i + 1]),
                    inclusive_sum);
            }
        }
        __utils::slmBarrier(id);

        sum = sycl::exclusive_scan_over_group(id.get_sub_group(), sum, sycl::plus<uint16_t>());
        sum += __lmem.offset_buffer[GROUP_WARPS + warp];
#pragma unroll
        for (uint16_t j = 0; j < RADIX_DIGITS / WARP_THREADS; j++) {
            uint16_t tmp = __lmem.warp_counters[j + counters_offset];
            __lmem.warp_offsets[j + counters_offset] = sum;
            sum += tmp;
        }
        __utils::slmBarrier(id);
    }

    FORCE_INLINE void computeSharedKeysOffset(Vector &keys, uint16_t *ranks, sycl::nd_item<1> &id) {
        rankSharedKeysMatchAny(keys, ranks, id);

        scanCounters(id);

#pragma unroll
        for (int i = 0; i < ITEMS_PER_THREAD; i++) {
            uint32_t digit = __utils::getDigit(keys[i], pass, RADIX_BITS);
            uint32_t bin = digit * GROUP_WARPS + warp;
            ranks[i] += __lmem.warp_offsets[bin];
        }
    }

    FORCE_INLINE uint32_t getGlobalBin(sycl::nd_item<1> &id, uint32_t u, uint32_t before) {
        return BINS_PER_THREAD * id.get_local_linear_id() + (group_id - before) * RADIX_DIGITS + u;
    }

    FORCE_INLINE uint32_t getLocalBin(sycl::nd_item<1> &id, uint32_t u) {
        return BINS_PER_THREAD * id.get_local_linear_id() + u;
    }

    FORCE_INLINE void computePartialOffsets(sycl::nd_item<1> &id) {
#pragma unroll
        for (uint32_t i = 0; i < BINS_PER_THREAD; i++) {
            uint32_t bin = getLocalBin(id, i), bin_global = getGlobalBin(id, i, 0);
            if (bin < RADIX_DIGITS) {
                uint32_t offset_digit =
                    __lmem.warp_offsets[(bin + 1) * GROUP_WARPS] - __lmem.warp_offsets[bin * GROUP_WARPS];

                __lmem.group_offsets[bin] = offset_digit;

                auto offset_atomic = atomic_global(global_offsets[bin_global]);
                offset_atomic = (offset_digit | 0x40000000);
            }
        }
    }

    FORCE_INLINE void lookBackGlobalOffsets(sycl::nd_item<1> &id) {
#pragma unroll
        for (uint32_t i = 0; i < BINS_PER_THREAD; i++) {
            uint32_t global_bin = getGlobalBin(id, i, 0), bin = getLocalBin(id, i);
            if (bin < RADIX_DIGITS) {
                uint32_t global_offset = 0;
                for (uint32_t j = 1; j <= group_id; j++) {
                    uint32_t bin_before = getGlobalBin(id, i, j), offset_before = 0;
                    do {
                        auto offset_atomic = atomic_global(global_offsets[bin_before]);
                        offset_before = offset_atomic.load();
                    } while ((offset_before & 0x40000000) == 0);
                    global_offset += (offset_before & 0x3FFFFFFF);

                    if ((offset_before & 0x80000000) != 0) break;
                }
                global_offsets[global_bin] = (global_offset + __lmem.group_offsets[bin]) | 0xC0000000;

                __lmem.group_offsets[bin] = global_offset + digit_offsets[bin] -
                    (uint32_t)__lmem.warp_offsets[bin * GROUP_WARPS];
            }
        }
    }

    FORCE_INLINE void scatterSharedKeys(uint16_t *rank, Vector& keys, sycl::nd_item<1> &id) {
        uint32_t lane_offset = warp * WARP_ITEMS + lane;
#pragma unroll
        for (uint32_t i = 0; i < ITEMS_PER_THREAD; i++) {
            __lmem.shared_keys[rank[i]] = keys[i];
        }
    }

    FORCE_INLINE void scatterGlobalKeys(sycl::nd_item<1> &id) {
        if (group_limit >= GROUP_ITEMS) {
#pragma unroll
            for (uint32_t i = 0; i < ITEMS_PER_THREAD; i++) {
                uint32_t idx = id.get_local_linear_id() + i * GROUP_THREADS;
                keyT key = __lmem.shared_keys[idx];
                uint32_t digit = __utils::getDigit(key, pass, RADIX_BITS);

                array_out[__lmem.group_offsets[digit] + idx] = key;
            }
        } else {
#pragma unroll
            for (uint32_t i = 0; i < ITEMS_PER_THREAD; i++) {
                uint32_t idx = id.get_local_linear_id() + i * GROUP_THREADS;
                keyT key = __lmem.shared_keys[idx];
                if (key != 0xFFFFFFFF) { //TODO: replace magic no. with range check
                    uint32_t digit = __utils::getDigit(key, pass, RADIX_BITS);

                    array_out[__lmem.group_offsets[digit] + idx] = key;
                } else {
                    break;
                }
            }
        }
    }

    FORCE_INLINE void loadKeys(Vector &keys, sycl::nd_item<1> &id) {
        uint32_t warp_offset = warp * WARP_ITEMS;
        if (group_limit >= warp_offset + WARP_ITEMS) {
            keys.load(id.get_sub_group(), array_in + warp_offset);
        } else {
            warp_offset += lane;
#pragma unroll
            for (uint32_t i = 0; i < ITEMS_PER_THREAD; i++) {
                uint32_t idx = warp_offset + i * WARP_THREADS;
                if (idx < group_limit)
                    keys[i] = array_in[idx];
                else
                    keys[i] = 0xFFFFFFFF;
            }
        }
    }

    FORCE_INLINE void process(sycl::nd_item<1> &id) {
        lane = id.get_sub_group().get_local_linear_id();
        warp = id.get_sub_group().get_group_linear_id();

        initializeSLM(id);
        generateDynamicGroupId(id);

        array_in = array_in + GROUP_ITEMS * group_id;
        group_limit = array_size - GROUP_ITEMS * group_id;

        uint16_t ranks[ITEMS_PER_THREAD];
        Vector keys;

        loadKeys(keys, id);
        computeSharedKeysOffset(keys, ranks, id);

        computePartialOffsets(id);

        lookBackGlobalOffsets(id);

        scatterSharedKeys(ranks, keys, id);

        __utils::slmBarrier(id);

        scatterGlobalKeys(id);
    }

    _InKeysRng array_in;
    _OutKeysRng array_out;
    uint32_t *digit_offsets, *global_offsets, pass, array_size, group_limit, *dynamic_id_ptr,
        group_id, warp, lane;
    Data& __lmem;
};

} // namespace oneapi::dpl::experimental::kt::gpu::__impl

#endif // _ONEDPL_KT_GPU_RADIX_SORT_KERNELS_H
