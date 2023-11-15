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

namespace oneapi::dpl::experimental::kt::gpu::__impl
{

template <uint16_t GROUP_THREAD, uint16_t ITEMS_PER_THREAD, typename keyT, typename _GlobalOffsetData,
          ::std::uint8_t __radix_bits, ::std::uint32_t __stage_count,
          bool __is_ascending> //TODO: hook up __is_ascending
struct RadixSortHistogram
{
    static constexpr ::std::uint32_t __bin_count = 1 << __radix_bits;
    static constexpr ::std::uint32_t __histogram_elements = __bin_count * __stage_count;

    using atomic_local = sycl::atomic_ref<_GlobalOffsetData, sycl::memory_order::relaxed,
        sycl::memory_scope::device, sycl::access::address_space::local_space>;

    FORCE_INLINE RadixSortHistogram(_GlobalOffsetData *digit_bins_histogram, _GlobalOffsetData *shared_digit_histogram,
        keyT *array, uint32_t num_keys)
        : digit_bins_histogram(digit_bins_histogram),
          array(array),
          num_keys_global(num_keys),
          shared_digit_histogram(shared_digit_histogram) {}

    FORCE_INLINE void accumulateFullTile(keyT *array) {
        keyT key[ITEMS_PER_THREAD];
#pragma unroll
        for (uint32_t i = 0; i < ITEMS_PER_THREAD; i++) {
            key[i] = array[GROUP_THREAD * i];
        }

#pragma unroll
        for (uint32_t i = 0; i < ITEMS_PER_THREAD; i++) {
#pragma unroll
            for (uint32_t d = 0; d < __stage_count; d++) {
                uint32_t digit = getDigit(key[i], d, __radix_bits);
                atomic_local(shared_digit_histogram[d * __bin_count + digit])++;
            }
        }
    }

    FORCE_INLINE void accumulatePartialTile(keyT *array, uint32_t eleCnt) {
        for (uint32_t i = 0; i < eleCnt; i++) {
            keyT key = array[i * GROUP_THREAD];
#pragma unroll
            for (uint32_t d = 0; d < __stage_count; d++) {
                uint32_t digit = getDigit(key, d, __radix_bits);
                atomic_local(shared_digit_histogram[d * __bin_count + digit])++;
            }
        }
    }

    FORCE_INLINE void accumulateSharedHistogram(keyT *array, uint32_t offset) {
        if (offset + ITEMS_PER_THREAD * GROUP_THREAD <= num_keys_global)
            accumulateFullTile(&array[offset]);
        else {
            uint32_t eleCnt = (num_keys_global - offset + GROUP_THREAD - 1) / GROUP_THREAD;
            accumulatePartialTile(&array[offset], eleCnt);
        }
    }

    FORCE_INLINE void accumulateGlobalHistogram(uint32_t local_id) {
#pragma unroll
        for (uint32_t idx = local_id; idx < __histogram_elements; idx += GROUP_THREAD) {
            auto atomic_global_counter =
                sycl::atomic_ref<_GlobalOffsetData, sycl::memory_order::relaxed, sycl::memory_scope::device,
                    sycl::access::address_space::global_space>(digit_bins_histogram[idx]);
            atomic_global_counter += shared_digit_histogram[idx];
        }
    }

    FORCE_INLINE void process(sycl::nd_item<1> &id) {
        uint32_t localId = id.get_local_linear_id();
#pragma unroll
        for (uint32_t i = localId; i < __histogram_elements; i += GROUP_THREAD) {
            shared_digit_histogram[i] = 0;
        }
        uint32_t offset = id.get_group_linear_id() * ITEMS_PER_THREAD * GROUP_THREAD + localId;
        slmBarrier(id);
        accumulateSharedHistogram(array, offset);
        slmBarrier(id);
        accumulateGlobalHistogram(localId);
    }

    _GlobalOffsetData *digit_bins_histogram;

    uint32_t num_keys_global;

    _GlobalOffsetData *shared_digit_histogram;

    keyT *array;
};

//TODO: Currently unused. Do we want to use this scan or the version from esimd_radix_sort?
template <uint32_t NUM_BINS>
FORCE_INLINE void globalExclusiveScan(uint32_t *histogram, uint32_t *scan, sycl::nd_item<1> &id) {
    uint32_t offset = id.get_group_linear_id() * NUM_BINS;
    sycl::joint_exclusive_scan(id.get_group(), histogram + offset, histogram + offset + NUM_BINS,
        scan + offset, sycl::plus<uint32_t>());
}

template <bool __is_ascending, ::std::uint8_t __radix_bits, ::std::uint16_t __data_per_work_item,
          ::std::uint16_t __work_group_size, typename _KeyT, typename _InputT, typename _OutputT>
struct __radix_sort_onesweep_slm_reorder_kernel;

} // namespace oneapi::dpl::experimental::kt::gpu::__impl

#endif // _ONEDPL_KT_GPU_RADIX_SORT_KERNELS_H
