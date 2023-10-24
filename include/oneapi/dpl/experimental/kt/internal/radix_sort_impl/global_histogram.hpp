#pragma once
#include <CL/sycl.hpp>
#include <cstdint>

#include "utils.hpp"

template <uint32_t GROUP_THREAD = 16, uint32_t ITEMS_PER_THREAD = 32, typename keyT = uint32_t,
    bool IS_DESC = false>
struct RadixSortHistogram {
    enum {
        RADIX_BITS = 8,
        RADIX_DIGITS = 1 << RADIX_BITS,
        NUM_DIGITS = sizeof(keyT) * 8 / RADIX_BITS,
        HISTOGRAM_ELEMENTS = RADIX_DIGITS * NUM_DIGITS,
    };

    using atomic_local = sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
        sycl::memory_scope::device, sycl::access::address_space::local_space>;

    FORCE_INLINE RadixSortHistogram(uint32_t *digit_bins_histogram, uint32_t *shared_digit_histogram,
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
            for (uint32_t d = 0; d < NUM_DIGITS; d++) {
                uint32_t digit = getDigit(key[i], d, RADIX_BITS);
                atomic_local(shared_digit_histogram[d * RADIX_DIGITS + digit])++;
            }
        }
    }

    FORCE_INLINE void accumulatePartialTile(keyT *array, uint32_t eleCnt) {
        for (uint32_t i = 0; i < eleCnt; i++) {
            keyT key = array[i * GROUP_THREAD];
#pragma unroll
            for (uint32_t d = 0; d < NUM_DIGITS; d++) {
                uint32_t digit = getDigit(key, d, RADIX_BITS);
                atomic_local(shared_digit_histogram[d * RADIX_DIGITS + digit])++;
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
        for (uint32_t idx = local_id; idx < HISTOGRAM_ELEMENTS; idx += GROUP_THREAD) {
            auto atomic_global_counter =
                sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                    sycl::access::address_space::global_space>(digit_bins_histogram[idx]);
            atomic_global_counter += shared_digit_histogram[idx];
        }
    }

    FORCE_INLINE void process(sycl::nd_item<1> &id) {
        uint32_t localId = id.get_local_linear_id();
#pragma unroll
        for (uint32_t i = localId; i < RADIX_DIGITS * NUM_DIGITS; i += GROUP_THREAD) {
            shared_digit_histogram[i] = 0;
        }
        uint32_t offset = id.get_group_linear_id() * ITEMS_PER_THREAD * GROUP_THREAD + localId;
        slmBarrier(id);
        accumulateSharedHistogram(array, offset);
        slmBarrier(id);
        accumulateGlobalHistogram(localId);
    }

    uint32_t *digit_bins_histogram;

    uint32_t num_keys_global;

    uint32_t *shared_digit_histogram;

    keyT *array;
};

FORCE_INLINE void refHistogram(uint32_t *array, uint32_t num_keys, uint32_t *histogram) {
    memset(histogram, 0, 1024 * sizeof(uint32_t));
    for (uint32_t i = 0; i < num_keys; i++) {
        uint32_t key = array[i];
        for (uint32_t lane = 0; lane < 4; lane++) {
            histogram[lane * 256 + getDigit(key, lane, 8)]++;
        }
    }
}

template <uint32_t NUM_BINS>
FORCE_INLINE void globalExclusiveScan(uint32_t *histogram, uint32_t *scan, sycl::nd_item<1> &id) {
    uint32_t offset = id.get_group_linear_id() * NUM_BINS;
    sycl::joint_exclusive_scan(id.get_group(), histogram + offset, histogram + offset + NUM_BINS,
        scan + offset, sycl::plus<uint32_t>());
}