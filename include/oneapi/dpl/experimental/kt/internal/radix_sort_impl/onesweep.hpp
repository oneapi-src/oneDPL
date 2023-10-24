#pragma once
#include <CL/sycl.hpp>
#include <cstdint>
#include <unordered_set>

#include "utils.hpp"

template <uint32_t RADIX_DIGITS, uint32_t GROUP_WARPS, uint32_t ITEMS_PER_THREAD,
    uint32_t GROUP_THREADS, typename keyT = uint32_t>
struct SharedData {
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

template <uint32_t RADIX_BITS = 8, uint32_t GROUP_THREADS = 256, uint32_t ITEMS_PER_THREAD = 32,
    bool USE_DYNAMIC_ID = false, uint32_t WARP_THREADS = 16, typename keyT = uint32_t,
    bool IS_DESC = false>
struct OneSweepRadixSort {
    enum {
        RADIX_DIGITS = 1 << RADIX_BITS,
        NUM_DIGITS = sizeof(keyT) * 8 / RADIX_BITS,
        // seperate the threads into parts, each parts share a single counter.
        HISTOGRAM_ELEMENTS = RADIX_DIGITS * NUM_DIGITS,
        BINS_PER_THREAD = (RADIX_DIGITS + GROUP_THREADS - 1) / GROUP_THREADS,
        GROUP_WARPS = (GROUP_THREADS + WARP_THREADS - 1) / WARP_THREADS,
        WARP_ITEMS = ITEMS_PER_THREAD * WARP_THREADS,
        GROUP_ITEMS = ITEMS_PER_THREAD * GROUP_THREADS,
    };

    using Data = SharedData<RADIX_DIGITS, GROUP_WARPS, ITEMS_PER_THREAD, GROUP_THREADS, keyT>;
    using atomic_global = sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
        sycl::memory_scope::device, sycl::access::address_space::global_space>;
    using Vector = Vector<ITEMS_PER_THREAD, keyT>;

    OneSweepRadixSort(uint32_t pass, keyT *array_in, keyT *array_out, uint32_t *digit_offsets,
        uint32_t *global_offsets, uint32_t *dynamic_id_ptr, uint32_t size, Data &s)
        : array_in(array_in),
          array_out(array_out),
          s(s),                            // in SLM
          digit_offsets(digit_offsets),    // the counts for fronter digits
          global_offsets(global_offsets),  // in ugm
          pass(pass),
          array_size(size),
          dynamic_id_ptr(dynamic_id_ptr) {}

    FORCE_INLINE void initialilzeSLM(sycl::nd_item<1> &id) {
        uint16_t j = 0;
        uint32_t warp_offset = warp * WARP_THREADS;
#pragma unroll
        for (uint32_t i = 0; i < GROUP_WARPS * RADIX_DIGITS; i += GROUP_THREADS) {
            id.get_sub_group().store(
                sycl::local_ptr<uint16_t>(&s.warp_counters[i + warp_offset]), j);
        }
        if (id.get_group().leader()) {
            s.warp_offsets[GROUP_WARPS * RADIX_DIGITS] = ITEMS_PER_THREAD * GROUP_THREADS;
        }
    }

    FORCE_INLINE void generateDynamicGroupId(sycl::nd_item<1> &id) {
        if (USE_DYNAMIC_ID) {
            if (id.get_group().leader()) {
                s.group_id = atomic_global(dynamic_id_ptr[0])++;
            }
            slmBarrier(id);
            group_id = s.group_id;
        } else {
            group_id = id.get_group_linear_id();
            slmBarrier(id);
        }
    }

    FORCE_INLINE void rankSharedKeysMatchAny(Vector &keys, uint16_t *ranks, sycl::nd_item<1> &id) {
        uint16_t *warp_counters = &s.warp_counters[warp];
#pragma unroll
        for (uint32_t i = 0; i < ITEMS_PER_THREAD; i++) {
            uint32_t digit = getDigit(keys[i], pass, RADIX_BITS);
            uint32_t bin = digit * GROUP_WARPS;
            uint32_t bin_mask = matchAny<RADIX_BITS>(id, digit);
            bool last = (bin_mask >> (lane + 1)) == 0;

            bin_mask <<= (31 - lane);
            uint16_t popc = (uint16_t)(sycl::popcount(bin_mask));
            uint16_t warp_counter = warp_counters[bin];
            ranks[i] = warp_counter + popc - 1;
            if (last) {
                warp_counters[bin] += popc;
            }
        }
        slmBarrier(id);
    }

    FORCE_INLINE void scanCounters(sycl::nd_item<1> &id) {
        uint16_t sum = 0, i = id.get_local_linear_id(),
                 counters_offset = i * RADIX_DIGITS / WARP_THREADS;
#pragma unroll
        for (uint16_t j = 0; j < RADIX_DIGITS / WARP_THREADS; j++)
            sum += s.warp_counters[j + counters_offset];
        uint16_t offset = sycl::reduce_over_group(id.get_sub_group(), sum, sycl::plus<uint16_t>());
        if(lane==0)
            s.offset_buffer[warp] = offset;

        slmBarrier(id);

        if (warp == 0) {
            s.offset_buffer[GROUP_WARPS] = 0;
#pragma unroll
            for (uint32_t i = 0; i < GROUP_WARPS; i += WARP_THREADS) {
                uint16_t inclusive_sum =
                    id.get_sub_group().load(sycl::local_ptr<uint16_t>(&s.offset_buffer[i]));
                inclusive_sum = s.offset_buffer[GROUP_WARPS + i] +
                    sycl::inclusive_scan_over_group(
                        id.get_sub_group(), inclusive_sum, sycl::plus<uint16_t>());

                id.get_sub_group().store(
                    sycl::local_ptr<uint16_t>(&s.offset_buffer[GROUP_WARPS + i + 1]),
                    inclusive_sum);
            }
        }
        slmBarrier(id);

        sum = sycl::exclusive_scan_over_group(id.get_sub_group(), sum, sycl::plus<uint16_t>());
        sum += s.offset_buffer[GROUP_WARPS + warp];
#pragma unroll
        for (uint16_t j = 0; j < RADIX_DIGITS / WARP_THREADS; j++) {
            uint16_t tmp = s.warp_counters[j + counters_offset];
            s.warp_offsets[j + counters_offset] = sum;
            sum += tmp;
        }
        slmBarrier(id);
    }

    FORCE_INLINE void computeSharedKeysOffset(Vector &keys, uint16_t *ranks, sycl::nd_item<1> &id) {
        rankSharedKeysMatchAny(keys, ranks, id);

        scanCounters(id);

#pragma unroll
        for (int i = 0; i < ITEMS_PER_THREAD; i++) {
            uint32_t digit = getDigit(keys[i], pass, RADIX_BITS);
            uint32_t bin = digit * GROUP_WARPS + warp;
            ranks[i] += s.warp_offsets[bin];
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
                    s.warp_offsets[(bin + 1) * GROUP_WARPS] - s.warp_offsets[bin * GROUP_WARPS];

                s.group_offsets[bin] = offset_digit;

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
                global_offsets[global_bin] = (global_offset + s.group_offsets[bin]) | 0xC0000000;

                s.group_offsets[bin] = global_offset + digit_offsets[bin] -
                    (uint32_t)s.warp_offsets[bin * GROUP_WARPS];
            }
        }
    }

    FORCE_INLINE void scatterSharedKeys(uint16_t *rank, Vector& keys, sycl::nd_item<1> &id) {
        uint32_t lane_offset = warp * WARP_ITEMS + lane;
#pragma unroll
        for (uint32_t i = 0; i < ITEMS_PER_THREAD; i++) {
            s.shared_keys[rank[i]] = keys[i];
        }
    }

    FORCE_INLINE void scatterGlobalKeys(sycl::nd_item<1> &id) {
        if (group_limit >= GROUP_ITEMS) {
#pragma unroll
            for (uint32_t i = 0; i < ITEMS_PER_THREAD; i++) {
                uint32_t idx = id.get_local_linear_id() + i * GROUP_THREADS;
                keyT key = s.shared_keys[idx];
                uint32_t digit = getDigit(key, pass, RADIX_BITS);

                array_out[s.group_offsets[digit] + idx] = key;
            }
        } else {
#pragma unroll
            for (uint32_t i = 0; i < ITEMS_PER_THREAD; i++) {
                uint32_t idx = id.get_local_linear_id() + i * GROUP_THREADS;
                keyT key = s.shared_keys[idx];
                if (key != 0xFFFFFFFF) { //TODO replace magic no. with range check?
                    uint32_t digit = getDigit(key, pass, RADIX_BITS);

                    array_out[s.group_offsets[digit] + idx] = key;
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

        initialilzeSLM(id);
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

        slmBarrier(id);

        scatterGlobalKeys(id);
    }

    keyT *array_in, *array_out;
    uint32_t *digit_offsets, *global_offsets, pass, array_size, group_limit, *dynamic_id_ptr,
        group_id, warp, lane;
    Data &s;
};
