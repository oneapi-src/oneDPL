#pragma once

#include <CL/sycl.hpp>
#include <cstdint>

#include "global_histogram.hpp"
#include "onesweep.hpp"
#include "utils.hpp"

template <uint32_t RADIX_BITS = 8, uint32_t GROUP_THREADS = 256, uint32_t ITEMS_PER_THREADS = 32,
    bool USE_DYNAMIC_ID = true, uint32_t WARP_THREADS = 16, typename keyT = uint32_t,
    bool IS_DESC = false>
struct RadixSortKernel_Impl {
    enum {
        RADIX_DIGITS = 1 << RADIX_BITS,
        NUM_DIGITS = (sizeof(keyT) * 8 + RADIX_BITS - 1) / RADIX_BITS,
        GROUP_WARPS = (GROUP_THREADS + WARP_THREADS - 1) / WARP_THREADS,
        GROUP_ITEMS = GROUP_THREADS * ITEMS_PER_THREADS,
        HISTOGRAM_ITEMS_PER_THREADS = 32,
        HISTOGRAM_GROUP_THREADS = 1024,
        HISTOGRAM_GROUP_ITEMS = HISTOGRAM_GROUP_THREADS * HISTOGRAM_ITEMS_PER_THREADS,
    };
    using Data = SharedData<RADIX_DIGITS, GROUP_WARPS, ITEMS_PER_THREADS, GROUP_THREADS>;
    using OneSweepKernel = OneSweepRadixSort<RADIX_BITS, GROUP_THREADS, ITEMS_PER_THREADS,
        USE_DYNAMIC_ID, WARP_THREADS>;

    RadixSortKernel_Impl(keyT *array, uint32_t size) : _array(array), _size(size) {}

    void process(sycl::queue &q) {
        uint32_t *array = _array, size = _size, groups = (size + GROUP_ITEMS - 1) / GROUP_ITEMS,
                 histogram_groups = (size + HISTOGRAM_GROUP_ITEMS - 1) / HISTOGRAM_GROUP_ITEMS;

        uint32_t *storage = sycl::malloc_device<uint32_t>(
            NUM_DIGITS * RADIX_DIGITS * 2 + NUM_DIGITS + groups * RADIX_DIGITS * NUM_DIGITS + size,
            q);
        uint32_t *dynamic_id = storage, *digits_histograms = &dynamic_id[NUM_DIGITS],
                 *digits_offsets = &digits_histograms[NUM_DIGITS * RADIX_DIGITS],
                 *global_offsets = &digits_offsets[NUM_DIGITS * RADIX_DIGITS],
                 *array_out = &global_offsets[groups * RADIX_DIGITS * NUM_DIGITS],
                 *buffer[2]{array_out, array};
        q.parallel_for<>(2 + groups, [=](sycl::item<1> id) {
#pragma unroll
            for (uint32_t i = 0; i < NUM_DIGITS * RADIX_DIGITS + 1; i++) {
                storage[i * (2 + groups) + id.get_linear_id()] = 0;
            }
        });

        q.submit([&](sycl::handler &h) {
            sycl::local_accessor<uint32_t> shared_histogram(NUM_DIGITS * RADIX_DIGITS, h);
            h.parallel_for(sycl::nd_range<1>(
                               histogram_groups * HISTOGRAM_GROUP_THREADS, HISTOGRAM_GROUP_THREADS),
                [=](sycl::nd_item<1> id) [[intel::reqd_sub_group_size(WARP_THREADS)]] {
                    RadixSortHistogram<HISTOGRAM_GROUP_THREADS, HISTOGRAM_ITEMS_PER_THREADS> kernel(
                        digits_histograms, shared_histogram.get_pointer(), array, size);
                    kernel.process(id);
                });
        });
        q.submit([&](sycl::handler &h) {
            h.parallel_for(sycl::nd_range<1>(RADIX_DIGITS * NUM_DIGITS, RADIX_DIGITS),
                [=](sycl::nd_item<1> id) [[intel::reqd_sub_group_size(WARP_THREADS)]] {
                    globalExclusiveScan<RADIX_DIGITS>(digits_histograms, digits_offsets, id);
                });
        });

        for (int pass = 0; pass < NUM_DIGITS; pass++) {
            uint32_t *array_in = buffer[1 - (pass & 1)], *array_out = buffer[pass & 1];
            q.submit([&](sycl::handler &h) {
                auto s = sycl::local_accessor<Data>(1, h);

                h.parallel_for(sycl::nd_range<1>(groups * GROUP_THREADS, GROUP_THREADS),
                    [=](sycl::nd_item<1> id) [[intel::reqd_sub_group_size(WARP_THREADS)]] {
                        OneSweepKernel kernel(pass, array_in, array_out,
                            &digits_offsets[RADIX_DIGITS * pass],
                            &global_offsets[groups * RADIX_DIGITS * pass], dynamic_id + pass, size,
                            s[0]);
                        kernel.process(id);
                    });
            });
        }
        sycl::free(storage, q);

        q.wait();
    }

    uint32_t *_array, _size;
};

template <typename KeyT = uint32_t, bool IS_DESC = false, uint32_t WARP_THREADS = 16>
struct RadixSortKernel {
    KeyT *_array;
    uint32_t _size;
    RadixSortKernel(KeyT *array, uint32_t size) : _array(array), _size(size) {}

    void process(sycl::queue &q) {
        if (_size < std::pow(2, 15.5)) {
            RadixSortKernel_Impl<8, 128, 16, true, WARP_THREADS>(_array, _size).process(q);
        } else if(_size < std::pow(2, 18.5)) {
            RadixSortKernel_Impl<8, 256, 16, true, WARP_THREADS>(_array, _size).process(q);
        } else if(_size < std::pow(2, 19.5)) {
            RadixSortKernel_Impl<8, 256, 20, true, WARP_THREADS>(_array, _size).process(q);
        } else if(_size < std::pow(2, 22.5)) {
            RadixSortKernel_Impl<8, 256, 40, true, WARP_THREADS>(_array, _size).process(q);
        } else if(_size < std::pow(2, 23.5)) {
            RadixSortKernel_Impl<8, 256, 28, true, WARP_THREADS>(_array, _size).process(q);
        } else if(_size < std::pow(2, 24.5)) {
            RadixSortKernel_Impl<8, 128, 44, true, WARP_THREADS>(_array, _size).process(q);
        } else if(_size < std::pow(2, 27.5)) {
            RadixSortKernel_Impl<8, 256, 32, true, WARP_THREADS>(_array, _size).process(q);
        } else if(_size < std::pow(2, 28.5)) {
            RadixSortKernel_Impl<8, 256, 28, true, WARP_THREADS>(_array, _size).process(q);
        } else{
            RadixSortKernel_Impl<8, 128, 48, true, WARP_THREADS>(_array, _size).process(q);
        }
    }
};
