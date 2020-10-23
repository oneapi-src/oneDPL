// -*- C++ -*-
//===-- permutation_iterator_sycl.pass.cpp --------------------------------===//
//
// Copyright (C) 2019-2020 Intel Corporation
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

#include <iostream>
#include <chrono>

#include <CL/sycl.hpp>

#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/numeric>
#include <oneapi/dpl/iterator>

template<typename _T1, typename _T2> void ASSERT_EQUAL(_T1&& X, _T2&& Y) {
    if(X!=Y)
        std::cout << "CHECK CORRECTNESS: fail (" << X << "," << Y << ")" << std::endl;
}

// Collect statistics of the times sampled, including the 95% confidence interval using the t-distribution
struct statistics {
    statistics(std::vector<size_t> const& samples)
        : min(std::numeric_limits<size_t>::max()), max(0), mean(0.), stddev(0.), confint(0.)
    {
        if (samples.size() > 10)
            std::cout << "Warning: statistics reported using first 10 samples\n";

        for (int i = 0; i != 10; ++i) {
            if (samples[i] < min)
                min = samples[i];
            if (samples[i] > max)
                max = samples[i];
            mean += samples[i];
        }

        mean /= samples.size();

        for (int i = 0; i != 10; ++i) {
            stddev += (samples[i] - mean) * (samples[i] - mean);
        }
        stddev /= samples.size() - 1;
        stddev = std::sqrt(stddev);

        // value for 95% confidence interval with 10 samples (9 degrees of freedom) is 2.262
        confint = 2.262 * stddev/std::sqrt(10.0);
    }

    size_t min;
    size_t max;
    float mean;
    float stddev;
    float confint;
};


template <typename RefPolicy, typename Policy, typename Iterator, typename IndexMap>
void evaluate(RefPolicy&& ref_policy, Policy&& policy, Iterator ref_begin, Iterator ref_end,
              oneapi::dpl::permutation_iterator<Iterator, IndexMap> perm_start, std::string test) {
    using value_type = typename std::iterator_traits<Iterator>::value_type;
    using clock = std::chrono::high_resolution_clock;

    // Collect 10 samples and compute statistics using t-distribution
    std::vector<size_t> ref_times;
    std::vector<size_t> perm_times;

    ref_times.reserve(10);
    perm_times.reserve(10);

    const size_t n = std::distance(ref_begin, ref_end);
    const value_type zero = 0;
    for (int i = 0; i != 10; ++i) {
        auto start = clock::now();
        value_type ref_sum = std::reduce(std::forward<RefPolicy>(ref_policy), ref_begin, ref_end, zero);
        auto stop = clock::now();
        ref_times.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count());

        start = clock::now();
        value_type perm_sum = std::reduce(std::forward<Policy>(policy), perm_start, perm_start+n, zero);
        stop = clock::now();
        perm_times.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count());

        ASSERT_EQUAL(ref_sum, perm_sum);
    }
    statistics ref_stats(ref_times);
    statistics perm_stats(perm_times);
#if PRINT_STATISTICS
    std::cout << "permutation_iterator(" << test << ") (ns): " << perm_stats.mean
              << " conf. int. " << perm_stats.confint
              << ", min " << perm_stats.min << ", max " << perm_stats.max << "\n"
              << "reference (ns): " << ref_stats.mean << " conf. int. " << ref_stats.confint
              << ", min " << ref_stats.min << ", max " << ref_stats.max << "\n"
              << "slowdown (perm / ref): " << perm_stats.mean / ref_stats.mean << std::endl;
#endif
}

struct cyclic_mapper
{
    cyclic_mapper(size_t n, size_t ncycles) : num_cycles(ncycles), cycle_size(n/ncycles) {}

    size_t operator[](size_t i) const {
       return (i / num_cycles) + (i % num_cycles)*cycle_size;
    }

private:
    int num_cycles;
    int cycle_size;
};

int main(int argc, char** argv) {

    int n = 10000;
    if (argc == 2)
        n = std::atoi(argv[1]);

    if (n % 1000 != 0) {
        std::cout << "Error: num_elements must be a multiple of 1000\n";
        return 1;
    }

    // storage for the reference, and source/map elements needed to construct permutation_iterator
    std::vector<uint64_t> ref(n);
    std::vector<uint64_t> src(n);
    std::vector<uint64_t> map(n);

    // Case 1 -- Compare linear traversal to evaluate overhead of permutation_iterator
    std::iota(ref.begin(), ref.end(), 0);
    std::iota(src.begin(), src.end(), 0);
    std::iota(map.begin(), map.end(), 0);

    using iterator = std::vector<uint64_t>::iterator;
    oneapi::dpl::permutation_iterator<iterator, iterator> p(src.begin(), map.begin());

    evaluate(oneapi::dpl::execution::par, oneapi::dpl::execution::par, ref.begin(), ref.end(), p, std::string("CPU Linear"));

    // Case 2 -- Block-cyclic access, block size = 1000
    // Example: n = 12, block size = 4, data = { 0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11 }
    uint64_t block_size = 1000;
    uint64_t num_blocks = n/1000;
    for (uint64_t i = 0; i != n; ++i) {
        // first value in a block is the block id, remaining values are offset with stride num_blocks
        uint64_t value = (i / block_size) + (i % block_size) * num_blocks;
        ref[i] = value;
        map[i] = value;
    }

    evaluate(oneapi::dpl::execution::par, oneapi::dpl::execution::par, ref.begin(), ref.end(), p, std::string("CPU Block Cyclic"));

    // Case 3 -- Reverse order access
    for (uint64_t i = 1; i <= n; ++i) {
        ref[i-1] = n - i;
        map[i-1] = n - i;
    }

    evaluate(oneapi::dpl::execution::par, oneapi::dpl::execution::par, ref.begin(), ref.end(), p, std::string("CPU Reverse"));

    // Case 4 -- Linear traversal on accelerator
    {
        using policy_type = decltype(oneapi::dpl::execution::dpcpp_default);

        // create buffers
        cl::sycl::buffer<uint64_t, 1> ref_buf{ cl::sycl::range<1>(n) };
        cl::sycl::buffer<uint64_t, 1> src_buf{ cl::sycl::range<1>(n) };
        cl::sycl::buffer<uint64_t, 1> map_buf{ cl::sycl::range<1>(n) };

        // initialize data
        {
            auto ref = ref_buf.template get_access<cl::sycl::access::mode::write>();
            auto src = src_buf.template get_access<cl::sycl::access::mode::write>();
            auto map = map_buf.template get_access<cl::sycl::access::mode::write>();

            for (uint64_t i  = 0; i != n; ++i) {
                ref[i] = i;
                src[i] = i;
                map[i] = i;
            }
        }

        auto ref_begin = oneapi::dpl::begin(ref_buf);
        auto ref_end = oneapi::dpl::end(ref_buf);
        auto src_begin = oneapi::dpl::begin(src_buf);
        auto map_begin = oneapi::dpl::begin(map_buf);
        auto perm_begin = oneapi::dpl::make_permutation_iterator(src_begin, map_begin);

        auto policy = oneapi::dpl::execution::make_device_policy<class GPULinearRef>(oneapi::dpl::execution::dpcpp_default);
        auto policy2 = oneapi::dpl::execution::make_device_policy<class GPULinear>(oneapi::dpl::execution::dpcpp_default);
        evaluate(policy, policy2, ref_begin, ref_end, perm_begin, std::string("GPU Linear"));
    }

    // Case 5 -- Cyclic traversal on accelerator using function object for map indices
    {
        using policy_type = decltype(oneapi::dpl::execution::dpcpp_default);

        // create buffers
        cl::sycl::buffer<uint64_t, 1> ref_buf{ cl::sycl::range<1>(n) };
        cl::sycl::buffer<uint64_t, 1> src_buf{ cl::sycl::range<1>(n) };
        cyclic_mapper map(n, 100);

        // initialize data
        {
            auto ref = ref_buf.template get_access<cl::sycl::access::mode::write>();
            auto src = src_buf.template get_access<cl::sycl::access::mode::write>();

            for (uint64_t i  = 0; i != n; ++i) {
                ref[i] = map[i];
                src[i] = i;
            }
        }

        auto ref_begin = oneapi::dpl::begin(ref_buf);
        auto ref_end = oneapi::dpl::end(ref_buf);
        auto src_begin = oneapi::dpl::begin(src_buf);
        auto perm_begin = oneapi::dpl::make_permutation_iterator(src_begin, map);

        auto policy = oneapi::dpl::execution::make_device_policy<class GPUCyclicRef>(oneapi::dpl::execution::dpcpp_default);
        auto policy2 = oneapi::dpl::execution::make_device_policy<class GPUCyclic>(oneapi::dpl::execution::dpcpp_default);
        evaluate(policy, policy2, ref_begin, ref_end, perm_begin, std::string("GPU Cyclic"));
    }
    std::cout << "done" << std::endl;
    return 0;
}
