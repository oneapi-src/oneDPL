// -*- C++ -*-
//===-- discard_iterator_sycl.pass.cpp --------------------------------===//
//
// Copyright (C) 2020 Intel Corporation
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

template <typename Policy, typename Iterator>
void evaluate(Policy&& policy, Iterator ref_begin, Iterator ref_end,
              oneapi::dpl::discard_iterator dev_null, std::string test) {
    using policy_type = typename std::decay<Policy>::type;
    using value_type = typename std::iterator_traits<Iterator>::value_type;
    using clock = std::chrono::high_resolution_clock;

    // Collect 10 samples and compute statistics using t-distribution
    std::vector<size_t> ref_times;
    std::vector<size_t> discard_times;

    ref_times.reserve(10);
    discard_times.reserve(10);

    const size_t n = std::distance(ref_begin, ref_end);
    const value_type zero = 0;
    for (int i = 0; i != 10; ++i) {
        auto start = clock::now();
        std::transform(std::forward<Policy>(policy), ref_begin, ref_end, ref_begin, oneapi::dpl::identity{});
        auto stop = clock::now();
        ref_times.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count());

        typename oneapi::dpl::internal::rebind_policy<policy_type, class DiscardEval>::type new_policy(policy);
        
        start = clock::now();
        std::transform(std::move(new_policy), ref_begin, ref_end, dev_null, oneapi::dpl::identity{});
        stop = clock::now();
        discard_times.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count());
    }

    statistics ref_stats(ref_times);
    statistics discard_stats(discard_times);
#if PRINT_STATISTICS
    std::cout << "discard_iterator(" << test << ") (ns): " << discard_stats.mean
              << " conf. int. " << discard_stats.confint
              << ", min " << discard_stats.min << ", max " << discard_stats.max << "\n"
              << "reference (ns): " << ref_stats.mean << " conf. int. " << ref_stats.confint
              << ", min " << ref_stats.min << ", max " << ref_stats.max << "\n"
              << "slowdown (discard / ref): " << discard_stats.mean / ref_stats.mean << std::endl;
#endif
}

struct negate_stencil_fun {
    typedef bool result_of;

    template<typename _T1> result_of operator() (_T1&& a) const {
        using std::get;
        return !get<1>(a);
    }
};

int main(int argc, char** argv) {

    int n = 10000;
    if (argc == 2)
        n = std::atoi(argv[1]);

    if (n % 1000 != 0) {
        std::cout << "Error: num_elements must be a multiple of 1000\n";
        return 1;
    }

    // storage for the reference and source elements needed to construct discard_iterator
    std::vector<uint64_t> ref(n);
    std::vector<uint64_t> src(n);

    // Case 0 -- Functional Tests

    {
        using oneapi::dpl::discard_iterator;
        // Increment and comparison
        discard_iterator d1;
        discard_iterator d2(0);

        ASSERT_EQUAL(0, d1 - d2);

        d1++;

        ASSERT_EQUAL(1, d1 - d2);

        d2+=2;

        ASSERT_EQUAL(0, ++d1 - d2);

        // Wrapped iterator

        *d1 = 0;
        *d2 = 5;

        ASSERT_EQUAL(0, d1 - d2);

        auto zip = oneapi::dpl::make_zip_iterator(discard_iterator());

        *zip = std::make_tuple(0);

    }

    // Case 1 -- Compare traversal on CPU to evaluate overhead of discard_iterator
    std::iota(ref.begin(), ref.end(), 0);
    std::iota(src.begin(), src.end(), 0);

    using iterator = std::vector<uint64_t>::iterator;
    oneapi::dpl::discard_iterator p;

    evaluate(oneapi::dpl::execution::par, ref.begin(), ref.end(), p, std::string("CPU discard"));

    // Case 2 -- Compare traversal on accelerator
    {
        using policy_type = decltype(oneapi::dpl::execution::dpcpp_default);

        // create buffers
        cl::sycl::buffer<uint64_t, 1> ref_buf{ cl::sycl::range<1>(n) };
        cl::sycl::buffer<uint64_t, 1> src_buf{ cl::sycl::range<1>(n) };

        // initialize data
        {
            auto ref = ref_buf.template get_access<cl::sycl::access::mode::write>();
            auto src = src_buf.template get_access<cl::sycl::access::mode::write>();

            for (uint64_t i  = 0; i != n; ++i) {
                ref[i] = i;
                src[i] = i;
            }
        }

        auto ref_begin = oneapi::dpl::begin(ref_buf);
        auto ref_end = oneapi::dpl::end(ref_buf);
        auto src_begin = oneapi::dpl::begin(src_buf);
        auto out_begin = oneapi::dpl::discard_iterator();

        auto policy = oneapi::dpl::execution::make_device_policy<class GPULinear>(oneapi::dpl::execution::dpcpp_default);

        evaluate(policy, ref_begin, ref_end, out_begin, std::string("GPU discard"));
    }

    {
        using policy_type = decltype(oneapi::dpl::execution::dpcpp_default);

        // create buffers
        cl::sycl::buffer<uint64_t, 1> mask_buf{ cl::sycl::range<1>(n) };
        cl::sycl::buffer<uint64_t, 1> src_buf{ cl::sycl::range<1>(n) };
        cl::sycl::buffer<uint64_t, 1> dst_buf{ cl::sycl::range<1>(n) };
    
        auto policy = oneapi::dpl::execution::make_device_policy<class GPUCopyIf>(oneapi::dpl::execution::dpcpp_default);
        
        {
            auto stencil = mask_buf.template get_access<cl::sycl::access::mode::write>();
            auto src = src_buf.template get_access<cl::sycl::access::mode::write>();
            auto dst = dst_buf.template get_access<cl::sycl::access::mode::write>();

            uint64_t even_odd = 0;
            for (uint64_t i  = 0; i != n; ++i) {
                stencil[i] = (even_odd++)%2;
                src[i] = 1;
                dst[i] = -1;
            }
        }

        auto stencil = oneapi::dpl::begin(mask_buf);
        auto first = oneapi::dpl::begin(src_buf);
        auto last = oneapi::dpl::end(src_buf);
        auto tmp = oneapi::dpl::begin(dst_buf);
        auto ret_val = std::copy_if(policy,
                                    oneapi::dpl::make_zip_iterator(first,stencil),
                                    oneapi::dpl::make_zip_iterator(first,stencil) + std::distance(first,last),
                                    oneapi::dpl::make_zip_iterator(tmp, oneapi::dpl::discard_iterator()),
                                    negate_stencil_fun());

        auto dist = std::distance(tmp, std::get<0>(ret_val.base()));
        auto sum = std::reduce(policy, tmp, std::get<0>(ret_val.base()), 0);

        ASSERT_EQUAL(dist, n/2);
        ASSERT_EQUAL(sum, n/2);
        {
            auto stencil = mask_buf.template get_access<cl::sycl::access::mode::read>();
            uint64_t even_odd = 0;
            for (uint64_t i  = 0; i != n; ++i) {
                ASSERT_EQUAL(stencil[i], (even_odd++)%2);
            }
        }
#ifdef VERBOSE
        {
            auto stencil = mask_buf.template get_access<cl::sycl::access::mode::read>();
            auto src = src_buf.template get_access<cl::sycl::access::mode::read>();
            auto dst = dst_buf.template get_access<cl::sycl::access::mode::read>();
            for (uint64_t i  = 0; i != 10; ++i) {
                std::cout << "(" << stencil[i]
                          << "," << src[i]
                          << "," << dst[i] << ") ";
            }
            std::cout << std::endl;
        }
#endif
        
    }

	{
		// reduce_by_segment with discard_iterator as output to test use case from QMCPACK

		// create buffers
		cl::sycl::buffer<uint64_t, 1> key_buf{ cl::sycl::range<1>(13) };
		cl::sycl::buffer<uint64_t, 1> val_buf{ cl::sycl::range<1>(13) };
		cl::sycl::buffer<uint64_t, 1> val_res_buf{ cl::sycl::range<1>(13) };

		{
			auto keys = key_buf.template get_access<cl::sycl::access::mode::read_write>();
			auto vals = val_buf.template get_access<cl::sycl::access::mode::read_write>();
			auto vals_res = val_res_buf.template get_access<cl::sycl::access::mode::read_write>();

			//T keys[n1] = { 1, 2, 3, 4, 1, 1, 3, 3, 1, 1, 3, 3, 0 };
			//T vals[n1] = { 1, 2, 3, 4, 1, 1, 3, 3, 1, 1, 3, 3, 0 };

			// vals_result = {1, 2, 3, 4, 2, 6, 2, 6, 0};

			// Initialize data
			for (int i = 0; i != 12; ++i) {
				keys[i] = i % 4 + 1;
				vals[i] = i % 4 + 1;
				vals_res[i] = 1;
				if (i > 3) {
					++i;
					keys[i] = keys[i - 1];
					vals[i] = vals[i - 1];
					vals_res[i] = 0;
				}
			}
			keys[12] = 0;
			vals[12] = 0;
		}

		// create sycl iterators
		auto key_beg = oneapi::dpl::begin(key_buf);
		auto key_end = oneapi::dpl::end(key_buf);
		auto val_beg = oneapi::dpl::begin(val_buf);
		auto val_res_beg = oneapi::dpl::begin(val_res_buf);

		// call algorithm
		auto new_policy2 = oneapi::dpl::execution::make_device_policy<class ReduceBySegment2>(
			oneapi::dpl::execution::dpcpp_default);
		auto res2 = oneapi::dpl::reduce_by_segment(new_policy2, key_beg, key_end,
			val_beg, oneapi::dpl::discard_iterator(), val_res_beg);

		{
			// check values
			auto vals_res = val_res_buf.template get_access<cl::sycl::access::mode::read_write>();
			int n = std::distance(val_res_beg, res2.second);
			for (auto i = 0; i != n; ++i) {
				if (i < 4) {
					ASSERT_EQUAL(vals_res[i], i + 1);
				}
				else if (i == 4 || i == 6) {
					ASSERT_EQUAL(vals_res[i], 2);
				}
				else if (i == 5 || i == 7) {
					ASSERT_EQUAL(vals_res[i], 6);
				}
				else if (i == 8) {
					ASSERT_EQUAL(vals_res[i], 0);
				}
				else {
					std::cout << "fail: unexpected values in output range\n";
				}
			}
		}
	}

    std::cout << "done" << std::endl;
    return 0;
}
