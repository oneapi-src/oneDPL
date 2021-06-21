// -*- C++ -*-
//===-- transform_if.pass.cpp ----------------------------------------------------===//
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

// Tests for transform_if

#include "support/test_config.h"

#include _PSTL_TEST_HEADER(execution)
#include _PSTL_TEST_HEADER(algorithm)

#include "support/utils.h"
#include <random>

using namespace TestUtils;

struct test_transform_if
{
    template <typename Policy, typename InputIterator1, typename InputIterator2, typename OutputIterator, typename UnaryOperation, typename Predicate>
    void
    operator()(Policy&& exec, InputIterator1 first, InputIterator1 last, InputIterator2 mask, InputIterator2 /* mask_end */, OutputIterator result, OutputIterator /* result_last */, UnaryOperation op, Predicate pred)
    {
        const int n = ::std::distance(first, last);
        auto result_copy = result;

        // Try transform_if
        //transform_if(exec, first, last, mask, result, op, pred);
        oneapi::dpl::transform_if(exec, first, last, mask, result, op, pred);

        for (auto iter = first; iter != last; ++iter) {
            //EXPECT_EQ(*(result + (iter - first)), *(result2 + (iter - first)), "wrong value in output sequence"); 
        }
    }
};

/*template <typename Policy, typename Queue>
void test_transform_if(Policy policy, int input_size, Queue q) {
    auto sycl_deleter = [q](int* mem) { sycl::free(mem, q.get_context()); };

    ::std::unique_ptr<int, decltype(sycl_deleter)>

    data((int*)sycl::malloc_shared(sizeof(int) * input_size, q.get_device(), q.get_context()), sycl_deleter),

    mask((int*)sycl::malloc_shared(sizeof(int) * input_size, q.get_device(), q.get_context()), sycl_deleter),

    result((int*)sycl::malloc_shared(sizeof(int) * input_size, q.get_device(), q.get_context()), sycl_deleter);

    int* data_ptr = data.get();
    int* mask_ptr = mask.get();
    int* res_ptr = result.get();

    for (int i = 0; i != input_size; ++i) {
        data_ptr[i] = i+1; // data = {1, 2, 3, ..., n}
        mask_ptr[i] = (i+1) % 2; // mask = {1, 0, 1, 0, ..., 1, 0}
        res_ptr[i] = 0; // result = {0, 0, 0, ..., 0}
    }   

    // obtain a time-based seed:
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();

    // randomize mask indices
    shuffle(mask_ptr, mask_ptr + input_size, std::default_random_engine(seed));

    // call transform_if
    oneapi::dpl::transform_if(policy, data_ptr, data_ptr + input_size, mask_ptr, res_ptr, std::negate<int>(), oneapi::dpl::identity());

    // test if transform_if has correct output
    for (int i = 0; i != input_size; ++i) {
        if ((mask_ptr[i] == 1 && res_ptr[i] != -(data_ptr[i])) || (mask_ptr[i] == 0 && res_ptr[i] != 0)) {
            std::cout << "Input size " << input_size << ": Failed\n";
            break;
        }
    }
}*/

template <typename In1, typename In2, typename Out>
void test() {
    for (size_t n = 1; n <= max_n; n = n <= 16 ? n + 1 : size_t(3.1415 * n)) {
        Sequence<In1> in1(n, [](size_t k) { return k % 5 != 1 ? In1(3 * k + 7) : 0; });
        Sequence<In2> in2(n, [](size_t k) { return k % 2 == 0 ? 1 : 0; });

        Sequence<Out> out(n, [](size_t) { return 0; });

        invoke_on_all_policies<0>()(test_transform_if(), in1.begin(), in1.end(), in2.begin(),
                                    in2.end(), out.begin(), out.end(), std::negate<int>(), oneapi::dpl::identity());
        /*invoke_on_all_policies<1>()(test_transform_if(), in1.cbegin(), in1.cend(), in2.cbegin(),
                                    in2.cend(), out.begin(), out.end(), std::negate<int>(), oneapi::dpl::identity());*/

    }
}
 
int main() {
    /*const int max_n = 100000;
    sycl::queue q;
    for (size_t n = 1; n <= max_n; n = n <= 16 ? n + 1 : size_t(3.1415 * n)) {
        test_transform_if(oneapi::dpl::execution::seq, n, q);
        test_transform_if(oneapi::dpl::execution::unseq, n, q);
        test_transform_if(oneapi::dpl::execution::par, n, q);
        test_transform_if(oneapi::dpl::execution::par_unseq, n, q);
        test_transform_if(oneapi::dpl::execution::make_device_policy(q), n, q);

        #if ONEDPL_FPGA_DEVICE
            test_transform_if(oneapi::dpl::execution::dpcpp_fpga, n, q);
        #endif
    }*/

    test<int32_t, int32_t, int32_t>();

    return done();
}
