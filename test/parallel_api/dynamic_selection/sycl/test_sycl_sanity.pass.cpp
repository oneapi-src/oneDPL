// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cstdio>
#include <memory>
#include <iostream>
#include <vector>
#include "support/test_config.h"
#if TEST_DYNAMIC_SELECTION_AVAILABLE
#    include "support/sycl_sanity.h"

int
run_test(sycl::queue q)
{
    const int num_items = 1000000;
    using DoubleVector = std::vector<double>;
    DoubleVector a(num_items), b(num_items), c(num_items);
    for (int i = 0; i < num_items; ++i)
    {
        a[i] = i;
        b[i] = num_items - i;
        c[i] = 0;
    }

    auto device = q.get_info<sycl::info::queue::device>();
    if (device.is_cpu())
    {
        std::printf("running on cpu\n");
    }
    else if (device.is_gpu())
    {
        std::printf("running on gpu\n");
    }
    else
    {
        std::printf("running on other\n");
    }

    {
        sycl::buffer<double> a_buf(a), b_buf(b), c_buf(c);
        q.submit([&](sycl::handler& h) {
            sycl::accessor a_(a_buf, h, sycl::read_only);
            sycl::accessor b_(b_buf, h, sycl::read_only);
            sycl::accessor c_(c_buf, h, sycl::write_only);
            h.parallel_for<
                TestUtils::unique_kernel_name<class sum1, TestUtils::uniq_kernel_index<sycl::usm::alloc::shared>()>>(
                num_items, [=](auto j) { c_[j] += a_[j] + b_[j]; });
        });

        q.submit([&](sycl::handler& h) {
             sycl::accessor a_(a_buf, h, sycl::read_only);
             sycl::accessor b_(b_buf, h, sycl::read_only);
             sycl::accessor c_(c_buf, h, sycl::write_only);
             h.parallel_for<
                 TestUtils::unique_kernel_name<class sum2, TestUtils::uniq_kernel_index<sycl::usm::alloc::shared>()>>(
                 num_items, [=](auto j) { c_[j] += a_[j] + b_[j]; });
         }).wait();
    }
    printf("%f, %f, %f, %f\n", c[0], c[1000], c[10000], c[100000]);
    return 0;
}

int
test_runner()
{
    int r = 0;
    try
    {
        sycl::queue default_queue;
        default_queue = sycl::queue{sycl::default_selector_v};
        r += run_test(default_queue);
    }
    catch (sycl::exception)
    {
        std::cout << "SKIPPED: Unable to run with default_selector\n";
    }

    try
    {
        sycl::queue gpu_queue;
        gpu_queue = sycl::queue{sycl::gpu_selector_v};
        r += run_test(gpu_queue);
    }
    catch (sycl::exception)
    {
        std::cout << "SKIPPED: Unable to run with gpu_selector\n";
    }

    try
    {
        sycl::queue cpu_queue;
        cpu_queue = sycl::queue{sycl::cpu_selector_v};
        r += run_test(cpu_queue);
    }
    catch (sycl::exception)
    {
        std::cout << "SKIPPED: Unable to run with cpu_selector\n";
    }

    return r;
}
#endif

int
main()
{
#if TEST_DYNAMIC_SELECTION_AVAILABLE
    return test_runner();
#else
    std::cout << "SKIPPED\n";
    return 0;
#endif // TEST_DYNAMIC_SELECTION_AVAILABLE
}
