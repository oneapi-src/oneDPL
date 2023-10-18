// -*- C++ -*-
//===-- reduce.pass.cpp ---------------------------------------------------===//
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

//#define WAIT_DEBUGGER_FOR_ATTACH 30

#include "support/test_config.h"

#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/numeric>

// to check __has_known_identity
#include <oneapi/dpl/pstl/hetero/dpcpp/unseq_backend_sycl.h>

#include "support/utils.h"

#include <cstdint>
#include <iostream>
#if WAIT_DEBUGGER_FOR_ATTACH
#include <thread>
#include <chrono>
#endif
#include <sycl/sycl.hpp>

int
main(int argc, const char* argv[])
{
#if WAIT_DEBUGGER_FOR_ATTACH
    std::cout << "Waiting for attach from debugger " << WAIT_DEBUGGER_FOR_ATTACH  << " sec." << std::endl << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(WAIT_DEBUGGER_FOR_ATTACH));
#endif

    sycl::queue q{sycl::default_selector_v};
    const sycl::device& d = q.get_device();

    const std::string& name = d.get_info<sycl::info::device::name>();
    const std::string& driver_version = d.get_info<sycl::info::device::driver_version>();

    std::cout << "Device " << name << " [" << driver_version << "]" << std::endl;
    std::cout << "_USE_GROUP_ALGOS = " << _USE_GROUP_ALGOS << std::endl;
    std::cout << "_ONEDPL_SYCL_INTEL_COMPILER = " << _ONEDPL_SYCL_INTEL_COMPILER << std::endl;
    std::cout << "_ONEDPL_SYCL2020_COLLECTIVES_PRESENT = " << _ONEDPL_SYCL2020_COLLECTIVES_PRESENT << std::endl;

    constexpr size_t N = 6;
    using T = std::int64_t;
    const T init = T(2);

    std::cout << std::endl;
    std::cout << "Call sycl::reduce_over_group inside oneDPL library" << std::endl;
    {
        using __has_known_identity_t = typename oneapi::dpl::unseq_backend::__has_known_identity<std::multiplies<T>, T>::type;
        std::cout << "__has_known_identity_t = " << __has_known_identity_t{} << std::endl;

        T* data = sycl::malloc_shared<T>(N + 1, q);

        for (int i = 0; i < N; ++i)
        {
            data[i] = T(i + 1);
            std::cout << data[i] << " ";
        }

        auto policy = oneapi::dpl::execution::make_device_policy(q);

        T acc = std::reduce(policy, data, data + N, init, std::multiplies<T>());

        std::cout << std::endl << "Result: " << acc << std::endl;
        sycl::free(data, q);
    }

    std::cout << std::endl;
    std::cout << "Call sycl::reduce_over_group directly" << std::endl;
    {
        sycl::buffer<T> bufA{N};
        {
            sycl::host_accessor accA{bufA};
            for (int i = 0; i < N; ++i)
            {
                accA[i] = T(i + 1);
                std::cout << accA[i] << " ";
            }
        }

        constexpr std::size_t kItems = 2;   //16;
        
        sycl::buffer<T> bufSum{N / kItems}; // partial values
        q.submit([&](sycl::handler& cgh) {
            sycl::accessor accA{bufA, cgh, sycl::read_only};
            sycl::accessor sum{bufSum, cgh, sycl::write_only};
            //cgh.parallel_for(sycl::nd_range<1>{N, kItems}, [=](sycl::nd_item<1> item) {
            cgh.parallel_for(sycl::nd_range<1>{256, 2}, [=](sycl::nd_item<1> item) {
                // Group algorithm
                int partial_sum = sycl::reduce_over_group(item.get_group(), accA[item.get_global_id()], std::multiplies<T>());
                if (item.get_group().leader())
                {
                    sum[item.get_group(0)] = partial_sum;
                }
            });
        }).wait();

        {
            auto result = init;

            auto sum = bufSum.get_host_access();
            for (auto it = sum.begin(); it != sum.end(); ++it)
            {
                std::multiplies<T> op;
                result = op(result, *it);
            }
            //auto result = std::accumulate(sum.begin(), sum.end(), 0);

            std::cout << std::endl << "Result: " << result << std::endl;
        }
    }

    return 0;
}