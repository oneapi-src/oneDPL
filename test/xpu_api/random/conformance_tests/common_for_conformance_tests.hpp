// -*- C++ -*-
//===-- common_for_conformance_tests.hpp -----------------------------------===//
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
//
// Abstract:
//
// Common functionality for conformance tests

#ifndef DPSTD_RANDOM_CONFORMANCE_TESTS_COMMON
#define DPSTD_RANDOM_CONFORMANCE_TESTS_COMMON

#include <vector>
#include <CL/sycl.hpp>
#include <random>

constexpr auto REF_SAMPLE_ID = 9999;

template<class Engine, int NGenSamples, int NElemsInResultType>
typename Engine::scalar_type test(sycl::queue& queue) {

    using result_type = typename Engine::scalar_type;

    // Memory allocation
    std::vector<result_type> dpstd_samples(NGenSamples);

    // Random number generation
    {
        sycl::buffer<result_type, 1> dpstd_buffer(dpstd_samples.data(), NGenSamples);

        try {
            queue.submit([&](sycl::handler &cgh) {
                auto dpstd_acc = dpstd_buffer.template get_access<sycl::access::mode::write>(cgh);

                cgh.parallel_for<>(sycl::range<1>(NGenSamples / NElemsInResultType),
                        [=](sycl::item<1> idx) {

                    unsigned long long offset = idx.get_linear_id() * NElemsInResultType;
                    Engine engine;
                    engine.discard(offset);

                    sycl::vec<result_type, NElemsInResultType> res = engine();
                    res.store(idx.get_linear_id(), dpstd_acc);
                });
            });
        }
        catch(sycl::exception const& e) {
            std::cout << "\t\tSYCL exception during generation\n"
                      << e.what() << std::endl << "OpenCL status: " << e.get_cl_code() << std::endl;
            return 0;
        }

        queue.wait_and_throw();
    }

    return dpstd_samples[REF_SAMPLE_ID];
}

#endif // ifndef DPSTD_RANDOM_CONFORMANCE_TESTS_COMMON