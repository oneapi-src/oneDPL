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

#ifndef _DPSTD_RANDOM_CONFORMANCE_TESTS_COMMON_HPP
#define _DPSTD_RANDOM_CONFORMANCE_TESTS_COMMON_HPP

#include <vector>
#include <random>

#include "support/utils.h"

constexpr auto REF_SAMPLE_ID = 9999;

template<class Engine, int NGenSamples, int NElemsInResultType>
typename Engine::scalar_type test(sycl::queue& queue) {

    using result_type = typename Engine::scalar_type;

    // Memory allocation
    std::vector<result_type> dpstd_samples(NGenSamples);

#if 1

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
            queue.wait_and_throw();
        }
        catch(sycl::exception const& e) {
            std::cout << "\t\tSYCL exception during generation\n"
                      << e.what() << std::endl;
            return 0;
        }
    }
    std::cout << "\n\t\tres sycl: " << dpstd_samples[REF_SAMPLE_ID] << std::endl;
    return dpstd_samples[REF_SAMPLE_ID];

#else
    result_type res;
    // iterate through the different value of the offset
    for(int itr = 0; itr < 999; ++itr) {
        Engine engine;
        int disgard_value = itr;
        std::cout << "\n\t\tdisgard_value: " << disgard_value;
        engine.discard(disgard_value);
        for(int i = 0; i < NGenSamples-disgard_value;i++){
            res = engine();
            //std::cout << " " << res << std::endl;
        }
        if(res!=1955073260)
            std::cout << "\t\tError";
    }
    
    return res;
#endif
}

#endif // _DPSTD_RANDOM_CONFORMANCE_TESTS_COMMON_HPP
