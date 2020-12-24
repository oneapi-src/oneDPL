// -*- C++ -*-
//===-- discard_block_std_template_test.cpp -------------------------------===//
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
//
// Abstract:
//
// Test of discard_block_engine - comparison with std::

#if (!defined(_ONEDPL_BACKEND_SYCL) || (_ONEDPL_BACKEND_SYCL == 0))
#include <iostream>

int main() {
    std::cout << "\tTest is skipped for non-SYCL backend. Passed" << std::endl;
    return 0;
}

#else

#include <iostream>
#include <vector>
#include <CL/sycl.hpp>
#include <random>
#include <oneapi/dpl/random>

template<class Engine, class StdEngine>
int test(oneapi::dpl::internal::element_type_t<typename Engine::result_type> seed, std::int32_t nsamples) {

    sycl::queue queue(sycl::default_selector{});

    // memory allocation
    std::vector<oneapi::dpl::internal::element_type_t<typename Engine::result_type>> std_samples(nsamples);
    std::vector<oneapi::dpl::internal::element_type_t<typename Engine::result_type>> dpstd_samples(nsamples);
    constexpr int num_elems = oneapi::dpl::internal::type_traits_t<typename Engine::result_type>::num_elems == 0 ? 1 :
        oneapi::dpl::internal::type_traits_t<typename Engine::result_type>::num_elems;

    // dpstd generation
    {
        sycl::buffer<oneapi::dpl::internal::element_type_t<typename Engine::result_type>, 1> dpstd_buffer(dpstd_samples.data(), nsamples);

        queue.submit([&](sycl::handler &cgh) {
            auto dpstd_acc = dpstd_buffer.template get_access<sycl::access::mode::write>(cgh);

            cgh.parallel_for<>(sycl::range<1>(nsamples / num_elems),
                    [=](sycl::item<1> idx) {

                unsigned long long offset = idx.get_linear_id() * num_elems;
                Engine engine(seed, offset);

                sycl::vec<oneapi::dpl::internal::element_type_t<typename Engine::result_type>, num_elems> res = engine();
                res.store(idx.get_linear_id(), dpstd_acc.get_pointer());
            });
        });
        queue.wait();
    }

    // std generation
    StdEngine std_engine(seed);
    for(int i = 0; i < nsamples; ++i)
        std_samples[i] = std_engine();

    // comparison
    int err = 0;
    for(int i = 0; i < nsamples; ++i) {
        if (std_samples[i] != dpstd_samples[i]) {
            std::cout << "\nError: std_sample[" << i << "] = " << std_samples[i] << ", dpstd_samples[" << i << "] = " << dpstd_samples[i];
            err++;
        }
    }

    if(err) {
        std::cout << "\tFailed" << std::endl;
    }
    else{
        std::cout << "\tPassed" << std::endl;
    }

    return err;
}

template<class Engine, class StdEngine>
int test_portion(oneapi::dpl::internal::element_type_t<typename Engine::result_type> seed, int nsamples, unsigned int part) {

    sycl::queue queue(sycl::default_selector{});

    // memory allocation
    std::vector<oneapi::dpl::internal::element_type_t<typename Engine::result_type>> std_samples(nsamples);
    std::vector<oneapi::dpl::internal::element_type_t<typename Engine::result_type>> dpstd_samples(nsamples);

    // Calculate n_wi
    constexpr std::int32_t num_elems = oneapi::dpl::internal::type_traits_t<typename Engine::result_type>::num_elems == 0 ? 1 :
        oneapi::dpl::internal::type_traits_t<typename Engine::result_type>::num_elems;
    unsigned int n_elems = (part >= num_elems) ? num_elems : part;
    // dpstd generation
    {
        sycl::buffer<oneapi::dpl::internal::element_type_t<typename Engine::result_type>, 1> dpstd_buffer(dpstd_samples.data(), nsamples);

        queue.submit([&](sycl::handler &cgh) {
            auto dpstd_acc = dpstd_buffer.template get_access<sycl::access::mode::write>(cgh);

            cgh.parallel_for<>(sycl::range<1>(nsamples / n_elems),
                    [=](sycl::item<1> idx) {

                unsigned long long offset = idx.get_linear_id() * n_elems;
                Engine engine(seed, offset);

                auto res = engine(part);
                for(int i = 0; i < n_elems; ++i)
                    dpstd_acc.get_pointer()[offset + i] = res[i];
            });
        });
        queue.wait();
    }

    // std generation
    StdEngine std_engine(seed);
    for(int i = 0; i < nsamples; ++i)
        std_samples[i] = std_engine();

    // comparison
    int err = 0;
    for(int i = 0; i < nsamples; ++i) {
        if (std_samples[i] != dpstd_samples[i]) {
            std::cout << "\nError: std_sample[" << i << "] = " << std_samples[i] << ", dpstd_samples[" << i << "] = " << dpstd_samples[i];
            err++;
        }
    }

    if(err) {
        std::cout << "\tFailed" << std::endl;
    }
    else {
        std::cout << "\tPassed" << std::endl;
    }

    return err;
}

template<class Engine, class StdEngine>
int tests_set(int nsamples) {
    const int nseeds = 2;
    int64_t seed_array [nseeds] = {0, 19780503u};

    int err;
    for(int i = 0; i < nseeds; ++i) {
        std::cout << "DB test<Engine, StdEngine>(" << seed_array[i] << ", "<< nsamples << ")";
        err = test<Engine, StdEngine>(seed_array[i], nsamples);
        if(err) {
            return 1;
        }
    }

    return 0;
}

template<class Engine, class StdEngine>
int tests_set_portion(int nsamples, unsigned int part) {
    const int nseeds = 1;
    int64_t seed_array [nseeds] = {19780503u};

    int err;
    for(int i = 0; i < nseeds; ++i) {
        std::cout << "DB test_portion<Engine, StdEngine>(" << seed_array[i] << ", "<< nsamples << ", " << part << ")";
        err = test_portion<Engine, StdEngine>(seed_array[i], nsamples, part);
        if(err) {
            return 1;
        }
    }

    return 0;
}

int main() {
    constexpr int nsamples = 100;
    int err;

    // testing discard_block_engine<minstd_rand0, 100, 10>
    std::cout << "-------------------------------------------" << std::endl;
    std::cout << "discard_block_engine<minstd_rand0, 100, 10>" << std::endl;
    std::cout << "-------------------------------------------" << std::endl;
    err = tests_set<oneapi::dpl::discard_block_engine<oneapi::dpl::minstd_rand0, 100, 10>,
        std::discard_block_engine<std::minstd_rand0, 100, 10>>(nsamples);
    if(err) {
        std::cout << "Test FAILED" << std::endl;
        return 1;
    }

    // testing discard_block_engine<ranlux24_base, 30, 5>
    std::cout << "------------------------------------------" << std::endl;
    std::cout << "discard_block_engine<ranlux24_base, 30, 5>" << std::endl;
    std::cout << "------------------------------------------" << std::endl;
    err += tests_set<oneapi::dpl::discard_block_engine<oneapi::dpl::ranlux24_base, 30, 5>,
        std::discard_block_engine<std::ranlux24_base, 30, 5>>(nsamples);
    if(err) {
        std::cout << "Test FAILED" << std::endl;
        return 1;
    }

    // testing discard_block_engine<ranlux48_base, 100, 100>
    std::cout << "---------------------------------------------" << std::endl;
    std::cout << "discard_block_engine<ranlux48_base, 100, 100>" << std::endl;
    std::cout << "---------------------------------------------" << std::endl;
    err += tests_set<oneapi::dpl::discard_block_engine<oneapi::dpl::ranlux48_base, 100, 100>,
        std::discard_block_engine<std::ranlux48_base, 100, 100>>(nsamples);
    if(err) {
        std::cout << "Test FAILED" << std::endl;
        return 1;
    }

    // testing discard_block_engine<ranlux24_base_vec<1>, 70, 69>
    std::cout << "--------------------------------------------------" << std::endl;
    std::cout << "discard_block_engine<ranlux24_base_vec<1>, 70, 69>" << std::endl;
    std::cout << "--------------------------------------------------" << std::endl;
    err += tests_set<oneapi::dpl::discard_block_engine<oneapi::dpl::ranlux24_base_vec<1>, 70, 69>,
        std::discard_block_engine<std::ranlux24_base, 70, 69>>(nsamples);
    err += tests_set_portion<oneapi::dpl::discard_block_engine<oneapi::dpl::ranlux24_base_vec<1>, 70, 69>,
        std::discard_block_engine<std::ranlux24_base, 70, 69>>(nsamples, 1);
    if(err) {
        std::cout << "Test FAILED" << std::endl;
        return 1;
    }

    // testing discard_block_engine<ranlux24_base_vec<3>, 70, 69>
    std::cout << "--------------------------------------------------" << std::endl;
    std::cout << "discard_block_engine<ranlux24_base_vec<3>, 70, 69>" << std::endl;
    std::cout << "--------------------------------------------------" << std::endl;
    err += tests_set<oneapi::dpl::discard_block_engine<oneapi::dpl::ranlux24_base_vec<3>, 70, 69>,
        std::discard_block_engine<std::ranlux24_base, 70, 69>>(99);
    err += tests_set_portion<oneapi::dpl::discard_block_engine<oneapi::dpl::ranlux24_base_vec<3>, 70, 69>,
        std::discard_block_engine<std::ranlux24_base, 70, 69>>(nsamples, 2);
    if(err) {
        std::cout << "Test FAILED" << std::endl;
        return 1;
    }

    // testing discard_block_engine<ranlux24_base_vec<16>, 70, 69>
    std::cout << "---------------------------------------------------" << std::endl;
    std::cout << "discard_block_engine<ranlux24_base_vec<16>, 70, 69>" << std::endl;
    std::cout << "---------------------------------------------------" << std::endl;
    err += tests_set<oneapi::dpl::discard_block_engine<oneapi::dpl::ranlux24_base_vec<16>, 70, 69>,
        std::discard_block_engine<std::ranlux24_base, 70, 69>>(16);
    err += tests_set_portion<oneapi::dpl::discard_block_engine<oneapi::dpl::ranlux24_base_vec<16>, 70, 69>,
        std::discard_block_engine<std::ranlux24_base, 70, 69>>(nsamples, 1);
    err += tests_set_portion<oneapi::dpl::discard_block_engine<oneapi::dpl::ranlux24_base_vec<16>, 70, 69>,
        std::discard_block_engine<std::ranlux24_base, 70, 69>>(nsamples, 2);
    err += tests_set_portion<oneapi::dpl::discard_block_engine<oneapi::dpl::ranlux24_base_vec<16>, 70, 69>,
        std::discard_block_engine<std::ranlux24_base, 70, 69>>(90, 3);
    err += tests_set_portion<oneapi::dpl::discard_block_engine<oneapi::dpl::ranlux24_base_vec<16>, 70, 69>,
        std::discard_block_engine<std::ranlux24_base, 70, 69>>(nsamples, 4);
    err += tests_set_portion<oneapi::dpl::discard_block_engine<oneapi::dpl::ranlux24_base_vec<16>, 70, 69>,
        std::discard_block_engine<std::ranlux24_base, 70, 69>>(70, 7);
    err += tests_set_portion<oneapi::dpl::discard_block_engine<oneapi::dpl::ranlux24_base_vec<16>, 70, 69>,
        std::discard_block_engine<std::ranlux24_base, 70, 69>>(160, 8);
    err += tests_set_portion<oneapi::dpl::discard_block_engine<oneapi::dpl::ranlux24_base_vec<16>, 70, 69>,
        std::discard_block_engine<std::ranlux24_base, 70, 69>>(90, 9);
    err += tests_set_portion<oneapi::dpl::discard_block_engine<oneapi::dpl::ranlux24_base_vec<16>, 70, 69>,
        std::discard_block_engine<std::ranlux24_base, 70, 69>>(110, 11);
    err += tests_set_portion<oneapi::dpl::discard_block_engine<oneapi::dpl::ranlux24_base_vec<16>, 70, 69>,
        std::discard_block_engine<std::ranlux24_base, 70, 69>>(160, 16);
    err += tests_set_portion<oneapi::dpl::discard_block_engine<oneapi::dpl::ranlux24_base_vec<16>, 70, 69>,
        std::discard_block_engine<std::ranlux24_base, 70, 69>>(160, 17);
    if(err) {
        std::cout << "Test FAILED" << std::endl;
        return 1;
    }

    std::cout << "Test PASSED" << std::endl;
    return 0;
}

#endif // #if (!defined(_ONEDPL_BACKEND_SYCL) || (_ONEDPL_BACKEND_SYCL == 0))
