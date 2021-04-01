// -*- C++ -*-
//===-- uniform_real_distribution_test.cpp ---------------------------------===//
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
// Test of uniform_real_distribution - comparison with std::
// Note not all types can be compared with std:: implementation is different

#include "support/utils.h"
#include <iostream>

#if TEST_DPCPP_BACKEND_PRESENT
#include <vector>
#include <CL/sycl.hpp>
#include <random>
#include <oneapi/dpl/random>

// Engine parameters
#define a 40014u
#define c 200u
#define m 2147483563u
#define seed 777
#define eps 0.00001

template<class RealType, class UIntType>
int test(oneapi::dpl::internal::element_type_t<RealType> left, oneapi::dpl::internal::element_type_t<RealType> right, int nsamples) {

    sycl::queue queue(sycl::default_selector{});

    // memory allocation
    std::vector<oneapi::dpl::internal::element_type_t<RealType>> std_samples(nsamples);
    std::vector<oneapi::dpl::internal::element_type_t<RealType>> dpstd_samples(nsamples);

    constexpr int num_elems = oneapi::dpl::internal::type_traits_t<RealType>::num_elems == 0 ? 1 : oneapi::dpl::internal::type_traits_t<RealType>::num_elems;

    // dpstd generation
    {
        sycl::buffer<oneapi::dpl::internal::element_type_t<RealType>, 1> dpstd_buffer(dpstd_samples.data(), nsamples);

        queue.submit([&](sycl::handler &cgh) {
            auto dpstd_acc = dpstd_buffer.template get_access<sycl::access::mode::write>(cgh);

            cgh.parallel_for<>(sycl::range<1>(nsamples / num_elems),
                    [=](sycl::item<1> idx) {

                unsigned long long offset = idx.get_linear_id() * num_elems;
                oneapi::dpl::linear_congruential_engine<UIntType, a, c, m> engine(seed, offset);
                oneapi::dpl::uniform_real_distribution<RealType> distr(left, right);

                sycl::vec<oneapi::dpl::internal::element_type_t<RealType>, num_elems> res = distr(engine);
                res.store(idx.get_linear_id(), dpstd_acc.get_pointer());
            });
        });
        queue.wait();
    }

    // std generation
    std::linear_congruential_engine<oneapi::dpl::internal::element_type_t<UIntType>, a, c, m> std_engine(seed);
    std::uniform_real_distribution<oneapi::dpl::internal::element_type_t <RealType>> std_distr(left, right);

    for(int i = 0; i < nsamples; ++i)
        std_samples[i] = std_distr(std_engine);

    // comparison
    int err = 0;
    for(int i = 0; i < nsamples; ++i) {
        if(fabs(std_samples[i] - dpstd_samples[i]) > eps) {
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

template<class RealType, class UIntType>
int test_portion(oneapi::dpl::internal::element_type_t<RealType> left, oneapi::dpl::internal::element_type_t<RealType> right,
    int nsamples, unsigned int part) {

    sycl::queue queue(sycl::default_selector{});

    // memory allocation
    std::vector<oneapi::dpl::internal::element_type_t<RealType>> std_samples(nsamples);
    std::vector<oneapi::dpl::internal::element_type_t<RealType>> dpstd_samples(nsamples);
    constexpr int num_elems = oneapi::dpl::internal::type_traits_t<RealType>::num_elems == 0 ? 1 : oneapi::dpl::internal::type_traits_t<RealType>::num_elems;
    int n_elems = (part >= num_elems) ? num_elems : part;

    // dpstd generation
    {
        sycl::buffer<oneapi::dpl::internal::element_type_t<RealType>, 1> dpstd_buffer(dpstd_samples.data(), nsamples);

        queue.submit([&](sycl::handler &cgh) {
            auto dpstd_acc = dpstd_buffer.template get_access<sycl::access::mode::write>(cgh);

            cgh.parallel_for<>(sycl::range<1>(nsamples / n_elems),
                    [=](sycl::item<1> idx) {

                unsigned long long offset = idx.get_linear_id() * n_elems;
                oneapi::dpl::linear_congruential_engine<UIntType, a, c, m> engine(seed, offset);
                oneapi::dpl::uniform_real_distribution<RealType> distr(left, right);

                sycl::vec<oneapi::dpl::internal::element_type_t<RealType>, num_elems> res = distr(engine, part);
                for(int i = 0; i < n_elems; ++i)
                    dpstd_acc.get_pointer()[offset + i] = res[i];
            });
        });
        queue.wait_and_throw();
    }

    // std generation
    std::linear_congruential_engine<oneapi::dpl::internal::element_type_t<UIntType>, a, c, m> std_engine(seed);
    std::uniform_real_distribution<oneapi::dpl::internal::element_type_t <RealType>> std_distr(left, right);

    for(int i = 0; i < nsamples; ++i) {
        std_samples[i] = std_distr(std_engine);
    }

    // comparison
    int err = 0;
    for(int i = 0; i < nsamples; ++i) {
        if (fabs(std_samples[i] - dpstd_samples[i]) > eps) {
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

template<class RealType, class UIntType>
int tests_set(int nsamples) {
    constexpr int nparams = 2;
    float left_array [nparams] = {0.0, -10.0};
    float right_array [nparams] = {1.0, 10.0};

    int err;
    // Test for all non-zero parameters
    for(int i = 0; i < nparams; ++i) {
        std::cout << "uniform_real_distribution test<type>, left = " << left_array[i] << ", right = " << right_array[i] <<
        ", nsamples  = " << nsamples;
        err = test<RealType, UIntType>(left_array[i], right_array[i], nsamples);
        if(err) {
            return 1;
        }
    }

    return 0;
}

template<class RealType, class UIntType>
int tests_set_portion(int nsamples, unsigned int part) {
    constexpr int nparams = 2;
    float left_array [nparams] = {0.0, -10.0};
    float right_array [nparams] = {1.0, 10.0};

    int err;
    // Test for all non-zero parameters
    for(int i = 0; i < nparams; ++i) {
        std::cout << "uniform_real_distribution test<type>, left = " << left_array[i] << ", right = " << right_array[i] <<
        ", nsamples = " << nsamples << ", part = " << part;
        err = test_portion<RealType, UIntType>(left_array[i], right_array[i], nsamples, part);
        if(err) {
            return 1;
        }
    }
    return 0;
}

#endif // TEST_DPCPP_BACKEND_PRESENT

int main() {

#if TEST_DPCPP_BACKEND_PRESENT

    constexpr int nsamples = 100;
    int err;

    // testing float and std::uint32_t
    std::cout << "-----------------------------" << std::endl;
    std::cout << "float, std::uint32_t type" << std::endl;
    std::cout << "-----------------------------" << std::endl;
    err = tests_set<float, std::uint32_t>(nsamples);
    err += tests_set<float, sycl::vec<std::uint32_t, 16>>(nsamples);
    EXPECT_TRUE(!err, "Test FAILED");

    // testing float and std::uint64_t
    std::cout << "-----------------------------" << std::endl;
    std::cout << "float, std::uint64_t type" << std::endl;
    std::cout << "-----------------------------" << std::endl;
    err = tests_set<float, std::uint64_t>(nsamples);
    err += tests_set<float, sycl::vec<std::uint64_t, 16>>(nsamples);
    EXPECT_TRUE(!err, "Test FAILED");

    // testing sycl::vec<float, 16> and std::uint32_t
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "sycl::vec<float, 16>, std::uint32_t type" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    err = tests_set<sycl::vec<float, 16>, std::uint32_t>(160);
    err += tests_set_portion<sycl::vec<float, 16>, std::uint32_t>(160, 1);
    err += tests_set_portion<sycl::vec<float, 16>, std::uint32_t>(100, 5);
    err += tests_set_portion<sycl::vec<float, 16>, std::uint32_t>(160, 16);
    err += tests_set_portion<sycl::vec<float, 16>, std::uint32_t>(160, 17);
    EXPECT_TRUE(!err, "Test FAILED");

    // testing sycl::vec<float, 16> and sycl::vec<uint32_t, 16>
    std::cout << "--------------------------------------------------" << std::endl;
    std::cout << "sycl::vec<float, 16>, sycl::vec<uint32_t, 16> type" << std::endl;
    std::cout << "--------------------------------------------------" << std::endl;
    err = tests_set<sycl::vec<float, 16>, sycl::vec<uint32_t, 16>>(160);
    err += tests_set_portion<sycl::vec<float, 16>, sycl::vec<uint32_t, 16>>(160, 1);
    err += tests_set_portion<sycl::vec<float, 16>, sycl::vec<uint32_t, 16>>(100, 5);
    err += tests_set_portion<sycl::vec<float, 16>, sycl::vec<uint32_t, 16>>(160, 16);
    err += tests_set_portion<sycl::vec<float, 16>, sycl::vec<uint32_t, 16>>(160, 17);
    EXPECT_TRUE(!err, "Test FAILED");

    // testing sycl::vec<float, 8> and sycl::vec<uint32_t, 16>
    std::cout << "-------------------------------------------------" << std::endl;
    std::cout << "sycl::vec<float, 8>, sycl::vec<uint32_t, 16> type" << std::endl;
    std::cout << "-------------------------------------------------" << std::endl;
    err = tests_set<sycl::vec<float, 8>, sycl::vec<uint32_t, 16>>(160);
    err += tests_set_portion<sycl::vec<float, 8>, sycl::vec<uint32_t, 16>>(160, 1);
    err += tests_set_portion<sycl::vec<float, 8>, sycl::vec<uint32_t, 16>>(99, 3);
    err += tests_set_portion<sycl::vec<float, 8>, sycl::vec<uint32_t, 16>>(80, 8);
    err += tests_set_portion<sycl::vec<float, 8>, sycl::vec<uint32_t, 16>>(80, 9);
    EXPECT_TRUE(!err, "Test FAILED");

    // testing sycl::vec<float, 3> and sycl::vec<uint32_t, 16>
    std::cout << "-------------------------------------------------" << std::endl;
    std::cout << "sycl::vec<float, 3>, sycl::vec<uint32_t, 16> type" << std::endl;
    std::cout << "-------------------------------------------------" << std::endl;
    err = tests_set<sycl::vec<float, 3>, sycl::vec<uint32_t, 16>>(99);
    err += tests_set_portion<sycl::vec<float, 3>, sycl::vec<uint32_t, 16>>(160, 1);
    err += tests_set_portion<sycl::vec<float, 3>, sycl::vec<uint32_t, 16>>(100, 2);
    err += tests_set_portion<sycl::vec<float, 3>, sycl::vec<uint32_t, 16>>(99, 3);
    err += tests_set_portion<sycl::vec<float, 3>, sycl::vec<uint32_t, 16>>(99, 4);
    EXPECT_TRUE(!err, "Test FAILED");

#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
