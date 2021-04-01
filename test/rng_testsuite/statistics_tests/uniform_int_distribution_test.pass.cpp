// -*- C++ -*-
//===-- uniform_int_distribution_test.cpp ----------------------------------===//
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
// Test of uniform_int_distribution - comparison with std::
// Note not all types can be compared with std:: implementation is different

#include "support/utils.h"
#include <iostream>

#if TEST_DPCPP_BACKEND_PRESENT
#include <vector>
#include <CL/sycl.hpp>
#include <random>
#include <oneapi/dpl/random>

// Engine parameters
constexpr auto a = 40014u;
constexpr auto c = 200u;
constexpr auto m = 2147483647u;
constexpr auto seed = 777;
constexpr auto eps = 1;

template<typename IntType>
std::int32_t statistics_check(int nsamples, IntType left, IntType right,
    const std::vector<IntType>& dpstd_samples)
{
    // theoretical moments
    double tM = (left + right - 1.0) / 2.0;
    double tD = ((right - left) * (right - left) - 1.0) / 12.0;
    double tQ = (((right - left) * (right - left)) * ((1.0 / 80.0) * (right - left) * (right - left) -
        (1.0 / 24.0))) + (7.0 / 240.0);

    // sample moments
    double sum = 0.0;
    double sum2 = 0.0;
    for(std::int32_t i = 0; i < nsamples; i++) {
        sum += static_cast<double>(dpstd_samples[i]);
        sum2 += static_cast<double>(dpstd_samples[i] * dpstd_samples[i]);
    }
    double sM = sum / nsamples;
    double sD = sum2 / nsamples -  sM * sM;

    // comparison of theoretical and sample moments
    double tD2 = tD * tD;
    double s = ( (tQ - tD2) / nsamples) - (2 * (tQ - 2.0 * tD2) / (nsamples * nsamples)) +
        ((tQ - 3.0 * tD2) / (nsamples * nsamples * nsamples));

    double DeltaM = (tM - sM) / sqrt(tD / nsamples);
    double DeltaD = (tD - sD) / sqrt(s);

    if(fabs(DeltaM) > 3.0 || fabs(DeltaD) > 3.0) {
        std::cout << "Error: sample moments (mean= " << sM << ", variance= " << sD << ") disagree with theory (mean=" << tM << ", variance= " << tD << "). ";
        return 1;
    }

    return 0;
}

template<class IntType, class UIntType>
int test(oneapi::dpl::internal::element_type_t<IntType> left, oneapi::dpl::internal::element_type_t<IntType> right, int nsamples) {

    sycl::queue queue(sycl::default_selector{});

    // memory allocation
    std::vector<oneapi::dpl::internal::element_type_t<IntType>> std_samples(nsamples);
    std::vector<oneapi::dpl::internal::element_type_t<IntType>> dpstd_samples(nsamples);

    constexpr std::int32_t num_elems = oneapi::dpl::internal::type_traits_t<IntType>::num_elems == 0 ? 1 :
        oneapi::dpl::internal::type_traits_t<IntType>::num_elems;

    // dpstd generation
    {
        sycl::buffer<oneapi::dpl::internal::element_type_t<IntType>, 1> dpstd_buffer(dpstd_samples.data(), nsamples);

        queue.submit([&](sycl::handler &cgh) {
            auto dpstd_acc = dpstd_buffer.template get_access<sycl::access::mode::write>(cgh);

            cgh.parallel_for<>(sycl::range<1>(nsamples / num_elems),
                    [=](sycl::item<1> idx) {

                unsigned long long offset = idx.get_linear_id() * num_elems;
                oneapi::dpl::linear_congruential_engine<UIntType, a, c, m> engine(seed, offset);
                oneapi::dpl::uniform_int_distribution<IntType> distr(left, right);

                sycl::vec<oneapi::dpl::internal::element_type_t<IntType>, num_elems> res = distr(engine);
                res.store(idx.get_linear_id(), dpstd_acc.get_pointer());
            });
        });
        queue.wait();
    }

    // std generation
    std::linear_congruential_engine<std::uint32_t, a, c, m> std_engine(seed);
    std::uniform_real_distribution<float> std_distr(static_cast<float>(left), static_cast<float>(right + 1));

    for(int i = 0; i < nsamples; ++i)
        std_samples[i] = static_cast<std::int32_t>(std_distr(std_engine));

    // comparison
    int err = 0;
    for(int i = 0; i < nsamples; ++i) {
        if(abs(std_samples[i] - dpstd_samples[i]) > eps) {
            std::cout << "\nError: std_sample[" << i << "] = " << std_samples[i] << ", dpstd_samples[" << i << "] = " << dpstd_samples[i];
            err++;
        }
    }

    // statistics check
    err+=statistics_check(nsamples, left, right, dpstd_samples);

    if(err) {
        std::cout << "\tFailed" << std::endl;
    }
    else {
        std::cout << "\tPassed" << std::endl;
    }

    return err;
}

template<class IntType, class UIntType>
int test_portion(oneapi::dpl::internal::element_type_t<IntType> left, oneapi::dpl::internal::element_type_t<IntType> right,
    int nsamples, unsigned int part) {

    sycl::queue queue(sycl::default_selector{});

    // memory allocation
    std::vector<oneapi::dpl::internal::element_type_t<IntType>> std_samples(nsamples);
    std::vector<oneapi::dpl::internal::element_type_t<IntType>> dpstd_samples(nsamples);
    constexpr unsigned int num_elems = oneapi::dpl::internal::type_traits_t<IntType>::num_elems == 0 ? 1 : oneapi::dpl::internal::type_traits_t<IntType>::num_elems;
    int n_elems = (part >= num_elems) ? num_elems : part;

    // dpstd generation
    {
        sycl::buffer<oneapi::dpl::internal::element_type_t<IntType>, 1> dpstd_buffer(dpstd_samples.data(), nsamples);

        queue.submit([&](sycl::handler &cgh) {
            auto dpstd_acc = dpstd_buffer.template get_access<sycl::access::mode::write>(cgh);

            cgh.parallel_for<>(sycl::range<1>(nsamples / n_elems),
                    [=](sycl::item<1> idx) {

                unsigned long long offset = idx.get_linear_id() * n_elems;
                oneapi::dpl::linear_congruential_engine<UIntType, a, c, m> engine(seed, offset);
                oneapi::dpl::uniform_int_distribution<IntType> distr(left, right);

                sycl::vec<oneapi::dpl::internal::element_type_t<IntType>, num_elems> res = distr(engine, part);
                for(int i = 0; i < n_elems; ++i)
                    dpstd_acc.get_pointer()[offset + i] = res[i];
            });
        });
        queue.wait_and_throw();
    }

    // std generation
    std::linear_congruential_engine<oneapi::dpl::internal::element_type_t<UIntType>, a, c, m> std_engine(seed);
    std::uniform_real_distribution<float> std_distr(static_cast<float>(left), static_cast<float>(right + 1));

    for(int i = 0; i < nsamples; ++i)
        std_samples[i] = static_cast<std::int32_t>(std_distr(std_engine));

    // comparison
    int err = 0;
    for(int i = 0; i < nsamples; ++i) {
        if(abs(std_samples[i] - dpstd_samples[i]) > eps) {
            std::cout << "\nError: std_sample[" << i << "] = " << std_samples[i] << ", dpstd_samples[" << i << "] = " << dpstd_samples[i];
            err++;
        }
    }

    // statistics check
    err+=statistics_check(nsamples, left, right, dpstd_samples);

    if(err) {
        std::cout << "\tFailed" << std::endl;
    }
    else {
        std::cout << "\tPassed" << std::endl;
    }

    return err;
}

template<class IntType, class UIntType>
int tests_set(int nsamples) {
    constexpr int nparams = 2;

    int left_array [nparams] = {0, -10};
    int right_array [nparams] = {1000, 10};

    int err;
    // Test for all non-zero parameters
    for(int i = 0; i < nparams; ++i) {
        std::cout << "uniform_int_distribution test<type>, left = " << left_array[i] << ", right = " << right_array[i] <<
        ", nsamples = " << nsamples;
        err = test<IntType, UIntType>(left_array[i], right_array[i], nsamples);
        if(err) {
            return 1;
        }
    }

    return 0;
}

template<class IntType, class UIntType>
int tests_set_portion(int nsamples, unsigned int part) {
    constexpr int nparams = 2;

    int left_array [nparams] = {0, -10};
    int right_array [nparams] = {1000, 10};

    int err;
    // Test for all non-zero parameters
    for(int i = 0; i < nparams; ++i) {
        std::cout << "uniform_int_distribution test<type>, left = " << left_array[i] << ", right = " << right_array[i] <<
        ", nsamples = " << nsamples <<  ", part = " << part;
        err = test_portion<IntType, UIntType>(left_array[i], right_array[i], nsamples, part);
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

    // testing std::int32_t and std::uint32_t
    std::cout << "-------------------------------" << std::endl;
    std::cout << "std::int32_t, std::uint32_t type" << std::endl;
    std::cout << "-------------------------------" << std::endl;
    err = tests_set<std::int32_t, std::uint32_t>(nsamples);
    err += tests_set<std::int32_t, sycl::vec<std::uint32_t, 16>>(nsamples);
    EXPECT_TRUE(!err, "Test FAILED");

    // testing sycl::vec<std::int32_t, 16> and std::uint32_t
    std::cout << "------------------------------------------" << std::endl;
    std::cout << "sycl::vec<std::int32_t, 16>, std::uint32_t" << std::endl;
    std::cout << "------------------------------------------" << std::endl;
    err = tests_set<sycl::vec<std::int32_t, 16>, std::uint32_t>(160);
    err += tests_set_portion<sycl::vec<std::int32_t, 16>, std::uint32_t>(160, 1);
    err += tests_set_portion<sycl::vec<std::int32_t, 16>, std::uint32_t>(100, 5);
    err += tests_set_portion<sycl::vec<std::int32_t, 16>, std::uint32_t>(160, 16);
    err += tests_set_portion<sycl::vec<std::int32_t, 16>, std::uint32_t>(160, 17);
    EXPECT_TRUE(!err, "Test FAILED");

    // testing sycl::vec<std::int32_t, 16> and sycl::vec<uint32_t, 16>
    std::cout << "----------------------------------------------------" << std::endl;
    std::cout << "sycl::vec<std::int32_t, 16>, sycl::vec<uint32_t, 16>" << std::endl;
    std::cout << "----------------------------------------------------" << std::endl;
    err = tests_set<sycl::vec<std::int32_t, 16>, sycl::vec<uint32_t, 16>>(160);
    err += tests_set_portion<sycl::vec<std::int32_t, 16>, sycl::vec<uint32_t, 16>>(160, 1);
    err += tests_set_portion<sycl::vec<std::int32_t, 16>, sycl::vec<uint32_t, 16>>(100, 5);
    err += tests_set_portion<sycl::vec<std::int32_t, 16>, sycl::vec<uint32_t, 16>>(160, 16);
    err += tests_set_portion<sycl::vec<std::int32_t, 16>, sycl::vec<uint32_t, 16>>(160, 17);
    EXPECT_TRUE(!err, "Test FAILED");

    // testing sycl::vec<std::int32_t, 8> and sycl::vec<uint32_t, 16>
    std::cout << "---------------------------------------------------" << std::endl;
    std::cout << "sycl::vec<std::int32_t, 8>, sycl::vec<uint32_t, 16>" << std::endl;
    std::cout << "---------------------------------------------------" << std::endl;
    err = tests_set<sycl::vec<std::int32_t, 8>, sycl::vec<uint32_t, 16>>(160);
    err += tests_set_portion<sycl::vec<std::int32_t, 8>, sycl::vec<uint32_t, 16>>(160, 1);
    err += tests_set_portion<sycl::vec<std::int32_t, 8>, sycl::vec<uint32_t, 16>>(99, 3);
    err += tests_set_portion<sycl::vec<std::int32_t, 8>, sycl::vec<uint32_t, 16>>(80, 8);
    err += tests_set_portion<sycl::vec<std::int32_t, 8>, sycl::vec<uint32_t, 16>>(80, 9);
    EXPECT_TRUE(!err, "Test FAILED");

    // testing sycl::vec<std::int32_t, 3> and sycl::vec<uint32_t, 16>
    std::cout << "---------------------------------------------------" << std::endl;
    std::cout << "sycl::vec<std::int32_t, 3>, sycl::vec<uint32_t, 16>" << std::endl;
    std::cout << "---------------------------------------------------" << std::endl;
    err = tests_set<sycl::vec<std::int32_t, 3>, sycl::vec<uint32_t, 16>>(99);
    err += tests_set_portion<sycl::vec<std::int32_t, 3>, sycl::vec<uint32_t, 16>>(160, 1);
    err += tests_set_portion<sycl::vec<std::int32_t, 3>, sycl::vec<uint32_t, 16>>(100, 2);
    err += tests_set_portion<sycl::vec<std::int32_t, 3>, sycl::vec<uint32_t, 16>>(99, 3);
    err += tests_set_portion<sycl::vec<std::int32_t, 3>, sycl::vec<uint32_t, 16>>(99, 4);
    EXPECT_TRUE(!err, "Test FAILED");

#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
