// -*- C++ -*-
//===-- lognormal_distribution_test.cpp ---------------------------------------===//
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
//===-------------------------------------------------------------------------===//
//
// Abstract:
//
// Test of lognormal_distribution - check statistical properties of the distribution

#include "support/utils.h"
#include <iostream>

#if TEST_DPCPP_BACKEND_PRESENT && TEST_UNNAMED_LAMBDAS
#include <CL/sycl.hpp>
#include <random>
#include <limits>
#include <oneapi/dpl/random>
#include <math.h>
#include "statistics_common.h"

// Engine parameters
constexpr auto a = 40014u;
constexpr auto c = 200u;
constexpr auto m = 2147483563u;
constexpr auto seed = 777;

template<typename ScalarRealType>
int statistics_check(int nsamples, ScalarRealType mean, ScalarRealType stddev,
    const std::vector<ScalarRealType>& samples) {
    // theoretical moments
    double tM = exp(mean + stddev * stddev / 2);
    double tD = (exp(stddev * stddev) - 1) * exp(2 * mean + stddev * stddev);
    double tQ = (exp(4 * stddev * stddev) + 2 * exp(3 * stddev * stddev) + 3 * exp(2 * stddev * stddev) - 3) * tD * tD;

    return compare_moments(nsamples, samples, tM, tD, tQ);
}

template<class RealType, class UIntType>
int test(oneapi::dpl::internal::element_type_t<RealType> mean, oneapi::dpl::internal::element_type_t<RealType> stddev, int nsamples) {

    sycl::queue queue;

    // memory allocation
    std::vector<oneapi::dpl::internal::element_type_t<RealType>> samples(nsamples);

    constexpr int num_elems = oneapi::dpl::internal::type_traits_t<RealType>::num_elems == 0 ? 1 : oneapi::dpl::internal::type_traits_t<RealType>::num_elems;

    // generation
    {
        sycl::buffer<oneapi::dpl::internal::element_type_t<RealType>, 1> buffer(samples.data(), nsamples);

        queue.submit([&](sycl::handler &cgh) {
            sycl::accessor acc(buffer, cgh, sycl::write_only);

            cgh.parallel_for<>(sycl::range<1>(nsamples / num_elems),
                    [=](sycl::item<1> idx) {

                unsigned long long offset = idx.get_linear_id() * num_elems;
                oneapi::dpl::linear_congruential_engine<UIntType, a, c, m> engine(seed, offset);
                oneapi::dpl::lognormal_distribution<RealType> distr(mean, stddev);

                sycl::vec<oneapi::dpl::internal::element_type_t<RealType>, num_elems> res = distr(engine);
                res.store(idx.get_linear_id(), acc.get_pointer());
            });
        });
        queue.wait();
    }

    // statistics check
    int err = statistics_check(nsamples, mean, stddev, samples);

    if(err) {
        std::cout << "\tFailed" << std::endl;
    }
    else {
        std::cout << "\tPassed" << std::endl;
    }

    return err;
}

template<class RealType, class UIntType>
int test_portion(oneapi::dpl::internal::element_type_t<RealType> mean, oneapi::dpl::internal::element_type_t<RealType> stddev,
    int nsamples, unsigned int part) {

    sycl::queue queue;

    // memory allocation
    std::vector<oneapi::dpl::internal::element_type_t<RealType>> samples(nsamples);
    constexpr unsigned int num_elems = oneapi::dpl::internal::type_traits_t<RealType>::num_elems == 0 ? 1 : oneapi::dpl::internal::type_traits_t<RealType>::num_elems;
    int n_elems = (part >= num_elems) ? num_elems : part;

    // generation
    {
        sycl::buffer<oneapi::dpl::internal::element_type_t<RealType>, 1> buffer(samples.data(), nsamples);

        queue.submit([&](sycl::handler &cgh) {
            sycl::accessor acc(buffer, cgh, sycl::write_only);

            cgh.parallel_for<>(sycl::range<1>(nsamples / n_elems),
                    [=](sycl::item<1> idx) {

                unsigned long long offset = idx.get_linear_id() * num_elems;
                oneapi::dpl::linear_congruential_engine<UIntType, a, c, m> engine(seed, offset);
                oneapi::dpl::lognormal_distribution<RealType> distr(mean, stddev);

                sycl::vec<oneapi::dpl::internal::element_type_t<RealType>, num_elems> res = distr(engine, part);
                for(int i = 0; i < n_elems; ++i)
                    acc.get_pointer()[idx.get_linear_id() * n_elems + i] = res[i];
            });
        });
        queue.wait_and_throw();
    }

    // statistics check
    int err = statistics_check(nsamples, mean, stddev, samples);

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

    oneapi::dpl::internal::element_type_t<RealType> mean = 0.0;
    oneapi::dpl::internal::element_type_t<RealType> stddev= 1.0;

    // Test for all non-zero parameters
    std::cout << "lognormal_distribution test<type>, mean = " << mean << ", stddev = " << stddev <<
    ", nsamples = " << nsamples;
    if (test<RealType, UIntType>(mean, stddev, nsamples))
         return 1;
    return 0;
}

template<class RealType, class UIntType>
int tests_set_portion(std::int32_t nsamples, unsigned int part) {

    oneapi::dpl::internal::element_type_t<RealType> mean = 0.0;
    oneapi::dpl::internal::element_type_t<RealType> stddev = 1.0;

    // Test for all non-zero parameters
    std::cout << "lognormal_distribution test<type>, mean = " << mean << ", stddev = " << stddev <<
    ", nsamples = " << nsamples << ", part = "<< part;
    if(test_portion<RealType, UIntType>(mean, stddev, nsamples, part))
        return 1;
    return 0;
}

#endif // TEST_DPCPP_BACKEND_PRESENT && TEST_UNNAMED_LAMBDAS

int main() {

#if TEST_DPCPP_BACKEND_PRESENT && TEST_UNNAMED_LAMBDAS

    constexpr int nsamples = 100;
    int err = 0;

    // testing float and std::uint32_t ... sycl::vec<std::uint32_t, 16>
    std::cout << "--------------------------------------------------------" << std::endl;
    std::cout << "float, std::uint32_t ... sycl::vec<std::uint32_t, 16> type" << std::endl;
    std::cout << "--------------------------------------------------------" << std::endl;
    err += tests_set<float, std::uint32_t>(nsamples);
#if TEST_LONG_RUN
    err += tests_set<float, sycl::vec<std::uint32_t, 16>>(nsamples);
    err += tests_set<float, sycl::vec<std::uint32_t, 8>>(nsamples);
    err += tests_set<float, sycl::vec<std::uint32_t, 4>>(nsamples);
    err += tests_set<float, sycl::vec<std::uint32_t, 3>>(nsamples);
    err += tests_set<float, sycl::vec<std::uint32_t, 2>>(nsamples);
    err += tests_set<float, sycl::vec<std::uint32_t, 1>>(nsamples);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // testing sycl::vec<float, 1> and std::uint32_t
    std::cout << "---------------------------------------------------------------------" << std::endl;
    std::cout << "sycl::vec<float,1>, std::uint32_t> type" << std::endl;
    std::cout << "---------------------------------------------------------------------" << std::endl;
    err = tests_set<sycl::vec<float, 1>, std::uint32_t>(nsamples);
#if TEST_LONG_RUN
    err += tests_set_portion<sycl::vec<float, 1>, std::uint32_t>(100, 1);
    err += tests_set_portion<sycl::vec<float, 1>, std::uint32_t>(100, 2);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // testing sycl::vec<float, 2> and std::uint32_t, sycl::vec<std::uint32_t, 3>
    std::cout << "------------------------------------------------------------------" << std::endl;
    std::cout << "sycl::vec<float,2>, std::uint32_t, sycl::vec<std::uint32_t, 3> type" << std::endl;
    std::cout << "------------------------------------------------------------------" << std::endl;
    err = tests_set<sycl::vec<float, 2>, std::uint32_t>(nsamples);
#if TEST_LONG_RUN
    err += tests_set<sycl::vec<float, 2>, sycl::vec<std::uint32_t, 3>>(100);
    err += tests_set_portion<sycl::vec<float, 2>, std::uint32_t>(100, 1);
    err += tests_set_portion<sycl::vec<float, 2>, std::uint32_t>(100, 3);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // testing sycl::vec<float, 3> and std::uint32_t, sycl::vec<std::uint32_t, 2>, sycl::vec<std::uint32_t, 4>
    std::cout << "----------------------------------------------------------------------------------------------" << std::endl;
    std::cout << "sycl::vec<float,3>, std::uint32_t, sycl::vec<std::uint32_t, 2>, sycl::vec<std::uint32_t, 4> type" << std::endl;
    std::cout << "----------------------------------------------------------------------------------------------" << std::endl;
    err = tests_set<sycl::vec<float, 3>, std::uint32_t>(99);
#if TEST_LONG_RUN
    err += tests_set<sycl::vec<float, 3>, sycl::vec<std::uint32_t, 2>>(100);
    err += tests_set<sycl::vec<float, 3>, sycl::vec<std::uint32_t, 4>>(100);
    err += tests_set_portion<sycl::vec<float, 3>, std::uint32_t>(99, 1);
    err += tests_set_portion<sycl::vec<float, 3>, std::uint32_t>(99, 4);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // testing sycl::vec<float, 4> and std::uint32_t, sycl::vec<std::uint32_t, 3>
    std::cout << "-------------------------------------------------------------------" << std::endl;
    std::cout << "sycl::vec<float,4>, std::uint32_t, sycl::vec<std::uint32_t, 3> type" << std::endl;
    std::cout << "-------------------------------------------------------------------" << std::endl;
    err = tests_set<sycl::vec<float, 4>, std::uint32_t>(100);
#if TEST_LONG_RUN
    err += tests_set<sycl::vec<float, 4>, sycl::vec<std::uint32_t, 3>>(100);
    err += tests_set_portion<sycl::vec<float, 4>, std::uint32_t>(100, 1);
    err += tests_set_portion<sycl::vec<float, 4>, std::uint32_t>(100, 5);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // testing sycl::vec<float, 8> and std::uint32_t
    std::cout << "-------------------------------------" << std::endl;
    std::cout << "sycl::vec<float,8>, std::uint32_t type" << std::endl;
    std::cout << "-------------------------------------" << std::endl;
    err = tests_set<sycl::vec<float, 8>, std::uint32_t>(160);
#if TEST_LONG_RUN
    err += tests_set_portion<sycl::vec<float, 8>, std::uint32_t>(160, 1);
    err += tests_set_portion<sycl::vec<float, 8>, std::uint32_t>(160, 5);
    err += tests_set_portion<sycl::vec<float, 8>, std::uint32_t>(160, 9);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // testing sycl::vec<float, 16> and std::uint32_t
    std::cout << "--------------------------------------" << std::endl;
    std::cout << "sycl::vec<float,16>, std::uint32_t type" << std::endl;
    std::cout << "--------------------------------------" << std::endl;
    err = tests_set<sycl::vec<float, 16>, std::uint32_t>(160);
#if TEST_LONG_RUN
    err += tests_set_portion<sycl::vec<float, 16>, std::uint32_t>(160, 1);
    err += tests_set_portion<sycl::vec<float, 16>, std::uint32_t>(140, 7);
    err += tests_set_portion<sycl::vec<float, 16>, std::uint32_t>(160, 17);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");
 
// testing double and std::uint32_t ... sycl::vec<std::uint32_t, 16>
    std::cout << "--------------------------------------------------------" << std::endl;
    std::cout << "double, std::uint32_t ... sycl::vec<std::uint32_t, 16> type" << std::endl;
    std::cout << "--------------------------------------------------------" << std::endl;
    err += tests_set<double, std::uint32_t>(nsamples);
#if TEST_LONG_RUN
    err += tests_set<double, sycl::vec<std::uint32_t, 16>>(nsamples);
    err += tests_set<double, sycl::vec<std::uint32_t, 8>>(nsamples);
    err += tests_set<double, sycl::vec<std::uint32_t, 4>>(nsamples);
    err += tests_set<double, sycl::vec<std::uint32_t, 3>>(nsamples);
    err += tests_set<double, sycl::vec<std::uint32_t, 2>>(nsamples);
    err += tests_set<double, sycl::vec<std::uint32_t, 1>>(nsamples);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // testing sycl::vec<double, 1> and std::uint32_t
    std::cout << "---------------------------------------------------------------------" << std::endl;
    std::cout << "sycl::vec<double,1>, std::uint32_t type" << std::endl;
    std::cout << "---------------------------------------------------------------------" << std::endl;
    err = tests_set<sycl::vec<double, 1>, std::uint32_t>(nsamples);
#if TEST_LONG_RUN
    err += tests_set_portion<sycl::vec<double, 1>, std::uint32_t>(100, 1);
    err += tests_set_portion<sycl::vec<double, 1>, std::uint32_t>(100, 2);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // testing sycl::vec<double, 2> and std::uint32_t, sycl::vec<std::uint32_t, 3>
    std::cout << "------------------------------------------------------------------" << std::endl;
    std::cout << "sycl::vec<double,2>, std::uint32_t, sycl::vec<std::uint32_t, 3> type" << std::endl;
    std::cout << "------------------------------------------------------------------" << std::endl;
    err = tests_set<sycl::vec<double, 2>, std::uint32_t>(nsamples);
#if TEST_LONG_RUN
    err += tests_set<sycl::vec<double, 2>, sycl::vec<std::uint32_t, 3>>(100);
    err += tests_set_portion<sycl::vec<double, 2>, std::uint32_t>(100, 1);
    err += tests_set_portion<sycl::vec<double, 2>, std::uint32_t>(100, 3);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // testing sycl::vec<double, 3> and std::uint32_t, sycl::vec<std::uint32_t, 2>, sycl::vec<std::uint32_t, 4>
    std::cout << "----------------------------------------------------------------------------------------------" << std::endl;
    std::cout << "sycl::vec<double,3>, std::uint32_t, sycl::vec<std::uint32_t, 2>, sycl::vec<std::uint32_t, 4> type" << std::endl;
    std::cout << "----------------------------------------------------------------------------------------------" << std::endl;
    err = tests_set<sycl::vec<double, 3>, std::uint32_t>(99);
#if TEST_LONG_RUN
    err += tests_set<sycl::vec<double, 3>, sycl::vec<std::uint32_t, 2>>(100);
    err += tests_set<sycl::vec<double, 3>, sycl::vec<std::uint32_t, 4>>(100);
    err += tests_set_portion<sycl::vec<double, 3>, std::uint32_t>(99, 1);
    err += tests_set_portion<sycl::vec<double, 3>, std::uint32_t>(99, 4);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // testing sycl::vec<double, 4> and std::uint32_t, sycl::vec<std::uint32_t, 3>
    std::cout << "-------------------------------------------------------------------" << std::endl;
    std::cout << "sycl::vec<double,4>, std::uint32_t, sycl::vec<std::uint32_t, 3> type" << std::endl;
    std::cout << "-------------------------------------------------------------------" << std::endl;
    err = tests_set<sycl::vec<double, 4>, std::uint32_t>(100);
#if TEST_LONG_RUN
    err += tests_set<sycl::vec<double, 4>, sycl::vec<std::uint32_t, 3>>(100);
    err += tests_set_portion<sycl::vec<double, 4>, std::uint32_t>(100, 1);
    err += tests_set_portion<sycl::vec<double, 4>, std::uint32_t>(100, 5);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // testing sycl::vec<double, 8> and std::uint32_t
    std::cout << "-------------------------------------" << std::endl;
    std::cout << "sycl::vec<double,8>, std::uint32_t type" << std::endl;
    std::cout << "-------------------------------------" << std::endl;
    err = tests_set<sycl::vec<double, 8>, std::uint32_t>(160);
#if TEST_LONG_RUN
    err += tests_set_portion<sycl::vec<double, 8>, std::uint32_t>(160, 1);
    err += tests_set_portion<sycl::vec<double, 8>, std::uint32_t>(160, 5);
    err += tests_set_portion<sycl::vec<double, 8>, std::uint32_t>(160, 9);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // testing sycl::vec<double, 16> and std::uint32_t
    std::cout << "--------------------------------------" << std::endl;
    std::cout << "sycl::vec<double,16>, std::uint32_t type" << std::endl;
    std::cout << "--------------------------------------" << std::endl;
    err = tests_set<sycl::vec<double, 16>, std::uint32_t>(160);
#if TEST_LONG_RUN
    err += tests_set_portion<sycl::vec<double, 16>, std::uint32_t>(160, 1);
    err += tests_set_portion<sycl::vec<double, 16>, std::uint32_t>(140, 7);
    err += tests_set_portion<sycl::vec<double, 16>, std::uint32_t>(160, 17);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

#endif // TEST_DPCPP_BACKEND_PRESENT && TEST_UNNAMED_LAMBDAS

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT && TEST_UNNAMED_LAMBDAS);
}
