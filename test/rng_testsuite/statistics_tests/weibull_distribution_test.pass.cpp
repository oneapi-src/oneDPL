// -*- C++ -*-
//===-- weibull_distribution_test.cpp ---------------------------------===//
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
// Test of weibull_distribution - check statistical properties of the distribution

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

template <typename ScalarRealType>
int
statistics_check(int nsamples, ScalarRealType _a, ScalarRealType _b, const std::vector<ScalarRealType>& samples)
{
    // theoretical moments
    double G1 = sycl::tgamma(1 + 1 / _a);
    double G2 = sycl::tgamma(1 + 2 / _a);
    double G3 = sycl::tgamma(1 + 3 / _a);
    double G4 = sycl::tgamma(1 + 4 / _a);
    double tM = _b * G1;
    double tD = _b * _b * (G2 - G1 * G1);
    double tQ = _b * _b * _b * _b * ((-3) * G1 * G1 * G1 * G1 + 12 * G1 * G1 * G2 - 4 * G1 * G3 + G4 - 6 * G2 * G1 *G1);

    return compare_moments(nsamples, samples, tM, tD, tQ);
}

template <class RealType, class UIntType>
int
test(sycl::queue& queue, oneapi::dpl::internal::element_type_t<RealType> _a,  oneapi::dpl::internal::element_type_t<RealType> _b, int nsamples)
{
    // memory allocation
    std::vector<oneapi::dpl::internal::element_type_t<RealType>> samples(nsamples);

    constexpr int num_elems = oneapi::dpl::internal::type_traits_t<RealType>::num_elems == 0
                                  ? 1
                                  : oneapi::dpl::internal::type_traits_t<RealType>::num_elems;

    // generation
    {
        sycl::buffer<oneapi::dpl::internal::element_type_t<RealType>, 1> buffer(samples.data(), nsamples);

        queue.submit([&](sycl::handler& cgh) {
            sycl::accessor acc(buffer, cgh, sycl::write_only);

            cgh.parallel_for<>(sycl::range<1>(nsamples / num_elems), [=](sycl::item<1> idx) {
                unsigned long long offset = idx.get_linear_id() * num_elems;
                oneapi::dpl::linear_congruential_engine<UIntType, a, c, m> engine(seed, offset);
                oneapi::dpl::weibull_distribution<RealType> distr(_a, _b);

                sycl::vec<oneapi::dpl::internal::element_type_t<RealType>, num_elems> res = distr(engine);
                res.store(idx.get_linear_id(), acc.get_pointer());
            });
        });
    }

    // statistics check
    int err = statistics_check(nsamples, _a, _b, samples);

    if (err)
    {
        std::cout << "\tFailed" << std::endl;
    }
    else
    {
        std::cout << "\tPassed" << std::endl;
    }

    return err;
}

template <class RealType, class UIntType>
int
test_portion(sycl::queue& queue, oneapi::dpl::internal::element_type_t<RealType> _a,  oneapi::dpl::internal::element_type_t<RealType> _b, 
                    int nsamples, unsigned int part)
{
    // memory allocation
    std::vector<oneapi::dpl::internal::element_type_t<RealType>> samples(nsamples);
    constexpr unsigned int num_elems = oneapi::dpl::internal::type_traits_t<RealType>::num_elems == 0
                                           ? 1
                                           : oneapi::dpl::internal::type_traits_t<RealType>::num_elems;
    int n_elems = (part >= num_elems) ? num_elems : part;

    // generation
    {
        sycl::buffer<oneapi::dpl::internal::element_type_t<RealType>, 1> buffer(samples.data(), nsamples);

        queue.submit([&](sycl::handler& cgh) {
            sycl::accessor acc(buffer, cgh, sycl::write_only);

            cgh.parallel_for<>(sycl::range<1>(nsamples / n_elems), [=](sycl::item<1> idx) {
                unsigned long long offset = idx.get_linear_id() * n_elems;
                oneapi::dpl::linear_congruential_engine<UIntType, a, c, m> engine(seed, offset);
                oneapi::dpl::weibull_distribution<RealType> distr(_a, _b);

                sycl::vec<oneapi::dpl::internal::element_type_t<RealType>, num_elems> res = distr(engine, part);
                for (int i = 0; i < n_elems; ++i)
                    acc.get_pointer()[offset + i] = res[i];
            });
        });
        queue.wait_and_throw();
    }

    // statistics check
    int err = statistics_check(nsamples, _a, _b, samples);

    if (err)
    {
        std::cout << "\tFailed" << std::endl;
    }
    else
    {
        std::cout << "\tPassed" << std::endl;
    }

    return err;
}

template <class RealType, class UIntType>
int
tests_set(sycl::queue& queue, int nsamples)
{
    oneapi::dpl::internal::element_type_t<RealType> a = 2.0;
    oneapi::dpl::internal::element_type_t<RealType> b = 1.0;

    // Test for all non-zero parameters
    std::cout << "weibull_distribution test<type>, a = " << a << ", b = " << b <<
    ", nsamples  = " << nsamples;
    if(test<RealType, UIntType>(queue, a, b, nsamples))
        return 1;
    return 0;
}

template <class RealType, class UIntType>
int
tests_set_portion(sycl::queue& queue, std::int32_t nsamples, unsigned int part)
{
    oneapi::dpl::internal::element_type_t<RealType> a = 2.0;
    oneapi::dpl::internal::element_type_t<RealType> b = 1.0;

    // Test for all non-zero parameters
    std::cout << "weibull_distribution test<type>, a = " << a << ", b = " << b <<
    ", nsamples = " << nsamples << ", part = " << part;
    if(test_portion<RealType, UIntType>(queue, a, b, nsamples, part))
        return 1;
    return 0;
}

#endif // TEST_DPCPP_BACKEND_PRESENT && TEST_UNNAMED_LAMBDAS

int
main()
{

#if TEST_DPCPP_BACKEND_PRESENT && TEST_UNNAMED_LAMBDAS

    sycl::queue queue;

    constexpr int nsamples = 100;
    int err = 0;

    // testing float and std::uint32_t ... sycl::vec<std::uint32_t, 3>
    std::cout << "--------------------------------------------------------" << std::endl;
    std::cout << "float, std::uint32_t ... sycl::vec<std::uint32_t, 3> type" << std::endl;
    std::cout << "--------------------------------------------------------" << std::endl;
    err += tests_set<float, std::uint32_t>(queue, nsamples);
#if TEST_LONG_RUN
    err += tests_set<float, sycl::vec<std::uint32_t, 3>>(queue, nsamples);
    err += tests_set<float, sycl::vec<std::uint32_t, 2>>(queue, nsamples);
    err += tests_set<float, sycl::vec<std::uint32_t, 1>>(queue, nsamples);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // testing sycl::vec<float, 1> and std::uint32_t
    std::cout << "-------------------------------------" << std::endl;
    std::cout << "sycl::vec<float,1>, std::uint32_t type" << std::endl;
    std::cout << "-------------------------------------" << std::endl;
    err = tests_set<sycl::vec<float, 1>, std::uint32_t>(queue, nsamples);
#if TEST_LONG_RUN
    err += tests_set_portion<sycl::vec<float, 1>, std::uint32_t>(queue, 100, 1);
    err += tests_set_portion<sycl::vec<float, 1>, std::uint32_t>(queue, 100, 2);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // testing sycl::vec<float, 2> and std::uint32_t, sycl::vec<std::uint32_t, 3>
    std::cout << "------------------------------------------------------------------" << std::endl;
    std::cout << "sycl::vec<float,2>, std::uint32_t, sycl::vec<std::uint32_t, 3> type" << std::endl;
    std::cout << "------------------------------------------------------------------" << std::endl;
    err = tests_set<sycl::vec<float, 2>, std::uint32_t>(queue, nsamples);
#if TEST_LONG_RUN
    err += tests_set<sycl::vec<float, 2>, sycl::vec<std::uint32_t, 3>>(queue, 100);
    err += tests_set_portion<sycl::vec<float, 2>, std::uint32_t>(queue, 100, 1);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // testing sycl::vec<float, 3> and std::uint32_t, sycl::vec<std::uint32_t, 2>
    std::cout << "------------------------------------------------------------------" << std::endl;
    std::cout << "sycl::vec<float,3>, std::uint32_t, sycl::vec<std::uint32_t, 3> type" << std::endl;
    std::cout << "------------------------------------------------------------------" << std::endl;
    err = tests_set<sycl::vec<float, 3>, std::uint32_t>(queue, 99);
#if TEST_LONG_RUN
    err += tests_set<sycl::vec<float, 3>, sycl::vec<std::uint32_t, 2>>(queue, 100);
    err += tests_set_portion<sycl::vec<float, 3>, std::uint32_t>(queue, 99, 1);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // testing sycl::vec<float, 4> and std::uint32_t, sycl::vec<std::uint32_t, 3>
    std::cout << "-------------------------------------------------------------------" << std::endl;
    std::cout << "sycl::vec<float,4>, std::uint32_t, sycl::vec<std::uint32_t, 3> type" << std::endl;
    std::cout << "-------------------------------------------------------------------" << std::endl;
    err = tests_set<sycl::vec<float, 4>, std::uint32_t>(queue, 100);
#if TEST_LONG_RUN
    err += tests_set<sycl::vec<float, 4>, sycl::vec<std::uint32_t, 3>>(queue, 100);
    err += tests_set_portion<sycl::vec<float, 4>, std::uint32_t>(queue, 100, 1);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // testing sycl::vec<float, 8> and std::uint32_t
    std::cout << "-------------------------------------" << std::endl;
    std::cout << "sycl::vec<float,8>, std::uint32_t type" << std::endl;
    std::cout << "-------------------------------------" << std::endl;
    err = tests_set<sycl::vec<float, 8>, std::uint32_t>(queue, 160);
#if TEST_LONG_RUN
    err += tests_set_portion<sycl::vec<float, 8>, std::uint32_t>(queue, 160, 5);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // testing sycl::vec<float, 16> and std::uint32_t
    std::cout << "--------------------------------------" << std::endl;
    std::cout << "sycl::vec<float,16>, std::uint32_t type" << std::endl;
    std::cout << "--------------------------------------" << std::endl;
    err = tests_set<sycl::vec<float, 16>, std::uint32_t>(queue, 160);
#if TEST_LONG_RUN
    err += tests_set_portion<sycl::vec<float, 16>, std::uint32_t>(queue, 140, 7);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");
 
// testing double and std::uint32_t ... sycl::vec<std::uint32_t, 3>
    std::cout << "--------------------------------------------------------" << std::endl;
    std::cout << "double, std::uint32_t ... sycl::vec<std::uint32_t, 3> type" << std::endl;
    std::cout << "--------------------------------------------------------" << std::endl;
    err += tests_set<double, std::uint32_t>(queue, nsamples);
#if TEST_LONG_RUN
    err += tests_set<double, sycl::vec<std::uint32_t, 3>>(queue, nsamples);
    err += tests_set<double, sycl::vec<std::uint32_t, 2>>(queue, nsamples);
    err += tests_set<double, sycl::vec<std::uint32_t, 1>>(queue, nsamples);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // testing sycl::vec<double, 1> and std::uint32_t
    std::cout << "--------------------------------------" << std::endl;
    std::cout << "sycl::vec<double,1>, std::uint32_t type" << std::endl;
    std::cout << "--------------------------------------" << std::endl;
    err = tests_set<sycl::vec<double, 1>, std::uint32_t>(queue, nsamples);
#if TEST_LONG_RUN
    err += tests_set_portion<sycl::vec<double, 1>, std::uint32_t>(queue, 100, 1);
    err += tests_set_portion<sycl::vec<double, 1>, std::uint32_t>(queue, 100, 2);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // testing sycl::vec<double, 2> and std::uint32_t, sycl::vec<std::uint32_t, 3>
    std::cout << "------------------------------------------------------------------" << std::endl;
    std::cout << "sycl::vec<double,2>, std::uint32_t, sycl::vec<std::uint32_t, 3> type" << std::endl;
    std::cout << "------------------------------------------------------------------" << std::endl;
    err = tests_set<sycl::vec<double, 2>, std::uint32_t>(queue, nsamples);
#if TEST_LONG_RUN
    err += tests_set<sycl::vec<double, 2>, sycl::vec<std::uint32_t, 3>>(queue, 100);
    err += tests_set_portion<sycl::vec<double, 2>, std::uint32_t>(queue, 100, 1);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // testing sycl::vec<double, 3> and std::uint32_t, sycl::vec<std::uint32_t, 2>
    std::cout << "-------------------------------------------------------------------" << std::endl;
    std::cout << "sycl::vec<double,3>, std::uint32_t, sycl::vec<std::uint32_t, 2> type" << std::endl;
    std::cout << "-------------------------------------------------------------------" << std::endl;
    err = tests_set<sycl::vec<double, 3>, std::uint32_t>(queue, 99);
#if TEST_LONG_RUN
    err += tests_set<sycl::vec<double, 3>, sycl::vec<std::uint32_t, 2>>(queue, 100);
    err += tests_set_portion<sycl::vec<double, 3>, std::uint32_t>(queue, 99, 1);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // testing sycl::vec<double, 4> and std::uint32_t, sycl::vec<std::uint32_t, 3>
    std::cout << "-------------------------------------------------------------------" << std::endl;
    std::cout << "sycl::vec<double,4>, std::uint32_t, sycl::vec<std::uint32_t, 3> type" << std::endl;
    std::cout << "-------------------------------------------------------------------" << std::endl;
    err = tests_set<sycl::vec<double, 4>, std::uint32_t>(queue, 100);
#if TEST_LONG_RUN
    err += tests_set<sycl::vec<double, 4>, sycl::vec<std::uint32_t, 3>>(queue, 100);
    err += tests_set_portion<sycl::vec<double, 4>, std::uint32_t>(queue, 100, 1);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // testing sycl::vec<double, 8> and std::uint32_t
    std::cout << "-------------------------------------" << std::endl;
    std::cout << "sycl::vec<double,8>, std::uint32_t type" << std::endl;
    std::cout << "-------------------------------------" << std::endl;
    err = tests_set<sycl::vec<double, 8>, std::uint32_t>(queue, 160);
#if TEST_LONG_RUN
    err += tests_set_portion<sycl::vec<double, 8>, std::uint32_t>(queue, 160, 5);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // testing sycl::vec<double, 16> and std::uint32_t
    std::cout << "--------------------------------------" << std::endl;
    std::cout << "sycl::vec<double,16>, std::uint32_t type" << std::endl;
    std::cout << "--------------------------------------" << std::endl;
    err = tests_set<sycl::vec<double, 16>, std::uint32_t>(queue, 160);
#if TEST_LONG_RUN
    err += tests_set_portion<sycl::vec<double, 16>, std::uint32_t>(queue, 140, 7);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");


#endif // TEST_DPCPP_BACKEND_PRESENT && TEST_UNNAMED_LAMBDAS

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT && TEST_UNNAMED_LAMBDAS);
}