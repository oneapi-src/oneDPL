// -*- C++ -*-
//===-- extreme_value_distribution_test.cpp ---------------------------------===//
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
// Test of extreme_value_distribution - check statistical properties of the distribution

#include "support/utils.h"
#include <iostream>

#if TEST_DPCPP_BACKEND_PRESENT && TEST_UNNAMED_LAMBDAS
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
    const double y = 0.5772156649015328606065120;
    const double pi = 3.1415926535897932384626433;
    double tM = _a + _b * y;
    double tD = pi * pi / 6.0 * _b * _b;
    double tQ = 27.0 / 5.0 * tD * tD;

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
                oneapi::dpl::extreme_value_distribution<RealType> distr(_a, _b);

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
                oneapi::dpl::extreme_value_distribution<RealType> distr(_a, _b);

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
    constexpr int nparams = 2;
    oneapi::dpl::internal::element_type_t<RealType> a_array [nparams] = {2.0, -10.0};
    oneapi::dpl::internal::element_type_t<RealType> b_array [nparams] = {1.0, 10.0};

    // Test for all non-zero parameters
    for(int i = 0; i < nparams; ++i) {
        std::cout << "extreme_value_distribution test<type>, a = " << a_array[i] << ", b = " << b_array[i] <<
        ", nsamples  = " << nsamples;
        if(test<RealType, UIntType>(queue, a_array[i], b_array[i], nsamples)) {
            return 1;
        }
    }
    return 0;
}

template <class RealType, class UIntType>
int
tests_set_portion(sycl::queue& queue, std::int32_t nsamples, unsigned int part)
{
    constexpr int nparams = 2;
    oneapi::dpl::internal::element_type_t<RealType> a_array [nparams] = {2.0, -10.0};
    oneapi::dpl::internal::element_type_t<RealType> b_array [nparams] = {1.0, 10.0};

    // Test for all non-zero parameters
    for(int i = 0; i < nparams; ++i) {
        std::cout << "extreme_value_distribution test<type>, a = " << a_array[i] << ", b = " << b_array[i] <<
        ", nsamples = " << nsamples << ", part = " << part;
        if(test_portion<RealType, UIntType>(queue, a_array[i], b_array[i], nsamples, part)) {
            return 1;
        }
    }
    return 0;
}

#endif // TEST_DPCPP_BACKEND_PRESENT && TEST_UNNAMED_LAMBDAS

int
main()
{

#if TEST_DPCPP_BACKEND_PRESENT && TEST_UNNAMED_LAMBDAS

    sycl::queue queue = TestUtils::get_test_queue();

    constexpr int nsamples = 100;
    int err = 0;

            // testing sycl::vec<float, 1> and std::uint32_t ... sycl::vec<std::uint32_t, 16>
    std::cout << "---------------------------------------------------------------------" << std::endl;
    std::cout << "sycl::vec<float,1>, std::uint32_t ... sycl::vec<std::uint32_t, 16> type" << std::endl;
    std::cout << "---------------------------------------------------------------------" << std::endl;
    err = tests_set<sycl::vec<float, 1>, std::uint32_t>(queue, nsamples);
#if TEST_LONG_RUN
    err += tests_set<sycl::vec<float, 1>, sycl::vec<std::uint32_t, 16>>(queue, nsamples);
    err += tests_set<sycl::vec<float, 1>, sycl::vec<std::uint32_t, 8>>(queue, nsamples);
    err += tests_set<sycl::vec<float, 1>, sycl::vec<std::uint32_t, 4>>(queue, nsamples);
    err += tests_set<sycl::vec<float, 1>, sycl::vec<std::uint32_t, 3>>(queue, nsamples);
    err += tests_set<sycl::vec<float, 1>, sycl::vec<std::uint32_t, 2>>(queue, nsamples);
    err += tests_set<sycl::vec<float, 1>, sycl::vec<std::uint32_t, 1>>(queue, nsamples);
    err += tests_set_portion<sycl::vec<float, 1>, std::uint32_t>(queue, 100, 1);
    err += tests_set_portion<sycl::vec<float, 1>, std::uint32_t>(queue, 100, 2);
    err += tests_set_portion<sycl::vec<float, 1>, sycl::vec<std::uint32_t, 3>>(queue, 100, 1);
    err += tests_set_portion<sycl::vec<float, 1>, sycl::vec<std::uint32_t, 2>>(queue, 100, 1);
    err += tests_set_portion<sycl::vec<float, 1>, sycl::vec<std::uint32_t, 1>>(queue, 100, 1);
    err += tests_set_portion<sycl::vec<float, 1>, sycl::vec<std::uint32_t, 16>>(queue, 100, 2);
    err += tests_set_portion<sycl::vec<float, 1>, sycl::vec<std::uint32_t, 8>>(queue, 100, 2);
    err += tests_set_portion<sycl::vec<float, 1>, sycl::vec<std::uint32_t, 4>>(queue, 100, 2);
    err += tests_set_portion<sycl::vec<float, 1>, sycl::vec<std::uint32_t, 3>>(queue, 100, 2);
    err += tests_set_portion<sycl::vec<float, 1>, sycl::vec<std::uint32_t, 2>>(queue, 100, 2);
    err += tests_set_portion<sycl::vec<float, 1>, sycl::vec<std::uint32_t, 1>>(queue, 100, 2);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // testing sycl::vec<float, 2> and std::uint32_t ... sycl::vec<std::uint32_t, 16>
    std::cout << "---------------------------------------------------------------------" << std::endl;
    std::cout << "sycl::vec<float,2>, std::uint32_t ... sycl::vec<std::uint32_t, 16> type" << std::endl;
    std::cout << "---------------------------------------------------------------------" << std::endl;
    err = tests_set<sycl::vec<float, 2>, std::uint32_t>(queue, nsamples);
#if TEST_LONG_RUN
    err += tests_set<sycl::vec<float, 2>, sycl::vec<std::uint32_t, 16>>(queue, nsamples);
    err += tests_set<sycl::vec<float, 2>, sycl::vec<std::uint32_t, 8>>(queue, nsamples);
    err += tests_set<sycl::vec<float, 2>, sycl::vec<std::uint32_t, 4>>(queue, nsamples);
    err += tests_set<sycl::vec<float, 2>, sycl::vec<std::uint32_t, 3>>(queue, nsamples);
    err += tests_set<sycl::vec<float, 2>, sycl::vec<std::uint32_t, 2>>(queue, nsamples);
    err += tests_set<sycl::vec<float, 2>, sycl::vec<std::uint32_t, 1>>(queue, nsamples);
    err += tests_set_portion<sycl::vec<float, 2>, std::uint32_t>(queue, 100, 1);
    err += tests_set_portion<sycl::vec<float, 2>, std::uint32_t>(queue, 100, 3);
    err += tests_set_portion<sycl::vec<float, 2>, sycl::vec<std::uint32_t, 16>>(queue, 100, 1);
    err += tests_set_portion<sycl::vec<float, 2>, sycl::vec<std::uint32_t, 8>>(queue, 100, 1);
    err += tests_set_portion<sycl::vec<float, 2>, sycl::vec<std::uint32_t, 4>>(queue, 100, 1);
    err += tests_set_portion<sycl::vec<float, 2>, sycl::vec<std::uint32_t, 3>>(queue, 100, 1);
    err += tests_set_portion<sycl::vec<float, 2>, sycl::vec<std::uint32_t, 2>>(queue, 100, 1);
    err += tests_set_portion<sycl::vec<float, 2>, sycl::vec<std::uint32_t, 1>>(queue, 100, 1);
    err += tests_set_portion<sycl::vec<float, 2>, sycl::vec<std::uint32_t, 16>>(queue, 100, 3);
    err += tests_set_portion<sycl::vec<float, 2>, sycl::vec<std::uint32_t, 8>>(queue, 100, 3);
    err += tests_set_portion<sycl::vec<float, 2>, sycl::vec<std::uint32_t, 4>>(queue, 100, 3);
    err += tests_set_portion<sycl::vec<float, 2>, sycl::vec<std::uint32_t, 3>>(queue, 100, 3);
    err += tests_set_portion<sycl::vec<float, 2>, sycl::vec<std::uint32_t, 2>>(queue, 100, 3);
    err += tests_set_portion<sycl::vec<float, 2>, sycl::vec<std::uint32_t, 1>>(queue, 100, 3);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // testing sycl::vec<float, 3> and std::uint32_t ... sycl::vec<std::uint32_t, 16>
    std::cout << "---------------------------------------------------------------------" << std::endl;
    std::cout << "sycl::vec<float,3>, std::uint32_t ... sycl::vec<std::uint32_t, 16> type" << std::endl;
    std::cout << "---------------------------------------------------------------------" << std::endl;
    err = tests_set<sycl::vec<float, 3>, std::uint32_t>(queue, 99);
#if TEST_LONG_RUN
    err += tests_set<sycl::vec<float, 3>, sycl::vec<std::uint32_t, 16>>(queue, 99);
    err += tests_set<sycl::vec<float, 3>, sycl::vec<std::uint32_t, 8>>(queue, 99);
    err += tests_set<sycl::vec<float, 3>, sycl::vec<std::uint32_t, 4>>(queue, 99);
    err += tests_set<sycl::vec<float, 3>, sycl::vec<std::uint32_t, 3>>(queue, 99);
    err += tests_set<sycl::vec<float, 3>, sycl::vec<std::uint32_t, 2>>(queue, 99);
    err += tests_set<sycl::vec<float, 3>, sycl::vec<std::uint32_t, 1>>(queue, 99);
    err += tests_set_portion<sycl::vec<float, 3>, std::uint32_t>(queue, 99, 1);
    err += tests_set_portion<sycl::vec<float, 3>, std::uint32_t>(queue, 99, 4);
    err += tests_set_portion<sycl::vec<float, 3>, sycl::vec<std::uint32_t, 16>>(queue, 99, 1);
    err += tests_set_portion<sycl::vec<float, 3>, sycl::vec<std::uint32_t, 8>>(queue, 99, 1);
    err += tests_set_portion<sycl::vec<float, 3>, sycl::vec<std::uint32_t, 4>>(queue, 99, 1);
    err += tests_set_portion<sycl::vec<float, 3>, sycl::vec<std::uint32_t, 3>>(queue, 99, 1);
    err += tests_set_portion<sycl::vec<float, 3>, sycl::vec<std::uint32_t, 2>>(queue, 99, 1);
    err += tests_set_portion<sycl::vec<float, 3>, sycl::vec<std::uint32_t, 1>>(queue, 99, 1);
    err += tests_set_portion<sycl::vec<float, 3>, sycl::vec<std::uint32_t, 16>>(queue, 99, 4);
    err += tests_set_portion<sycl::vec<float, 3>, sycl::vec<std::uint32_t, 8>>(queue, 99, 4);
    err += tests_set_portion<sycl::vec<float, 3>, sycl::vec<std::uint32_t, 4>>(queue, 99, 4);
    err += tests_set_portion<sycl::vec<float, 3>, sycl::vec<std::uint32_t, 3>>(queue, 99, 4);
    err += tests_set_portion<sycl::vec<float, 3>, sycl::vec<std::uint32_t, 2>>(queue, 99, 4);
    err += tests_set_portion<sycl::vec<float, 3>, sycl::vec<std::uint32_t, 1>>(queue, 99, 4);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // testing sycl::vec<float, 4> and std::uint32_t ... sycl::vec<std::uint32_t, 16>
    std::cout << "---------------------------------------------------------------------" << std::endl;
    std::cout << "sycl::vec<float,4>, std::uint32_t ... sycl::vec<std::uint32_t, 16> type" << std::endl;
    std::cout << "---------------------------------------------------------------------" << std::endl;
    err = tests_set<sycl::vec<float, 4>, std::uint32_t>(queue, 100);
#if TEST_LONG_RUN
    err += tests_set<sycl::vec<float, 4>, sycl::vec<std::uint32_t, 16>>(queue, 100);
    err += tests_set<sycl::vec<float, 4>, sycl::vec<std::uint32_t, 8>>(queue, 100);
    err += tests_set<sycl::vec<float, 4>, sycl::vec<std::uint32_t, 4>>(queue, 100);
    err += tests_set<sycl::vec<float, 4>, sycl::vec<std::uint32_t, 3>>(queue, 100);
    err += tests_set<sycl::vec<float, 4>, sycl::vec<std::uint32_t, 2>>(queue, 100);
    err += tests_set<sycl::vec<float, 4>, sycl::vec<std::uint32_t, 1>>(queue, 100);
    err += tests_set_portion<sycl::vec<float, 4>, std::uint32_t>(queue, 100, 1);
    err += tests_set_portion<sycl::vec<float, 4>, std::uint32_t>(queue, 100, 5);
    err += tests_set_portion<sycl::vec<float, 4>, sycl::vec<std::uint32_t, 16>>(queue, 100, 1);
    err += tests_set_portion<sycl::vec<float, 4>, sycl::vec<std::uint32_t, 8>>(queue, 100, 1);
    err += tests_set_portion<sycl::vec<float, 4>, sycl::vec<std::uint32_t, 4>>(queue, 100, 1);
    err += tests_set_portion<sycl::vec<float, 4>, sycl::vec<std::uint32_t, 3>>(queue, 100, 1);
    err += tests_set_portion<sycl::vec<float, 4>, sycl::vec<std::uint32_t, 2>>(queue, 100, 1);
    err += tests_set_portion<sycl::vec<float, 4>, sycl::vec<std::uint32_t, 1>>(queue, 100, 1);
    err += tests_set_portion<sycl::vec<float, 4>, sycl::vec<std::uint32_t, 16>>(queue, 100, 5);
    err += tests_set_portion<sycl::vec<float, 4>, sycl::vec<std::uint32_t, 8>>(queue, 100, 5);
    err += tests_set_portion<sycl::vec<float, 4>, sycl::vec<std::uint32_t, 4>>(queue, 100, 5);
    err += tests_set_portion<sycl::vec<float, 4>, sycl::vec<std::uint32_t, 3>>(queue, 100, 5);
    err += tests_set_portion<sycl::vec<float, 4>, sycl::vec<std::uint32_t, 2>>(queue, 100, 5);
    err += tests_set_portion<sycl::vec<float, 4>, sycl::vec<std::uint32_t, 1>>(queue, 100, 5);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // testing sycl::vec<float, 8> and std::uint32_t ... sycl::vec<std::uint32_t, 16>
    std::cout << "---------------------------------------------------------------------" << std::endl;
    std::cout << "sycl::vec<float,8>, std::uint32_t ... sycl::vec<std::uint32_t, 16> type" << std::endl;
    std::cout << "---------------------------------------------------------------------" << std::endl;
    err = tests_set<sycl::vec<float, 8>, std::uint32_t>(queue, 160);
#if TEST_LONG_RUN
    err += tests_set<sycl::vec<float, 8>, sycl::vec<std::uint32_t, 16>>(queue, 160);
    err += tests_set<sycl::vec<float, 8>, sycl::vec<std::uint32_t, 8>>(queue, 160);
    err += tests_set<sycl::vec<float, 8>, sycl::vec<std::uint32_t, 4>>(queue, 160);
    err += tests_set<sycl::vec<float, 8>, sycl::vec<std::uint32_t, 3>>(queue, 160);
    err += tests_set<sycl::vec<float, 8>, sycl::vec<std::uint32_t, 2>>(queue, 160);
    err += tests_set<sycl::vec<float, 8>, sycl::vec<std::uint32_t, 1>>(queue, 160);
    err += tests_set_portion<sycl::vec<float, 8>, std::uint32_t>(queue, 160, 1);
    err += tests_set_portion<sycl::vec<float, 8>, std::uint32_t>(queue, 160, 5);
    err += tests_set_portion<sycl::vec<float, 8>, std::uint32_t>(queue, 160, 9);
    err += tests_set_portion<sycl::vec<float, 8>, sycl::vec<std::uint32_t, 16>>(queue, 160, 1);
    err += tests_set_portion<sycl::vec<float, 8>, sycl::vec<std::uint32_t, 8>>(queue, 160, 1);
    err += tests_set_portion<sycl::vec<float, 8>, sycl::vec<std::uint32_t, 4>>(queue, 160, 1);
    err += tests_set_portion<sycl::vec<float, 8>, sycl::vec<std::uint32_t, 3>>(queue, 160, 1);
    err += tests_set_portion<sycl::vec<float, 8>, sycl::vec<std::uint32_t, 2>>(queue, 160, 1);
    err += tests_set_portion<sycl::vec<float, 8>, sycl::vec<std::uint32_t, 1>>(queue, 160, 1);
    err += tests_set_portion<sycl::vec<float, 8>, sycl::vec<std::uint32_t, 16>>(queue, 160, 5);
    err += tests_set_portion<sycl::vec<float, 8>, sycl::vec<std::uint32_t, 8>>(queue, 160, 5);
    err += tests_set_portion<sycl::vec<float, 8>, sycl::vec<std::uint32_t, 4>>(queue, 160, 5);
    err += tests_set_portion<sycl::vec<float, 8>, sycl::vec<std::uint32_t, 3>>(queue, 160, 5);
    err += tests_set_portion<sycl::vec<float, 8>, sycl::vec<std::uint32_t, 2>>(queue, 160, 5);
    err += tests_set_portion<sycl::vec<float, 8>, sycl::vec<std::uint32_t, 1>>(queue, 160, 5);
    err += tests_set_portion<sycl::vec<float, 8>, sycl::vec<std::uint32_t, 16>>(queue, 160, 9);
    err += tests_set_portion<sycl::vec<float, 8>, sycl::vec<std::uint32_t, 8>>(queue, 160, 9);
    err += tests_set_portion<sycl::vec<float, 8>, sycl::vec<std::uint32_t, 4>>(queue, 160, 9);
    err += tests_set_portion<sycl::vec<float, 8>, sycl::vec<std::uint32_t, 3>>(queue, 160, 9);
    err += tests_set_portion<sycl::vec<float, 8>, sycl::vec<std::uint32_t, 2>>(queue, 160, 9);
    err += tests_set_portion<sycl::vec<float, 8>, sycl::vec<std::uint32_t, 1>>(queue, 160, 9);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // testing sycl::vec<float, 16> and std::uint32_t ... sycl::vec<std::uint32_t, 16>
    std::cout << "---------------------------------------------------------------------" << std::endl;
    std::cout << "sycl::vec<float,16>, std::uint32_t ... sycl::vec<std::uint32_t, 16> type" << std::endl;
    std::cout << "---------------------------------------------------------------------" << std::endl;
    err = tests_set<sycl::vec<float, 16>, std::uint32_t>(queue, 160);
#if TEST_LONG_RUN
    err += tests_set<sycl::vec<float, 16>, sycl::vec<std::uint32_t, 16>>(queue, 160);
    err += tests_set<sycl::vec<float, 16>, sycl::vec<std::uint32_t, 8>>(queue, 160);
    err += tests_set<sycl::vec<float, 16>, sycl::vec<std::uint32_t, 4>>(queue, 160);
    err += tests_set<sycl::vec<float, 16>, sycl::vec<std::uint32_t, 3>>(queue, 160);
    err += tests_set<sycl::vec<float, 16>, sycl::vec<std::uint32_t, 2>>(queue, 160);
    err += tests_set<sycl::vec<float, 16>, sycl::vec<std::uint32_t, 1>>(queue, 160);
    err += tests_set_portion<sycl::vec<float, 16>, std::uint32_t>(queue, 160, 1);
    err += tests_set_portion<sycl::vec<float, 16>, std::uint32_t>(queue, 140, 7);
    err += tests_set_portion<sycl::vec<float, 16>, std::uint32_t>(queue, 160, 17);
    err += tests_set_portion<sycl::vec<float, 16>, sycl::vec<std::uint32_t, 16>>(queue, 160, 1);
    err += tests_set_portion<sycl::vec<float, 16>, sycl::vec<std::uint32_t, 8>>(queue, 160, 1);
    err += tests_set_portion<sycl::vec<float, 16>, sycl::vec<std::uint32_t, 4>>(queue, 160, 1);
    err += tests_set_portion<sycl::vec<float, 16>, sycl::vec<std::uint32_t, 3>>(queue, 160, 1);
    err += tests_set_portion<sycl::vec<float, 16>, sycl::vec<std::uint32_t, 2>>(queue, 160, 1);
    err += tests_set_portion<sycl::vec<float, 16>, sycl::vec<std::uint32_t, 1>>(queue, 160, 1);
    err += tests_set_portion<sycl::vec<float, 16>, sycl::vec<std::uint32_t, 16>>(queue, 140, 7);
    err += tests_set_portion<sycl::vec<float, 16>, sycl::vec<std::uint32_t, 8>>(queue, 140, 7);
    err += tests_set_portion<sycl::vec<float, 16>, sycl::vec<std::uint32_t, 4>>(queue, 140, 7);
    err += tests_set_portion<sycl::vec<float, 16>, sycl::vec<std::uint32_t, 3>>(queue, 140, 7);
    err += tests_set_portion<sycl::vec<float, 16>, sycl::vec<std::uint32_t, 2>>(queue, 140, 7);
    err += tests_set_portion<sycl::vec<float, 16>, sycl::vec<std::uint32_t, 1>>(queue, 140, 7);
    err += tests_set_portion<sycl::vec<float, 16>, sycl::vec<std::uint32_t, 16>>(queue, 160, 17);
    err += tests_set_portion<sycl::vec<float, 16>, sycl::vec<std::uint32_t, 8>>(queue, 160, 17);
    err += tests_set_portion<sycl::vec<float, 16>, sycl::vec<std::uint32_t, 4>>(queue, 160, 17);
    err += tests_set_portion<sycl::vec<float, 16>, sycl::vec<std::uint32_t, 3>>(queue, 160, 17);
    err += tests_set_portion<sycl::vec<float, 16>, sycl::vec<std::uint32_t, 2>>(queue, 160, 17);
    err += tests_set_portion<sycl::vec<float, 16>, sycl::vec<std::uint32_t, 1>>(queue, 160, 17);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // Skip tests if DP is not supported
    if (TestUtils::has_type_support<double>(queue.get_device())) {
        // testing sycl::vec<double, 1> and std::uint32_t ... sycl::vec<std::uint32_t, 16>
        std::cout << "---------------------------------------------------------------------" << std::endl;
        std::cout << "sycl::vec<double,1>, std::uint32_t ... sycl::vec<std::uint32_t, 16> type" << std::endl;
        std::cout << "---------------------------------------------------------------------" << std::endl;
        err = tests_set<sycl::vec<double, 1>, std::uint32_t>(queue, nsamples);
#if TEST_LONG_RUN
        err += tests_set<sycl::vec<double, 1>, sycl::vec<std::uint32_t, 16>>(queue, nsamples);
        err += tests_set<sycl::vec<double, 1>, sycl::vec<std::uint32_t, 8>>(queue, nsamples);
        err += tests_set<sycl::vec<double, 1>, sycl::vec<std::uint32_t, 4>>(queue, nsamples);
        err += tests_set<sycl::vec<double, 1>, sycl::vec<std::uint32_t, 3>>(queue, nsamples);
        err += tests_set<sycl::vec<double, 1>, sycl::vec<std::uint32_t, 2>>(queue, nsamples);
        err += tests_set<sycl::vec<double, 1>, sycl::vec<std::uint32_t, 1>>(queue, nsamples);
        err += tests_set_portion<sycl::vec<double, 1>, std::uint32_t>(queue, 100, 1);
        err += tests_set_portion<sycl::vec<double, 1>, std::uint32_t>(queue, 100, 2);
        err += tests_set_portion<sycl::vec<double, 1>, sycl::vec<std::uint32_t, 3>>(queue, 100, 1);
        err += tests_set_portion<sycl::vec<double, 1>, sycl::vec<std::uint32_t, 2>>(queue, 100, 1);
        err += tests_set_portion<sycl::vec<double, 1>, sycl::vec<std::uint32_t, 1>>(queue, 100, 1);
        err += tests_set_portion<sycl::vec<double, 1>, sycl::vec<std::uint32_t, 16>>(queue, 100, 2);
        err += tests_set_portion<sycl::vec<double, 1>, sycl::vec<std::uint32_t, 8>>(queue, 100, 2);
        err += tests_set_portion<sycl::vec<double, 1>, sycl::vec<std::uint32_t, 4>>(queue, 100, 2);
        err += tests_set_portion<sycl::vec<double, 1>, sycl::vec<std::uint32_t, 3>>(queue, 100, 2);
        err += tests_set_portion<sycl::vec<double, 1>, sycl::vec<std::uint32_t, 2>>(queue, 100, 2);
        err += tests_set_portion<sycl::vec<double, 1>, sycl::vec<std::uint32_t, 1>>(queue, 100, 2);
#endif // TEST_LONG_RUN
        EXPECT_TRUE(!err, "Test FAILED");

        // testing sycl::vec<double, 2> and std::uint32_t ... sycl::vec<std::uint32_t, 16>
        std::cout << "---------------------------------------------------------------------" << std::endl;
        std::cout << "sycl::vec<double,2>, std::uint32_t ... sycl::vec<std::uint32_t, 16> type" << std::endl;
        std::cout << "---------------------------------------------------------------------" << std::endl;
        err = tests_set<sycl::vec<double, 2>, std::uint32_t>(queue, nsamples);
#if TEST_LONG_RUN
        err += tests_set<sycl::vec<double, 2>, sycl::vec<std::uint32_t, 16>>(queue, nsamples);
        err += tests_set<sycl::vec<double, 2>, sycl::vec<std::uint32_t, 8>>(queue, nsamples);
        err += tests_set<sycl::vec<double, 2>, sycl::vec<std::uint32_t, 4>>(queue, nsamples);
        err += tests_set<sycl::vec<double, 2>, sycl::vec<std::uint32_t, 3>>(queue, nsamples);
        err += tests_set<sycl::vec<double, 2>, sycl::vec<std::uint32_t, 2>>(queue, nsamples);
        err += tests_set<sycl::vec<double, 2>, sycl::vec<std::uint32_t, 1>>(queue, nsamples);
        err += tests_set_portion<sycl::vec<double, 2>, std::uint32_t>(queue, 100, 1);
        err += tests_set_portion<sycl::vec<double, 2>, std::uint32_t>(queue, 100, 3);
        err += tests_set_portion<sycl::vec<double, 2>, sycl::vec<std::uint32_t, 16>>(queue, 100, 1);
        err += tests_set_portion<sycl::vec<double, 2>, sycl::vec<std::uint32_t, 8>>(queue, 100, 1);
        err += tests_set_portion<sycl::vec<double, 2>, sycl::vec<std::uint32_t, 4>>(queue, 100, 1);
        err += tests_set_portion<sycl::vec<double, 2>, sycl::vec<std::uint32_t, 3>>(queue, 100, 1);
        err += tests_set_portion<sycl::vec<double, 2>, sycl::vec<std::uint32_t, 2>>(queue, 100, 1);
        err += tests_set_portion<sycl::vec<double, 2>, sycl::vec<std::uint32_t, 1>>(queue, 100, 1);
        err += tests_set_portion<sycl::vec<double, 2>, sycl::vec<std::uint32_t, 16>>(queue, 100, 3);
        err += tests_set_portion<sycl::vec<double, 2>, sycl::vec<std::uint32_t, 8>>(queue, 100, 3);
        err += tests_set_portion<sycl::vec<double, 2>, sycl::vec<std::uint32_t, 4>>(queue, 100, 3);
        err += tests_set_portion<sycl::vec<double, 2>, sycl::vec<std::uint32_t, 3>>(queue, 100, 3);
        err += tests_set_portion<sycl::vec<double, 2>, sycl::vec<std::uint32_t, 2>>(queue, 100, 3);
        err += tests_set_portion<sycl::vec<double, 2>, sycl::vec<std::uint32_t, 1>>(queue, 100, 3);
#endif // TEST_LONG_RUN
        EXPECT_TRUE(!err, "Test FAILED");

        // testing sycl::vec<double, 3> and std::uint32_t ... sycl::vec<std::uint32_t, 16>
        std::cout << "---------------------------------------------------------------------" << std::endl;
        std::cout << "sycl::vec<double,3>, std::uint32_t ... sycl::vec<std::uint32_t, 16> type" << std::endl;
        std::cout << "---------------------------------------------------------------------" << std::endl;
        err = tests_set<sycl::vec<double, 3>, std::uint32_t>(queue, 99);
#if TEST_LONG_RUN
        err += tests_set<sycl::vec<double, 3>, sycl::vec<std::uint32_t, 16>>(queue, 99);
        err += tests_set<sycl::vec<double, 3>, sycl::vec<std::uint32_t, 8>>(queue, 99);
        err += tests_set<sycl::vec<double, 3>, sycl::vec<std::uint32_t, 4>>(queue, 99);
        err += tests_set<sycl::vec<double, 3>, sycl::vec<std::uint32_t, 3>>(queue, 99);
        err += tests_set<sycl::vec<double, 3>, sycl::vec<std::uint32_t, 2>>(queue, 99);
        err += tests_set<sycl::vec<double, 3>, sycl::vec<std::uint32_t, 1>>(queue, 99);
        err += tests_set_portion<sycl::vec<double, 3>, std::uint32_t>(queue, 99, 1);
        err += tests_set_portion<sycl::vec<double, 3>, std::uint32_t>(queue, 99, 4);
        err += tests_set_portion<sycl::vec<double, 3>, sycl::vec<std::uint32_t, 16>>(queue, 99, 1);
        err += tests_set_portion<sycl::vec<double, 3>, sycl::vec<std::uint32_t, 8>>(queue, 99, 1);
        err += tests_set_portion<sycl::vec<double, 3>, sycl::vec<std::uint32_t, 4>>(queue, 99, 1);
        err += tests_set_portion<sycl::vec<double, 3>, sycl::vec<std::uint32_t, 3>>(queue, 99, 1);
        err += tests_set_portion<sycl::vec<double, 3>, sycl::vec<std::uint32_t, 2>>(queue, 99, 1);
        err += tests_set_portion<sycl::vec<double, 3>, sycl::vec<std::uint32_t, 1>>(queue, 99, 1);
        err += tests_set_portion<sycl::vec<double, 3>, sycl::vec<std::uint32_t, 16>>(queue, 99, 4);
        err += tests_set_portion<sycl::vec<double, 3>, sycl::vec<std::uint32_t, 8>>(queue, 99, 4);
        err += tests_set_portion<sycl::vec<double, 3>, sycl::vec<std::uint32_t, 4>>(queue, 99, 4);
        err += tests_set_portion<sycl::vec<double, 3>, sycl::vec<std::uint32_t, 3>>(queue, 99, 4);
        err += tests_set_portion<sycl::vec<double, 3>, sycl::vec<std::uint32_t, 2>>(queue, 99, 4);
        err += tests_set_portion<sycl::vec<double, 3>, sycl::vec<std::uint32_t, 1>>(queue, 99, 4);
#endif // TEST_LONG_RUN
        EXPECT_TRUE(!err, "Test FAILED");

        // testing sycl::vec<double, 4> and std::uint32_t ... sycl::vec<std::uint32_t, 16>
        std::cout << "---------------------------------------------------------------------" << std::endl;
        std::cout << "sycl::vec<double,4>, std::uint32_t ... sycl::vec<std::uint32_t, 16> type" << std::endl;
        std::cout << "---------------------------------------------------------------------" << std::endl;
        err = tests_set<sycl::vec<double, 4>, std::uint32_t>(queue, 100);
#if TEST_LONG_RUN
        err += tests_set<sycl::vec<double, 4>, sycl::vec<std::uint32_t, 16>>(queue, 100);
        err += tests_set<sycl::vec<double, 4>, sycl::vec<std::uint32_t, 8>>(queue, 100);
        err += tests_set<sycl::vec<double, 4>, sycl::vec<std::uint32_t, 4>>(queue, 100);
        err += tests_set<sycl::vec<double, 4>, sycl::vec<std::uint32_t, 3>>(queue, 100);
        err += tests_set<sycl::vec<double, 4>, sycl::vec<std::uint32_t, 2>>(queue, 100);
        err += tests_set<sycl::vec<double, 4>, sycl::vec<std::uint32_t, 1>>(queue, 100);
        err += tests_set_portion<sycl::vec<double, 4>, std::uint32_t>(queue, 100, 1);
        err += tests_set_portion<sycl::vec<double, 4>, std::uint32_t>(queue, 100, 5);
        err += tests_set_portion<sycl::vec<double, 4>, sycl::vec<std::uint32_t, 16>>(queue, 100, 1);
        err += tests_set_portion<sycl::vec<double, 4>, sycl::vec<std::uint32_t, 8>>(queue, 100, 1);
        err += tests_set_portion<sycl::vec<double, 4>, sycl::vec<std::uint32_t, 4>>(queue, 100, 1);
        err += tests_set_portion<sycl::vec<double, 4>, sycl::vec<std::uint32_t, 3>>(queue, 100, 1);
        err += tests_set_portion<sycl::vec<double, 4>, sycl::vec<std::uint32_t, 2>>(queue, 100, 1);
        err += tests_set_portion<sycl::vec<double, 4>, sycl::vec<std::uint32_t, 1>>(queue, 100, 1);
        err += tests_set_portion<sycl::vec<double, 4>, sycl::vec<std::uint32_t, 16>>(queue, 100, 5);
        err += tests_set_portion<sycl::vec<double, 4>, sycl::vec<std::uint32_t, 8>>(queue, 100, 5);
        err += tests_set_portion<sycl::vec<double, 4>, sycl::vec<std::uint32_t, 4>>(queue, 100, 5);
        err += tests_set_portion<sycl::vec<double, 4>, sycl::vec<std::uint32_t, 3>>(queue, 100, 5);
        err += tests_set_portion<sycl::vec<double, 4>, sycl::vec<std::uint32_t, 2>>(queue, 100, 5);
        err += tests_set_portion<sycl::vec<double, 4>, sycl::vec<std::uint32_t, 1>>(queue, 100, 5);
#endif // TEST_LONG_RUN
        EXPECT_TRUE(!err, "Test FAILED");

        // testing sycl::vec<double, 8> and std::uint32_t ... sycl::vec<std::uint32_t, 16>
        std::cout << "---------------------------------------------------------------------" << std::endl;
        std::cout << "sycl::vec<double,8>, std::uint32_t ... sycl::vec<std::uint32_t, 16> type" << std::endl;
        std::cout << "---------------------------------------------------------------------" << std::endl;
        err = tests_set<sycl::vec<double, 8>, std::uint32_t>(queue, 160);
#if TEST_LONG_RUN
        err += tests_set<sycl::vec<double, 8>, sycl::vec<std::uint32_t, 16>>(queue, 160);
        err += tests_set<sycl::vec<double, 8>, sycl::vec<std::uint32_t, 8>>(queue, 160);
        err += tests_set<sycl::vec<double, 8>, sycl::vec<std::uint32_t, 4>>(queue, 160);
        err += tests_set<sycl::vec<double, 8>, sycl::vec<std::uint32_t, 3>>(queue, 160);
        err += tests_set<sycl::vec<double, 8>, sycl::vec<std::uint32_t, 2>>(queue, 160);
        err += tests_set<sycl::vec<double, 8>, sycl::vec<std::uint32_t, 1>>(queue, 160);
        err += tests_set_portion<sycl::vec<double, 8>, std::uint32_t>(queue, 160, 1);
        err += tests_set_portion<sycl::vec<double, 8>, std::uint32_t>(queue, 160, 5);
        err += tests_set_portion<sycl::vec<double, 8>, std::uint32_t>(queue, 160, 9);
        err += tests_set_portion<sycl::vec<double, 8>, sycl::vec<std::uint32_t, 16>>(queue, 160, 1);
        err += tests_set_portion<sycl::vec<double, 8>, sycl::vec<std::uint32_t, 8>>(queue, 160, 1);
        err += tests_set_portion<sycl::vec<double, 8>, sycl::vec<std::uint32_t, 4>>(queue, 160, 1);
        err += tests_set_portion<sycl::vec<double, 8>, sycl::vec<std::uint32_t, 3>>(queue, 160, 1);
        err += tests_set_portion<sycl::vec<double, 8>, sycl::vec<std::uint32_t, 2>>(queue, 160, 1);
        err += tests_set_portion<sycl::vec<double, 8>, sycl::vec<std::uint32_t, 1>>(queue, 160, 1);
        err += tests_set_portion<sycl::vec<double, 8>, sycl::vec<std::uint32_t, 16>>(queue, 160, 5);
        err += tests_set_portion<sycl::vec<double, 8>, sycl::vec<std::uint32_t, 8>>(queue, 160, 5);
        err += tests_set_portion<sycl::vec<double, 8>, sycl::vec<std::uint32_t, 4>>(queue, 160, 5);
        err += tests_set_portion<sycl::vec<double, 8>, sycl::vec<std::uint32_t, 3>>(queue, 160, 5);
        err += tests_set_portion<sycl::vec<double, 8>, sycl::vec<std::uint32_t, 2>>(queue, 160, 5);
        err += tests_set_portion<sycl::vec<double, 8>, sycl::vec<std::uint32_t, 1>>(queue, 160, 5);
        err += tests_set_portion<sycl::vec<double, 8>, sycl::vec<std::uint32_t, 16>>(queue, 160, 9);
        err += tests_set_portion<sycl::vec<double, 8>, sycl::vec<std::uint32_t, 8>>(queue, 160, 9);
        err += tests_set_portion<sycl::vec<double, 8>, sycl::vec<std::uint32_t, 4>>(queue, 160, 9);
        err += tests_set_portion<sycl::vec<double, 8>, sycl::vec<std::uint32_t, 3>>(queue, 160, 9);
        err += tests_set_portion<sycl::vec<double, 8>, sycl::vec<std::uint32_t, 2>>(queue, 160, 9);
        err += tests_set_portion<sycl::vec<double, 8>, sycl::vec<std::uint32_t, 1>>(queue, 160, 9);
#endif // TEST_LONG_RUN
        EXPECT_TRUE(!err, "Test FAILED");

        // testing sycl::vec<double, 16> and std::uint32_t ... sycl::vec<std::uint32_t, 16>
        std::cout << "---------------------------------------------------------------------" << std::endl;
        std::cout << "sycl::vec<double,16>, std::uint32_t ... sycl::vec<std::uint32_t, 16> type" << std::endl;
        std::cout << "---------------------------------------------------------------------" << std::endl;
        err = tests_set<sycl::vec<double, 16>, std::uint32_t>(queue, 160);
#if TEST_LONG_RUN
        err += tests_set<sycl::vec<double, 16>, sycl::vec<std::uint32_t, 16>>(queue, 160);
        err += tests_set<sycl::vec<double, 16>, sycl::vec<std::uint32_t, 8>>(queue, 160);
        err += tests_set<sycl::vec<double, 16>, sycl::vec<std::uint32_t, 4>>(queue, 160);
        err += tests_set<sycl::vec<double, 16>, sycl::vec<std::uint32_t, 3>>(queue, 160);
        err += tests_set<sycl::vec<double, 16>, sycl::vec<std::uint32_t, 2>>(queue, 160);
        err += tests_set<sycl::vec<double, 16>, sycl::vec<std::uint32_t, 1>>(queue, 160);
        err += tests_set_portion<sycl::vec<double, 16>, std::uint32_t>(queue, 160, 1);
        err += tests_set_portion<sycl::vec<double, 16>, std::uint32_t>(queue, 140, 7);
        err += tests_set_portion<sycl::vec<double, 16>, std::uint32_t>(queue, 160, 17);
        err += tests_set_portion<sycl::vec<double, 16>, sycl::vec<std::uint32_t, 16>>(queue, 160, 1);
        err += tests_set_portion<sycl::vec<double, 16>, sycl::vec<std::uint32_t, 8>>(queue, 160, 1);
        err += tests_set_portion<sycl::vec<double, 16>, sycl::vec<std::uint32_t, 4>>(queue, 160, 1);
        err += tests_set_portion<sycl::vec<double, 16>, sycl::vec<std::uint32_t, 3>>(queue, 160, 1);
        err += tests_set_portion<sycl::vec<double, 16>, sycl::vec<std::uint32_t, 2>>(queue, 160, 1);
        err += tests_set_portion<sycl::vec<double, 16>, sycl::vec<std::uint32_t, 1>>(queue, 160, 1);
        err += tests_set_portion<sycl::vec<double, 16>, sycl::vec<std::uint32_t, 16>>(queue, 140, 7);
        err += tests_set_portion<sycl::vec<double, 16>, sycl::vec<std::uint32_t, 8>>(queue, 140, 7);
        err += tests_set_portion<sycl::vec<double, 16>, sycl::vec<std::uint32_t, 4>>(queue, 140, 7);
        err += tests_set_portion<sycl::vec<double, 16>, sycl::vec<std::uint32_t, 3>>(queue, 140, 7);
        err += tests_set_portion<sycl::vec<double, 16>, sycl::vec<std::uint32_t, 2>>(queue, 140, 7);
        err += tests_set_portion<sycl::vec<double, 16>, sycl::vec<std::uint32_t, 1>>(queue, 140, 7);
        err += tests_set_portion<sycl::vec<double, 16>, sycl::vec<std::uint32_t, 16>>(queue, 160, 17);
        err += tests_set_portion<sycl::vec<double, 16>, sycl::vec<std::uint32_t, 8>>(queue, 160, 17);
        err += tests_set_portion<sycl::vec<double, 16>, sycl::vec<std::uint32_t, 4>>(queue, 160, 17);
        err += tests_set_portion<sycl::vec<double, 16>, sycl::vec<std::uint32_t, 3>>(queue, 160, 17);
        err += tests_set_portion<sycl::vec<double, 16>, sycl::vec<std::uint32_t, 2>>(queue, 160, 17);
        err += tests_set_portion<sycl::vec<double, 16>, sycl::vec<std::uint32_t, 1>>(queue, 160, 17);
#endif // TEST_LONG_RUN
        EXPECT_TRUE(!err, "Test FAILED");
    }

#endif // TEST_DPCPP_BACKEND_PRESENT && TEST_UNNAMED_LAMBDAS

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT && TEST_UNNAMED_LAMBDAS);
}