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
// Test of uniform_int_distribution - check statistical properties of the distribution

#include "support/utils.h"
#include <iostream>

#if TEST_DPCPP_BACKEND_PRESENT && TEST_UNNAMED_LAMBDAS
#include <vector>
#include <CL/sycl.hpp>
#include <random>
#include <oneapi/dpl/random>
#include "statistics_common.h"

// Engine parameters
constexpr auto a = 40014u;
constexpr auto c = 200u;
constexpr auto m = 2147483647u;
constexpr auto seed = 777;
constexpr auto eps = 1;

template<typename IntType>
std::int32_t statistics_check(int nsamples, IntType left, IntType right,
    const std::vector<IntType>& samples)
{
    // theoretical moments
    double tM = (left + right - 1.0) / 2.0;
    double tD = ((right - left) * (right - left) - 1.0) / 12.0;
    double tQ = (((right - left) * (right - left)) * ((1.0 / 80.0) * (right - left) * (right - left) -
        (1.0 / 24.0))) + (7.0 / 240.0);

    return compare_moments(nsamples, samples, tM, tD, tQ);
}

template<class IntType, class UIntType>
int test(oneapi::dpl::internal::element_type_t<IntType> left, oneapi::dpl::internal::element_type_t<IntType> right, int nsamples) {

    sycl::queue queue;

    // memory allocation
    std::vector<oneapi::dpl::internal::element_type_t<IntType>> samples(nsamples);

    constexpr std::int32_t num_elems = oneapi::dpl::internal::type_traits_t<IntType>::num_elems == 0 ? 1 :
        oneapi::dpl::internal::type_traits_t<IntType>::num_elems;

    // generation
    {
        sycl::buffer<oneapi::dpl::internal::element_type_t<IntType>, 1> buffer(samples.data(), nsamples);

        queue.submit([&](sycl::handler &cgh) {
            sycl::accessor acc(buffer, cgh, sycl::write_only);

            cgh.parallel_for<>(sycl::range<1>(nsamples / num_elems),
                    [=](sycl::item<1> idx) {

                unsigned long long offset = idx.get_linear_id() * num_elems;
                oneapi::dpl::linear_congruential_engine<UIntType, a, c, m> engine(seed, offset);
                oneapi::dpl::uniform_int_distribution<IntType> distr(left, right);

                sycl::vec<oneapi::dpl::internal::element_type_t<IntType>, num_elems> res = distr(engine);
                res.store(idx.get_linear_id(), acc.get_pointer());
            });
        });
        queue.wait();
    }

    // statistics check
    int err = statistics_check(nsamples, left, right, samples);

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

    sycl::queue queue;

    // memory allocation
    std::vector<oneapi::dpl::internal::element_type_t<IntType>> samples(nsamples);
    constexpr unsigned int num_elems = oneapi::dpl::internal::type_traits_t<IntType>::num_elems == 0 ? 1 : oneapi::dpl::internal::type_traits_t<IntType>::num_elems;
    int n_elems = (part >= num_elems) ? num_elems : part;

    // generation
    {
        sycl::buffer<oneapi::dpl::internal::element_type_t<IntType>, 1> buffer(samples.data(), nsamples);

        queue.submit([&](sycl::handler &cgh) {
            sycl::accessor acc(buffer, cgh, sycl::write_only);

            cgh.parallel_for<>(sycl::range<1>(nsamples / n_elems),
                    [=](sycl::item<1> idx) {

                unsigned long long offset = idx.get_linear_id() * n_elems;
                oneapi::dpl::linear_congruential_engine<UIntType, a, c, m> engine(seed, offset);
                oneapi::dpl::uniform_int_distribution<IntType> distr(left, right);

                sycl::vec<oneapi::dpl::internal::element_type_t<IntType>, num_elems> res = distr(engine, part);
                for(int i = 0; i < n_elems; ++i)
                    acc.get_pointer()[offset + i] = res[i];
            });
        });
        queue.wait_and_throw();
    }

    // statistics check
    int err = statistics_check(nsamples, left, right, samples);

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

    oneapi::dpl::internal::element_type_t<IntType> left = 0;
    oneapi::dpl::internal::element_type_t<IntType> right = 1000;

    // Test for all non-zero parameters
    std::cout << "uniform_int_distribution test<type>, left = " << left << ", right = " << right <<
    ", nsamples = " << nsamples;
    if(test<IntType, UIntType>(left, right, nsamples))
        return 1;
    return 0;
}

template<class IntType, class UIntType>
int tests_set_portion(int nsamples, unsigned int part) {

    oneapi::dpl::internal::element_type_t<IntType> left = 0;
    oneapi::dpl::internal::element_type_t<IntType> right = 1000;

    // Test for all non-zero parameters
    std::cout << "uniform_int_distribution test<type>, left = " << left << ", right = " << right <<
    ", nsamples = " << nsamples <<  ", part = " << part;
    if(test_portion<IntType, UIntType>(left, right, nsamples, part))
        return 1;
    return 0;
}

#endif // TEST_DPCPP_BACKEND_PRESENT && TEST_UNNAMED_LAMBDAS

int main() {

#if TEST_DPCPP_BACKEND_PRESENT && TEST_UNNAMED_LAMBDAS

    constexpr int nsamples = 100;
    int err = 0;

    // testing std::int32_t and std::uint32_t ... sycl::vec<std::uint32_t, 16>
    std::cout << "--------------------------------------------------------" << std::endl;
    std::cout << "std::int32_t, std::uint32_t ... sycl::vec<std::uint32_t, 16> type" << std::endl;
    std::cout << "--------------------------------------------------------" << std::endl;
    err += tests_set<std::int32_t, std::uint32_t>(nsamples);
#if TEST_LONG_RUN
    err += tests_set<std::int32_t, sycl::vec<std::uint32_t, 16>>(nsamples);
    err += tests_set<std::int32_t, sycl::vec<std::uint32_t, 8>>(nsamples);
    err += tests_set<std::int32_t, sycl::vec<std::uint32_t, 4>>(nsamples);
    err += tests_set<std::int32_t, sycl::vec<std::uint32_t, 3>>(nsamples);
    err += tests_set<std::int32_t, sycl::vec<std::uint32_t, 2>>(nsamples);
    err += tests_set<std::int32_t, sycl::vec<std::uint32_t, 1>>(nsamples);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // testing sycl::vec<std::int32_t, 1> and std::uint32_t ... sycl::vec<std::uint32_t, 3>
    std::cout << "---------------------------------------------------------------------" << std::endl;
    std::cout << "sycl::vec<std::int32_t,1>, std::uint32_t ... sycl::vec<std::uint32_t, 3> type" << std::endl;
    std::cout << "---------------------------------------------------------------------" << std::endl;
    err = tests_set<sycl::vec<std::int32_t, 1>, std::uint32_t>(nsamples);
#if TEST_LONG_RUN
    err += tests_set_portion<sycl::vec<std::int32_t, 1>, std::uint32_t>(100, 1);
    err += tests_set_portion<sycl::vec<std::int32_t, 1>, std::uint32_t>(100, 2);
    err += tests_set_portion<sycl::vec<std::int32_t, 1>, sycl::vec<std::uint32_t, 3>>(100, 1);
    err += tests_set_portion<sycl::vec<std::int32_t, 1>, sycl::vec<std::uint32_t, 2>>(100, 1);
    err += tests_set_portion<sycl::vec<std::int32_t, 1>, sycl::vec<std::uint32_t, 1>>(100, 1);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // testing sycl::vec<std::int32_t, 2> and std::uint32_t, sycl::vec<std::uint32_t, 3>
    std::cout << "------------------------------------------------------------------" << std::endl;
    std::cout << "sycl::vec<std::int32_t,2>, std::uint32_t, sycl::vec<std::uint32_t, 3> type" << std::endl;
    std::cout << "------------------------------------------------------------------" << std::endl;
    err = tests_set<sycl::vec<std::int32_t, 2>, std::uint32_t>(nsamples);
#if TEST_LONG_RUN
    err += tests_set<sycl::vec<std::int32_t, 2>, sycl::vec<std::uint32_t, 3>>(100);
    err += tests_set_portion<sycl::vec<std::int32_t, 2>, std::uint32_t>(100, 1);
    err += tests_set_portion<sycl::vec<std::int32_t, 2>, std::uint32_t>(100, 3);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // testing sycl::vec<std::int32_t, 3> and std::uint32_t, sycl::vec<std::uint32_t, 2>, sycl::vec<std::uint32_t, 4>
    std::cout << "----------------------------------------------------------------------------------------------" << std::endl;
    std::cout << "sycl::vec<std::int32_t,3>, std::uint32_t, sycl::vec<std::uint32_t, 2>, sycl::vec<std::uint32_t, 4> type" << std::endl;
    std::cout << "----------------------------------------------------------------------------------------------" << std::endl;
    err = tests_set<sycl::vec<std::int32_t, 3>, std::uint32_t>(99);
#if TEST_LONG_RUN
    err += tests_set<sycl::vec<std::int32_t, 3>, sycl::vec<std::uint32_t, 2>>(100);
    err += tests_set<sycl::vec<std::int32_t, 3>, sycl::vec<std::uint32_t, 4>>(100);
    err += tests_set_portion<sycl::vec<std::int32_t, 3>, std::uint32_t>(99, 1);
    err += tests_set_portion<sycl::vec<std::int32_t, 3>, std::uint32_t>(99, 4);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // testing sycl::vec<std::int32_t, 4> and std::uint32_t, sycl::vec<std::uint32_t, 3>
    std::cout << "-------------------------------------------------------------------" << std::endl;
    std::cout << "sycl::vec<std::int32_t,4>, std::uint32_t, sycl::vec<std::uint32_t, 3> type" << std::endl;
    std::cout << "-------------------------------------------------------------------" << std::endl;
    err = tests_set<sycl::vec<std::int32_t, 4>, std::uint32_t>(100);
#if TEST_LONG_RUN
    err += tests_set<sycl::vec<std::int32_t, 4>, sycl::vec<std::uint32_t, 3>>(100);
    err += tests_set_portion<sycl::vec<std::int32_t, 4>, std::uint32_t>(100, 1);
    err += tests_set_portion<sycl::vec<std::int32_t, 4>, std::uint32_t>(100, 5);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // testing sycl::vec<std::int32_t, 8> and std::uint32_t
    std::cout << "-------------------------------------" << std::endl;
    std::cout << "sycl::vec<std::int32_t,8>, std::uint32_t type" << std::endl;
    std::cout << "-------------------------------------" << std::endl;
    err = tests_set<sycl::vec<std::int32_t, 8>, std::uint32_t>(160);
#if TEST_LONG_RUN
    err += tests_set_portion<sycl::vec<std::int32_t, 8>, std::uint32_t>(160, 1);
    err += tests_set_portion<sycl::vec<std::int32_t, 8>, std::uint32_t>(160, 5);
    err += tests_set_portion<sycl::vec<std::int32_t, 8>, std::uint32_t>(160, 9);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // testing sycl::vec<std::int32_t, 16> and std::uint32_t
    std::cout << "--------------------------------------" << std::endl;
    std::cout << "sycl::vec<std::int32_t,16>, std::uint32_t type" << std::endl;
    std::cout << "--------------------------------------" << std::endl;
    err = tests_set<sycl::vec<std::int32_t, 16>, std::uint32_t>(160);
#if TEST_LONG_RUN
    err += tests_set_portion<sycl::vec<std::int32_t, 16>, std::uint32_t>(160, 1);
    err += tests_set_portion<sycl::vec<std::int32_t, 16>, std::uint32_t>(140, 7);
    err += tests_set_portion<sycl::vec<std::int32_t, 16>, std::uint32_t>(160, 17);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

#endif // TEST_DPCPP_BACKEND_PRESENT && TEST_UNNAMED_LAMBDAS

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT && TEST_UNNAMED_LAMBDAS);
}
