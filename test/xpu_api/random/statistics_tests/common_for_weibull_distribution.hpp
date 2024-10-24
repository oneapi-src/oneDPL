// -*- C++ -*-
//===----------------------------------------------------------===//
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
//===----------------------------------------------------------===//

#ifndef _ONEDPL_RANDOM_STATISTICS_TESTS_WEIBULL_COMMON_H
#define _ONEDPL_RANDOM_STATISTICS_TESTS_WEIBULL_COMMON_H

#include <iostream>
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
                res.store(idx.get_linear_id(), acc);
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
                    acc[offset + i] = res[i];
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
    oneapi::dpl::internal::element_type_t<RealType> a_array [nparams] = {2.0, 10.0};
    oneapi::dpl::internal::element_type_t<RealType> b_array [nparams] = {1.0, 10.0};

    // Test for all non-zero parameters
    for(int i = 0; i < nparams; ++i) {
        std::cout << "weibull_distribution test<type>, a = " << a_array[i] << ", b = " << b_array[i] <<
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
    oneapi::dpl::internal::element_type_t<RealType> a_array [nparams] = {2.0, 10.0};
    oneapi::dpl::internal::element_type_t<RealType> b_array [nparams] = {1.0, 10.0};

    // Test for all non-zero parameters
    for(int i = 0; i < nparams; ++i) {
        std::cout << "weibull_distribution test<type>, a = " << a_array[i] << ", b = " << b_array[i] <<
        ", nsamples = " << nsamples << ", part = " << part;
        if(test_portion<RealType, UIntType>(queue, a_array[i], b_array[i], nsamples, part)) {
            return 1;
        }
    }
    return 0;
}

#endif // _ONEDPL_RANDOM_STATISTICS_TESTS_WEIBULL_COMMON_H
