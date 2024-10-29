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

#ifndef _ONEDPL_RANDOM_STATISTICS_TESTS_COMMON_FOR_PHILOX_UNIFORM_REAL_H
#define _ONEDPL_RANDOM_STATISTICS_TESTS_COMMON_FOR_PHILOX_UNIFORM_REAL_H

#include <iostream>
#include <vector>
#include <random>
#include <oneapi/dpl/random>

#include "statistics_common.h"

template <typename RealType>
std::int32_t
statistics_check(int nsamples, RealType left, RealType right, const std::vector<RealType>& samples)
{
    // theoretical moments
    double tM = (right + left) / 2.0;
    double tD = ((right - left) * (right - left)) / 12.0;
    double tQ = ((right - left) * (right - left) * (right - left) * (right - left)) / 80.0;

    return compare_moments(nsamples, samples, tM, tD, tQ);
}

template <typename RealType, typename UIntType, typename Engine>
int
test(sycl::queue& queue, oneapi::dpl::internal::element_type_t<RealType> left,
     oneapi::dpl::internal::element_type_t<RealType> right, int nsamples)
{

    // memory allocation
    std::vector<oneapi::dpl::internal::element_type_t<RealType>> samples(nsamples);

    constexpr int num_elems = oneapi::dpl::internal::type_traits_t<RealType>::num_elems == 0
                                  ? 1
                                  : oneapi::dpl::internal::type_traits_t<RealType>::num_elems;
    // generation
    {
        sycl::buffer<oneapi::dpl::internal::element_type_t<RealType>> buffer(samples.data(), nsamples);

        queue.submit([&](sycl::handler& cgh) {
            sycl::accessor acc(buffer, cgh, sycl::write_only);

            cgh.parallel_for<>(sycl::range<1>(nsamples / num_elems), [=](sycl::item<1> idx) {
                unsigned long long offset = idx.get_linear_id() * num_elems;
                Engine engine;
                engine.discard(offset);
                oneapi::dpl::uniform_real_distribution<RealType> distr(left, right);

                sycl::vec<oneapi::dpl::internal::element_type_t<RealType>, num_elems> res = distr(engine);
                res.store(idx.get_linear_id(), acc);
            });
        });
        queue.wait();
    }

    // statistics check
    int err = statistics_check(nsamples, left, right, samples);

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

template <typename RealType, typename UIntType, typename Engine>
int
test_portion(sycl::queue& queue, oneapi::dpl::internal::element_type_t<RealType> left,
             oneapi::dpl::internal::element_type_t<RealType> right, int nsamples, unsigned int part)
{

    // memory allocation
    std::vector<oneapi::dpl::internal::element_type_t<RealType>> samples(nsamples);
    constexpr int num_elems = oneapi::dpl::internal::type_traits_t<RealType>::num_elems == 0
                                  ? 1
                                  : oneapi::dpl::internal::type_traits_t<RealType>::num_elems;
    int n_elems = (part >= num_elems) ? num_elems : part;

    // generation
    {
        sycl::buffer<oneapi::dpl::internal::element_type_t<RealType>> buffer(samples.data(), nsamples);

        queue.submit([&](sycl::handler& cgh) {
            sycl::accessor acc(buffer, cgh, sycl::write_only);

            cgh.parallel_for<>(sycl::range<1>(nsamples / n_elems), [=](sycl::item<1> idx) {
                unsigned long long offset = idx.get_linear_id() * n_elems;
                Engine engine;
                engine.discard(offset);
                oneapi::dpl::uniform_real_distribution<RealType> distr(left, right);

                sycl::vec<oneapi::dpl::internal::element_type_t<RealType>, num_elems> res = distr(engine, part);
                for (int i = 0; i < n_elems; ++i)
                    acc[offset + i] = res[i];
            });
        });
        queue.wait();
    }

    // statistics check
    int err = statistics_check(nsamples, left, right, samples);

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

template <typename RealType, typename UIntType, typename Engine>
int
tests_set(sycl::queue& queue, int nsamples)
{

    constexpr int nparams = 2;
    oneapi::dpl::internal::element_type_t<RealType> left_array[nparams] = {0.0, -10.0};
    oneapi::dpl::internal::element_type_t<RealType> right_array[nparams] = {1.0, 10.0};

    // Test for all non-zero parameters
    for (int i = 0; i < nparams; ++i)
    {
        std::cout << "uniform_real_distribution test<type>, left = " << left_array[i] << ", right = " << right_array[i]
                  << ", nsamples  = " << nsamples;
        if (test<RealType, UIntType, Engine>(queue, left_array[i], right_array[i], nsamples))
        {
            return 1;
        }
    }
    return 0;
}

template <typename RealType, typename UIntType, typename Engine>
int
tests_set_portion(sycl::queue& queue, int nsamples, unsigned int part)
{

    constexpr int nparams = 2;
    oneapi::dpl::internal::element_type_t<RealType> left_array[nparams] = {0.0, -10.0};
    oneapi::dpl::internal::element_type_t<RealType> right_array[nparams] = {1.0, 10.0};

    // Test for all non-zero parameters
    for (int i = 0; i < nparams; ++i)
    {
        std::cout << "uniform_real_distribution test<type>, left = " << left_array[i] << ", right = " << right_array[i]
                  << ", nsamples = " << nsamples << ", part = " << part;
        if (test_portion<RealType, UIntType, Engine>(queue, left_array[i], right_array[i], nsamples, part))
        {
            return 1;
        }
    }
    return 0;
}

#endif // _ONEDPL_RANDOM_STATISTICS_TESTS_COMMON_FOR_PHILOX_UNIFORM_REAL_H
