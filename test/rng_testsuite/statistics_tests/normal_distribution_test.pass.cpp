// -*- C++ -*-
//===-- normal_distribution_test.cpp ---------------------------------------===//
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
// Test of normal_distribution - comparison with std::

#if (!defined(_ONEDPL_BACKEND_SYCL) || (_ONEDPL_BACKEND_SYCL == 0))
#include <iostream>

int main() {
    std::cout << "\tTest is skipped for non-SYCL backend. Passed" << std::endl;
    return 0;
}

#else

#include <iostream>
#include <CL/sycl.hpp>
#include <random>
#include <limits>
#include <oneapi/dpl/random>
#include <math.h>

// Engine parameters
#define a 40014u
#define c 200u
#define m 2147483563u
#define seed 777

// Consts
#define eps 0.01
const double pi = std::acos(-1);

template<typename ScalarUintType, typename ScalarRealType>
void generate_std(int num_elems, int nsamples,
    ScalarRealType mean, ScalarRealType stddev, std::vector<ScalarRealType>& std_samples) {
    std::linear_congruential_engine<ScalarUintType, a, c, m> std_engine(seed);
    std::uniform_real_distribution<float> std_distr(0.0, 1.0);

    for(int i = 0; i < nsamples; i += num_elems) {
        for(int j = 0; j < num_elems; j += 2) {
            float u1 = std_distr(std_engine);
            float u2 = std_distr(std_engine);
            float ln = std::log(u1);

            std_samples[i + j] = mean + stddev * (std::sqrt(-2.0 * ln) *
                    std::sin(2.0 * pi * u2));
            if(((j + 1) < num_elems) && (i + j + 1 < nsamples)) {
                std_samples[i + j + 1] = mean + stddev * (std::sqrt(-2.0 * ln) *
                    std::cos(2.0 * pi * u2));
            }
        }
    }

}

template<typename ScalarRealType>
int statistics_check(int nsamples, ScalarRealType mean, ScalarRealType stddev,
    const std::vector<ScalarRealType>& dpstd_samples) {
    // theoretical moments
    double tM = mean;
    double tD = stddev * stddev;
    double tQ = 720.0 * stddev * stddev * stddev * stddev;

    // sample moments
    double sum = 0.0;
    double sum2 = 0.0;
    for(int i = 0; i < nsamples; i++) {
        sum += dpstd_samples[i];
        sum2 += dpstd_samples[i] * dpstd_samples[i];
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
        std::cout << "Error: sample moments (mean= " << sM << ", variance= " << sD << ") disagree with theory (mean=" << tM << ", variance= " << tD << ")";
        return 1;
    }

    return 0;
}

template<class RealType, class UIntType>
int test(oneapi::dpl::internal::element_type_t<RealType> mean, oneapi::dpl::internal::element_type_t<RealType> stddev, int nsamples) {

    sycl::queue queue(sycl::default_selector{});

    // memory allocation
    std::vector<oneapi::dpl::internal::element_type_t<RealType>> std_samples(nsamples);
    std::vector<oneapi::dpl::internal::element_type_t<RealType>> dpstd_samples(nsamples);

    constexpr int num_elems = oneapi::dpl::internal::type_traits_t<RealType>::num_elems == 0 ? 1 : oneapi::dpl::internal::type_traits_t<RealType>::num_elems;
    constexpr int num_to_skip = num_elems % 2 ? num_elems + 1 : num_elems;

    // dpstd generation
    {
        sycl::buffer<oneapi::dpl::internal::element_type_t<RealType>, 1> dpstd_buffer(dpstd_samples.data(), nsamples);

        queue.submit([&](sycl::handler &cgh) {
            auto dpstd_acc = dpstd_buffer.template get_access<sycl::access::mode::write>(cgh);

            cgh.parallel_for<>(sycl::range<1>(nsamples / num_elems),
                    [=](sycl::item<1> idx) {

                unsigned long long offset = idx.get_linear_id() * num_to_skip;
                oneapi::dpl::linear_congruential_engine<UIntType, a, c, m> engine(seed, offset);
                oneapi::dpl::normal_distribution<RealType> distr(mean, stddev);

                sycl::vec<oneapi::dpl::internal::element_type_t<RealType>, num_elems> res = distr(engine);
                res.store(idx.get_linear_id(), dpstd_acc.get_pointer());
            });
        });
        queue.wait();
    }

    // std generation
    generate_std<oneapi::dpl::internal::element_type_t<UIntType>, oneapi::dpl::internal::element_type_t<RealType>>
        (num_elems, nsamples, mean, stddev, std_samples);

    // comparison
    int err = 0;
    for(int i = 0; i < nsamples; ++i) {
        if (abs(std_samples[i] - dpstd_samples[i]) > eps) {
            std::cout << "\nError: std_sample[" << i << "] = " << std_samples[i] << ", dpstd_samples[" << i << "] = " << dpstd_samples[i];
            err++;
        }
    }

    // statistics check
    err += statistics_check(nsamples, mean, stddev, dpstd_samples);

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

    sycl::queue queue(sycl::default_selector{});

    // memory allocation
    std::vector<oneapi::dpl::internal::element_type_t<RealType>> std_samples(nsamples);
    std::vector<oneapi::dpl::internal::element_type_t<RealType>> dpstd_samples(nsamples);
    constexpr unsigned int num_elems = oneapi::dpl::internal::type_traits_t<RealType>::num_elems == 0 ? 1 : oneapi::dpl::internal::type_traits_t<RealType>::num_elems;
    int n_elems = (part >= num_elems) ? num_elems : part;
    int num_to_skip = n_elems % 2 ? n_elems + 1 : n_elems;

    // dpstd generation
    {
        sycl::buffer<oneapi::dpl::internal::element_type_t<RealType>, 1> dpstd_buffer(dpstd_samples.data(), nsamples);

        queue.submit([&](sycl::handler &cgh) {
            auto dpstd_acc = dpstd_buffer.template get_access<sycl::access::mode::write>(cgh);

            cgh.parallel_for<>(sycl::range<1>(nsamples / n_elems),
                    [=](sycl::item<1> idx) {

                unsigned long long offset = idx.get_linear_id() * num_to_skip;
                oneapi::dpl::linear_congruential_engine<UIntType, a, c, m> engine(seed, offset);
                oneapi::dpl::normal_distribution<RealType> distr(mean, stddev);

                sycl::vec<oneapi::dpl::internal::element_type_t<RealType>, num_elems> res = distr(engine, part);
                for(int i = 0; i < n_elems; ++i)
                    dpstd_acc.get_pointer()[idx.get_linear_id() * n_elems + i] = res[i];
            });
        });
        queue.wait_and_throw();
    }

    // std generation
    generate_std<oneapi::dpl::internal::element_type_t<UIntType>, oneapi::dpl::internal::element_type_t<RealType>>
        (n_elems, nsamples, mean, stddev, std_samples);

    // comparison
    int err = 0;
    for(int i = 0; i < nsamples; ++i) {
        if(abs(std_samples[i] - dpstd_samples[i]) > eps) {
            std::cout << "\nError: std_sample[" << i << "] = " << std_samples[i] << ", dpstd_samples[" << i << "] = " << dpstd_samples[i];
            err++;
        }
    }


    // statistics check
    err += statistics_check(nsamples, mean, stddev, dpstd_samples);

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

    oneapi::dpl::internal::element_type_t<RealType> mean_array [nparams] = {0.0, 1.0};
    oneapi::dpl::internal::element_type_t<RealType> stddev_array [nparams] = {1.0, 1000.0};

    int err;
    // Test for all non-zero parameters
    for(int i = 0; i < nparams; ++i) {
        std::cout << "normal_distribution test<type>, mean = " << mean_array[i] << ", stddev = " << stddev_array[i] <<
        ", nsamples = " << nsamples;
        err = test<RealType, UIntType>(mean_array[i], stddev_array[i], nsamples);
        if (err)
            return 1;
    }

    return 0;
}

template<class RealType, class UIntType>
int tests_set_portion(std::int32_t nsamples, unsigned int part) {
    constexpr int nparams = 2;

    oneapi::dpl::internal::element_type_t<RealType> mean_array [nparams] = {0.0, 1.0};
    oneapi::dpl::internal::element_type_t<RealType> stddev_array [nparams] = {1.0, 1000.0};

    int err;
    // Test for all non-zero parameters
    for(int i = 0; i < nparams; ++i) {
        std::cout << "normal_distribution test<type>, mean = " << mean_array[i] << ", stddev = " << stddev_array[i] <<
        ", nsamples = " << nsamples << ", part = "<< part;
        err = test_portion<RealType, UIntType>(mean_array[i], stddev_array[i], nsamples, part);
        if(err) {
            return 1;
        }
    }
    return 0;
}

int main() {
    constexpr int nsamples = 100;
    int err;

    // testing float and std::uint32_t / sycl::vec<std::uint32_t, 16>
    std::cout << "--------------------------------------------------------" << std::endl;
    std::cout << "float, std::uint32_t / sycl::vec<std::uint32_t, 16> type" << std::endl;
    std::cout << "--------------------------------------------------------" << std::endl;
    err = tests_set<float, std::uint32_t>(nsamples);
    err += tests_set<float, sycl::vec<std::uint32_t, 16>>(nsamples);
    if(err) {
        std::cout << "Test FAILED" << std::endl;
        return 1;
    }


    // testing double and std::uint32_t / sycl::vec<std::uint32_t, 16>
    std::cout << "---------------------------------------------------------" << std::endl;
    std::cout << "double, std::uint32_t / sycl::vec<std::uint32_t, 16> type" << std::endl;
    std::cout << "---------------------------------------------------------" << std::endl;
    err = tests_set<double, std::uint32_t>(nsamples);
    err += tests_set<double, sycl::vec<std::uint32_t, 16>>(nsamples);
    if(err) {
        std::cout << "Test FAILED" << std::endl;
        return 1;
    }


    // testing sycl::vec<float, 1> and std::uint32_t / sycl::vec<std::uint32_t, 16>
    std::cout << "---------------------------------------------------------------------" << std::endl;
    std::cout << "sycl::vec<float,1>, std::uint32_t / sycl::vec<std::uint32_t, 16> type" << std::endl;
    std::cout << "---------------------------------------------------------------------" << std::endl;
    err = tests_set<sycl::vec<float, 1>, std::uint32_t>(nsamples);
    err += tests_set<sycl::vec<float, 1>, sycl::vec<std::uint32_t, 16>>(nsamples);
    err += tests_set_portion<sycl::vec<float, 1>, std::uint32_t>(160, 1);
    err += tests_set_portion<sycl::vec<float, 1>, std::uint32_t>(100, 2);
    err += tests_set_portion<sycl::vec<float, 1>, sycl::vec<std::uint32_t, 16>>(160, 1);
    err += tests_set_portion<sycl::vec<float, 1>, sycl::vec<std::uint32_t, 16>>(100, 2);
    if(err) {
        std::cout << "Test FAILED" << std::endl;
        return 1;
    }


    // testing sycl::vec<float, 3> and std::uint32_t / sycl::vec<std::uint32_t, 16>
    std::cout << "---------------------------------------------------------------------" << std::endl;
    std::cout << "sycl::vec<float,3>, std::uint32_t / sycl::vec<std::uint32_t, 16> type" << std::endl;
    std::cout << "---------------------------------------------------------------------" << std::endl;
    err = tests_set<sycl::vec<float, 3>, std::uint32_t>(99);
    err += tests_set<sycl::vec<float, 3>, sycl::vec<std::uint32_t, 16>>(99);
    err += tests_set_portion<sycl::vec<float, 3>, std::uint32_t>(99, 1);
    err += tests_set_portion<sycl::vec<float, 3>, std::uint32_t>(99, 4);
    err += tests_set_portion<sycl::vec<float, 3>, sycl::vec<std::uint32_t, 16>>(99, 1);
    err += tests_set_portion<sycl::vec<float, 3>, sycl::vec<std::uint32_t, 16>>(99, 4);
    if(err) {
        std::cout << "Test FAILED" << std::endl;
        return 1;
    }

    // testing sycl::vec<float, 8> and std::uint32_t / sycl::vec<std::uint32_t, 16>
    std::cout << "---------------------------------------------------------------------" << std::endl;
    std::cout << "sycl::vec<float,8>, std::uint32_t / sycl::vec<std::uint32_t, 16> type" << std::endl;
    std::cout << "---------------------------------------------------------------------" << std::endl;
    err = tests_set<sycl::vec<float, 8>, std::uint32_t>(160);
    err += tests_set<sycl::vec<float, 8>, sycl::vec<std::uint32_t, 16>>(160);
    err += tests_set_portion<sycl::vec<float, 8>, std::uint32_t>(160, 1);
    err += tests_set_portion<sycl::vec<float, 8>, std::uint32_t>(120, 4);
    err += tests_set_portion<sycl::vec<float, 8>, std::uint32_t>(160, 9);
    err += tests_set_portion<sycl::vec<float, 8>, sycl::vec<std::uint32_t, 16>>(160, 1);
    err += tests_set_portion<sycl::vec<float, 8>, sycl::vec<std::uint32_t, 16>>(160, 4);
    err += tests_set_portion<sycl::vec<float, 8>, sycl::vec<std::uint32_t, 16>>(160, 9);
    if(err) {
        std::cout << "Test FAILED" << std::endl;
        return 1;
    }


    // testing sycl::vec<float, 16> and std::uint32_t / sycl::vec<std::uint32_t, 16>
    std::cout << "----------------------------------------------------------------------" << std::endl;
    std::cout << "sycl::vec<float,16>, std::uint32_t / sycl::vec<std::uint32_t, 16> type" << std::endl;
    std::cout << "----------------------------------------------------------------------" << std::endl;
    err = tests_set<sycl::vec<float, 16>, std::uint32_t>(160);
    err += tests_set<sycl::vec<float, 16>, sycl::vec<std::uint32_t, 16>>(160);
    err += tests_set_portion<sycl::vec<float, 16>, std::uint32_t>(160, 1);
    err += tests_set_portion<sycl::vec<float, 16>, std::uint32_t>(140, 7);
    err += tests_set_portion<sycl::vec<float, 16>, std::uint32_t>(160, 17);
    err += tests_set_portion<sycl::vec<float, 16>, sycl::vec<std::uint32_t, 16>>(160, 1);
    err += tests_set_portion<sycl::vec<float, 16>, sycl::vec<std::uint32_t, 16>>(140, 7);
    err += tests_set_portion<sycl::vec<float, 16>, sycl::vec<std::uint32_t, 16>>(160, 16);
    if(err) {
        std::cout << "Test FAILED" << std::endl;
        return 1;
    }

    std::cout << "Test PASSED" << std::endl;
    return 0;
}

#endif // #if (!defined(_ONEDPL_BACKEND_SYCL) || (_ONEDPL_BACKEND_SYCL == 0))
