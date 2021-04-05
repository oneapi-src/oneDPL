// -*- C++ -*-
//===-- distributions_methods.pass.cpp -------------------------------------===//
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
// Testing of different distributions' methods

#include "support/utils.h"

#if TEST_DPCPP_BACKEND_PRESENT && __SYCL_UNNAMED_LAMBDA__
#    include <iostream>
#    include <cmath>
#    include <vector>
#    include <CL/sycl.hpp>
#    include <oneapi/dpl/random>

constexpr auto SEED = 777;
constexpr auto N_GEN = 1000;

template <typename T>
using Element_type = typename oneapi::dpl::internal::type_traits_t<T>::element_type;

template <class T>
std::int32_t
check_params(oneapi::dpl::uniform_int_distribution<T>& distr, Element_type<T> a, Element_type<T> b)
{
    return ((distr.a() != a) || (distr.b() != b) || (distr.min() != a) || (distr.max() != b) ||
            (distr.param().first != a) || (distr.param().second != b));
}

template <class T>
std::int32_t
check_params(oneapi::dpl::uniform_real_distribution<T>& distr, Element_type<T> a, Element_type<T> b)
{
    return ((distr.a() != a) || (distr.b() != b) || (distr.min() != a) || (distr.max() != b) ||
            (distr.param().first != a) || (distr.param().second != b));
}

template <class T>
std::int32_t
check_params(oneapi::dpl::normal_distribution<T>& distr, Element_type<T> mean, Element_type<T> stddev)
{
    return ((distr.mean() != mean) || (distr.stddev() != stddev) || std::isinf(distr.min()) ||
            std::isinf(distr.max()) || (distr.param().first != mean) || (distr.param().second != stddev));
}

template <class Distr>
bool
test()
{
    // Catch asynchronous exceptions
    auto exception_handler = [](sycl::exception_list exceptions) {
        for (std::exception_ptr const& e : exceptions)
        {
            try
            {
                std::rethrow_exception(e);
            }
            catch (sycl::exception const& e)
            {
                std::cout << "Caught asynchronous SYCL exception during generation:\n" << e.what() << std::endl;
            }
        }
    };

    typename Distr::param_type params1(static_cast<typename Distr::scalar_type>(0),
                                       static_cast<typename Distr::scalar_type>(10));
    typename Distr::param_type params2(static_cast<typename Distr::scalar_type>(2),
                                       static_cast<typename Distr::scalar_type>(8));

    sycl::queue queue(sycl::default_selector{}, exception_handler);
    int sum = 0;

    // Memory allocation
    std::vector<typename Distr::scalar_type> dpstd_res(N_GEN);
    constexpr std::int32_t num_elems =
        oneapi::dpl::internal::type_traits_t<typename Distr::result_type>::num_elems == 0
            ? 1
            : oneapi::dpl::internal::type_traits_t<typename Distr::result_type>::num_elems;

    // Random number generation
    {
        sycl::buffer<typename Distr::scalar_type> dpstd_buffer(dpstd_res.data(), dpstd_res.size());

        try
        {

            queue.submit([&](sycl::handler& cgh) {
                auto dpstd_acc = dpstd_buffer.template get_access<sycl::access::mode::write>(cgh);

                cgh.parallel_for<>(sycl::range<1>(N_GEN / 4), [=](sycl::item<1> idx) {
                    unsigned long long offset = idx.get_linear_id();
                    oneapi::dpl::minstd_rand engine(SEED, offset);
                    Distr d1;
                    d1.param(params1);
                    Distr d2(params2);
                    d2.reset();
                    typename Distr::scalar_type res0;
                    typename Distr::scalar_type res1;
                    if constexpr (std::is_same<typename Distr::result_type,
                                               sycl::vec<typename Distr::scalar_type, num_elems>>::value)
                    {
                        res0 = d1(engine, params2, 1)[0];
                        res1 = d1(engine, params1, 1)[0];
                    }
                    else
                    {
                        res0 = d1(engine, params2);
                        res1 = d1(engine, params1);
                    }
                    dpstd_acc[offset * 2] = res0;
                    dpstd_acc[offset * 2 + 1] = res1;
                });
            });
        }
        catch (sycl::exception const& e)
        {
            std::cout << "\t\tSYCL exception during generation\n"
                      << e.what() << std::endl
                      << "OpenCL status: " << e.get_cl_code() << std::endl;
            return 1;
        }

        queue.wait_and_throw();
        Distr distr;
        // 0 and 1 are default parameters for uniform_int, uniform_real and normal distributions
        check_params(distr, static_cast<typename Distr::scalar_type>(0), static_cast<typename Distr::scalar_type>(1));
    }

    return sum;
}

#endif // TEST_DPCPP_BACKEND_PRESENT && __SYCL_UNNAMED_LAMBDA__

int
main()
{

#if TEST_DPCPP_BACKEND_PRESENT && __SYCL_UNNAMED_LAMBDA__

    std::int32_t err = 0;

    std::cout << "---------------------------------------------------" << std::endl;
    std::cout << "uniform_int_distribution<std::int32_t>" << std::endl;
    std::cout << "---------------------------------------------------" << std::endl;
    err += test<oneapi::dpl::uniform_int_distribution<std::int32_t>>();
    err += test<oneapi::dpl::uniform_int_distribution<sycl::vec<std::int32_t, 1>>>();
    err += test<oneapi::dpl::uniform_int_distribution<sycl::vec<std::int32_t, 2>>>();
    err += test<oneapi::dpl::uniform_int_distribution<sycl::vec<std::int32_t, 3>>>();
    err += test<oneapi::dpl::uniform_int_distribution<sycl::vec<std::int32_t, 4>>>();
    err += test<oneapi::dpl::uniform_int_distribution<sycl::vec<std::int32_t, 8>>>();
    err += test<oneapi::dpl::uniform_int_distribution<sycl::vec<std::int32_t, 16>>>();
    EXPECT_TRUE(!err, "Test FAILED");

    std::cout << "---------------------------------------------------" << std::endl;
    std::cout << "uniform_int_distribution<std::uint32_t>" << std::endl;
    std::cout << "---------------------------------------------------" << std::endl;
    err += test<oneapi::dpl::uniform_int_distribution<std::uint32_t>>();
    err += test<oneapi::dpl::uniform_int_distribution<sycl::vec<std::uint32_t, 1>>>();
    err += test<oneapi::dpl::uniform_int_distribution<sycl::vec<std::uint32_t, 2>>>();
    err += test<oneapi::dpl::uniform_int_distribution<sycl::vec<std::uint32_t, 3>>>();
    err += test<oneapi::dpl::uniform_int_distribution<sycl::vec<std::uint32_t, 4>>>();
    err += test<oneapi::dpl::uniform_int_distribution<sycl::vec<std::uint32_t, 8>>>();
    err += test<oneapi::dpl::uniform_int_distribution<sycl::vec<std::uint32_t, 16>>>();
    EXPECT_TRUE(!err, "Test FAILED");

    std::cout << "---------------------------------------------------" << std::endl;
    std::cout << "uniform_real_distribution<float>" << std::endl;
    std::cout << "---------------------------------------------------" << std::endl;
    err += test<oneapi::dpl::uniform_real_distribution<float>>();
    err += test<oneapi::dpl::uniform_real_distribution<sycl::vec<float, 1>>>();
    err += test<oneapi::dpl::uniform_real_distribution<sycl::vec<float, 2>>>();
    err += test<oneapi::dpl::uniform_real_distribution<sycl::vec<float, 3>>>();
    err += test<oneapi::dpl::uniform_real_distribution<sycl::vec<float, 4>>>();
    err += test<oneapi::dpl::uniform_real_distribution<sycl::vec<float, 8>>>();
    err += test<oneapi::dpl::uniform_real_distribution<sycl::vec<float, 16>>>();
    EXPECT_TRUE(!err, "Test FAILED");

    std::cout << "---------------------------------------------------" << std::endl;
    std::cout << "uniform_real_distribution<double>" << std::endl;
    std::cout << "---------------------------------------------------" << std::endl;
    err += test<oneapi::dpl::uniform_real_distribution<double>>();
    err += test<oneapi::dpl::uniform_real_distribution<sycl::vec<double, 1>>>();
    err += test<oneapi::dpl::uniform_real_distribution<sycl::vec<double, 2>>>();
    err += test<oneapi::dpl::uniform_real_distribution<sycl::vec<double, 3>>>();
    err += test<oneapi::dpl::uniform_real_distribution<sycl::vec<double, 4>>>();
    err += test<oneapi::dpl::uniform_real_distribution<sycl::vec<double, 8>>>();
    err += test<oneapi::dpl::uniform_real_distribution<sycl::vec<double, 16>>>();
    EXPECT_TRUE(!err, "Test FAILED");

    std::cout << "---------------------------------------------------" << std::endl;
    std::cout << "normal_distribution<float>" << std::endl;
    std::cout << "---------------------------------------------------" << std::endl;
    err += test<oneapi::dpl::normal_distribution<float>>();
    err += test<oneapi::dpl::normal_distribution<sycl::vec<float, 1>>>();
    err += test<oneapi::dpl::normal_distribution<sycl::vec<float, 2>>>();
    err += test<oneapi::dpl::normal_distribution<sycl::vec<float, 3>>>();
    err += test<oneapi::dpl::normal_distribution<sycl::vec<float, 4>>>();
    err += test<oneapi::dpl::normal_distribution<sycl::vec<float, 8>>>();
    err += test<oneapi::dpl::normal_distribution<sycl::vec<float, 16>>>();
    EXPECT_TRUE(!err, "Test FAILED");

    std::cout << "---------------------------------------------------" << std::endl;
    std::cout << "normal_distribution<double>" << std::endl;
    std::cout << "---------------------------------------------------" << std::endl;
    err += test<oneapi::dpl::normal_distribution<double>>();
    err += test<oneapi::dpl::normal_distribution<sycl::vec<double, 1>>>();
    err += test<oneapi::dpl::normal_distribution<sycl::vec<double, 2>>>();
    err += test<oneapi::dpl::normal_distribution<sycl::vec<double, 3>>>();
    err += test<oneapi::dpl::normal_distribution<sycl::vec<double, 4>>>();
    err += test<oneapi::dpl::normal_distribution<sycl::vec<double, 8>>>();
    err += test<oneapi::dpl::normal_distribution<sycl::vec<double, 16>>>();
    EXPECT_TRUE(!err, "Test FAILED");

#endif // TEST_DPCPP_BACKEND_PRESENT && __SYCL_UNNAMED_LAMBDA__

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
