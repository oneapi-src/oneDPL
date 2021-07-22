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

#if TEST_DPCPP_BACKEND_PRESENT && TEST_UNNAMED_LAMBDAS
#    include <iostream>
#    include <cmath>
#    include <vector>
#    include <CL/sycl.hpp>
#    include <oneapi/dpl/random>

constexpr auto SEED = 777;
constexpr auto N_GEN = 960;

template <typename T>
using Element_type = typename oneapi::dpl::internal::type_traits_t<T>::element_type;

template <class T>
std::int32_t
check_params(oneapi::dpl::uniform_int_distribution<T>& distr)
{
    Element_type<T> a = Element_type<T>{0};
    Element_type<T> b = std::numeric_limits<Element_type<T>>::max();
    return ((distr.a() != a) || (distr.b() != b) || (distr.min() != a) || (distr.max() != b) ||
            (distr.param().first != a) || (distr.param().second != b));
}

template <class T>
std::int32_t
check_params(oneapi::dpl::uniform_real_distribution<T>& distr)
{
    Element_type<T> a = Element_type<T>{0};
    Element_type<T> b = Element_type<T>{1};
    return ((distr.a() != a) || (distr.b() != b) || (distr.min() != a) || (distr.max() != b) ||
            (distr.param().first != a) || (distr.param().second != b));
}

template <class T>
std::int32_t
check_params(oneapi::dpl::normal_distribution<T>& distr)
{
    Element_type<T> mean = Element_type<T>{0};
    Element_type<T> stddev = Element_type<T>{1};
    return ((distr.mean() != mean) || (distr.stddev() != stddev) ||
            (distr.min() > -std::numeric_limits<Element_type<T>>::max()) ||
            (distr.max() < std::numeric_limits<Element_type<T>>::max()) || (distr.param().first != mean) ||
            (distr.param().second != stddev));
}

template <class T>
std::int32_t
check_params(oneapi::dpl::exponential_distribution<T>& distr)
{
    Element_type<T> lambda = Element_type<T>{1};
    return ((distr.lambda() != lambda) || (distr.min() != 0) ||
            (distr.max() < std::numeric_limits<Element_type<T>>::max()) || 
            (distr.param().lambda != lambda));
}

template <typename Distr>
typename ::std::enable_if<::std::is_same<typename Distr::param_type,
                                         ::std::pair<typename Distr::scalar_type, typename Distr::scalar_type>>::value,
                          void>::type
make_param(typename Distr::param_type& params1, typename Distr::param_type& params2)
{
    params1 =
        ::std::make_pair(typename Distr::scalar_type{0}, typename Distr::scalar_type{10});
    params2 =
        ::std::make_pair(typename Distr::scalar_type{2}, typename Distr::scalar_type{8});
}

template <typename Distr>
typename ::std::enable_if<!::std::is_same<typename Distr::param_type,
                                          ::std::pair<typename Distr::scalar_type, typename Distr::scalar_type>>::value,
                          void>::type
make_param(typename Distr::param_type& params1, typename Distr::param_type& params2)
{
    params1 = typename Distr::scalar_type{1};
    params2 = typename Distr::scalar_type{2};
}

template <class Distr>
bool
test_vec()
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

    typename Distr::param_type params1;
    typename Distr::param_type params2;

    make_param<Distr>(params1, params2);

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

                cgh.parallel_for<>(sycl::range<1>(N_GEN / (2 * num_elems)), [=](sycl::item<1> idx) {
                    unsigned long long offset = idx.get_linear_id() * num_elems;
                    oneapi::dpl::minstd_rand engine(SEED, offset);
                    Distr d1;
                    d1.param(params1);
                    Distr d2(params2);
                    d2.reset();
                    typename Distr::result_type res0 = d1(engine, params2, 1);
                    typename Distr::result_type res1 = d1(engine, params1, 1);
                    for (int i = 0; i < num_elems; ++i)
                    {
                        dpstd_acc[offset * 2 + i] = res0[i];
                        dpstd_acc[offset * 2 + num_elems + i] = res1[i];
                    }
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
        sum += check_params(distr);
    }

    return sum;
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

    typename Distr::param_type params1;
    typename Distr::param_type params2;

    make_param<Distr>(params1, params2);

    sycl::queue queue(sycl::default_selector{}, exception_handler);
    int sum = 0;

    // Memory allocation
    std::vector<typename Distr::scalar_type> dpstd_res(N_GEN);

    // Random number generation
    {
        sycl::buffer<typename Distr::scalar_type> dpstd_buffer(dpstd_res.data(), dpstd_res.size());

        try
        {

            queue.submit([&](sycl::handler& cgh) {
                auto dpstd_acc = dpstd_buffer.template get_access<sycl::access::mode::write>(cgh);

                cgh.parallel_for<>(sycl::range<1>(N_GEN / 2), [=](sycl::item<1> idx) {
                    unsigned long long offset = idx.get_linear_id();
                    oneapi::dpl::minstd_rand engine(SEED, offset);
                    Distr d1;
                    d1.param(params1);
                    Distr d2(params2);
                    d2.reset();
                    typename Distr::scalar_type res0 = d1(engine, params2);
                    typename Distr::scalar_type res1 = d1(engine, params1);
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
        sum += check_params(distr);
    }

    return sum;
}

#endif // TEST_DPCPP_BACKEND_PRESENT && TEST_UNNAMED_LAMBDAS

int
main()
{

#if TEST_DPCPP_BACKEND_PRESENT && TEST_UNNAMED_LAMBDAS

    std::int32_t err = 0;

    std::cout << "---------------------------------------------------" << std::endl;
    std::cout << "uniform_int_distribution<std::int32_t>" << std::endl;
    std::cout << "---------------------------------------------------" << std::endl;
    err += test<oneapi::dpl::uniform_int_distribution<std::int32_t>>();
#    if TEST_LONG_RUN
    err += test_vec<oneapi::dpl::uniform_int_distribution<sycl::vec<std::int32_t, 16>>>();
    err += test_vec<oneapi::dpl::uniform_int_distribution<sycl::vec<std::int32_t, 8>>>();
    err += test_vec<oneapi::dpl::uniform_int_distribution<sycl::vec<std::int32_t, 4>>>();
    err += test_vec<oneapi::dpl::uniform_int_distribution<sycl::vec<std::int32_t, 3>>>();
    err += test_vec<oneapi::dpl::uniform_int_distribution<sycl::vec<std::int32_t, 2>>>();
    err += test_vec<oneapi::dpl::uniform_int_distribution<sycl::vec<std::int32_t, 1>>>();
#    endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    std::cout << "---------------------------------------------------" << std::endl;
    std::cout << "uniform_int_distribution<std::uint32_t>" << std::endl;
    std::cout << "---------------------------------------------------" << std::endl;
    err += test<oneapi::dpl::uniform_int_distribution<std::uint32_t>>();
#    if TEST_LONG_RUN
    err += test_vec<oneapi::dpl::uniform_int_distribution<sycl::vec<std::uint32_t, 16>>>();
    err += test_vec<oneapi::dpl::uniform_int_distribution<sycl::vec<std::uint32_t, 8>>>();
    err += test_vec<oneapi::dpl::uniform_int_distribution<sycl::vec<std::uint32_t, 4>>>();
    err += test_vec<oneapi::dpl::uniform_int_distribution<sycl::vec<std::uint32_t, 3>>>();
    err += test_vec<oneapi::dpl::uniform_int_distribution<sycl::vec<std::uint32_t, 2>>>();
    err += test_vec<oneapi::dpl::uniform_int_distribution<sycl::vec<std::uint32_t, 1>>>();
#    endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    std::cout << "---------------------------------------------------" << std::endl;
    std::cout << "uniform_real_distribution<float>" << std::endl;
    std::cout << "---------------------------------------------------" << std::endl;
    err += test<oneapi::dpl::uniform_real_distribution<float>>();
#    if TEST_LONG_RUN
    err += test_vec<oneapi::dpl::uniform_real_distribution<sycl::vec<float, 16>>>();
    err += test_vec<oneapi::dpl::uniform_real_distribution<sycl::vec<float, 8>>>();
    err += test_vec<oneapi::dpl::uniform_real_distribution<sycl::vec<float, 4>>>();
    err += test_vec<oneapi::dpl::uniform_real_distribution<sycl::vec<float, 3>>>();
    err += test_vec<oneapi::dpl::uniform_real_distribution<sycl::vec<float, 2>>>();
    err += test_vec<oneapi::dpl::uniform_real_distribution<sycl::vec<float, 1>>>();
#    endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    std::cout << "---------------------------------------------------" << std::endl;
    std::cout << "uniform_real_distribution<double>" << std::endl;
    std::cout << "---------------------------------------------------" << std::endl;
    err += test<oneapi::dpl::uniform_real_distribution<double>>();
#    if TEST_LONG_RUN
    err += test_vec<oneapi::dpl::uniform_real_distribution<sycl::vec<double, 16>>>();
    err += test_vec<oneapi::dpl::uniform_real_distribution<sycl::vec<double, 8>>>();
    err += test_vec<oneapi::dpl::uniform_real_distribution<sycl::vec<double, 4>>>();
    err += test_vec<oneapi::dpl::uniform_real_distribution<sycl::vec<double, 3>>>();
    err += test_vec<oneapi::dpl::uniform_real_distribution<sycl::vec<double, 2>>>();
    err += test_vec<oneapi::dpl::uniform_real_distribution<sycl::vec<double, 1>>>();
#    endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    std::cout << "---------------------------------------------------" << std::endl;
    std::cout << "normal_distribution<float>" << std::endl;
    std::cout << "---------------------------------------------------" << std::endl;
    err += test<oneapi::dpl::normal_distribution<float>>();
#    if TEST_LONG_RUN
    err += test_vec<oneapi::dpl::normal_distribution<sycl::vec<float, 16>>>();
    err += test_vec<oneapi::dpl::normal_distribution<sycl::vec<float, 8>>>();
    err += test_vec<oneapi::dpl::normal_distribution<sycl::vec<float, 4>>>();
    err += test_vec<oneapi::dpl::normal_distribution<sycl::vec<float, 3>>>();
    err += test_vec<oneapi::dpl::normal_distribution<sycl::vec<float, 2>>>();
    err += test_vec<oneapi::dpl::normal_distribution<sycl::vec<float, 1>>>();
#    endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    std::cout << "---------------------------------------------------" << std::endl;
    std::cout << "normal_distribution<double>" << std::endl;
    std::cout << "---------------------------------------------------" << std::endl;
    err += test<oneapi::dpl::normal_distribution<double>>();
#    if TEST_LONG_RUN
    err += test_vec<oneapi::dpl::normal_distribution<sycl::vec<double, 16>>>();
    err += test_vec<oneapi::dpl::normal_distribution<sycl::vec<double, 8>>>();
    err += test_vec<oneapi::dpl::normal_distribution<sycl::vec<double, 4>>>();
    err += test_vec<oneapi::dpl::normal_distribution<sycl::vec<double, 3>>>();
    err += test_vec<oneapi::dpl::normal_distribution<sycl::vec<double, 2>>>();
    err += test_vec<oneapi::dpl::normal_distribution<sycl::vec<double, 1>>>();
#    endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    std::cout << "---------------------------------------------------" << std::endl;
    std::cout << "exponential_distribution<float>" << std::endl;
    std::cout << "---------------------------------------------------" << std::endl;
    err += test<oneapi::dpl::exponential_distribution<float>>();
#    if TEST_LONG_RUN
    err += test_vec<oneapi::dpl::exponential_distribution<sycl::vec<float, 16>>>();
    err += test_vec<oneapi::dpl::exponential_distribution<sycl::vec<float, 8>>>();
    err += test_vec<oneapi::dpl::exponential_distribution<sycl::vec<float, 4>>>();
    err += test_vec<oneapi::dpl::exponential_distribution<sycl::vec<float, 3>>>();
    err += test_vec<oneapi::dpl::exponential_distribution<sycl::vec<float, 2>>>();
    err += test_vec<oneapi::dpl::exponential_distribution<sycl::vec<float, 1>>>();
#    endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    std::cout << "---------------------------------------------------" << std::endl;
    std::cout << "exponential_distribution<double>" << std::endl;
    std::cout << "---------------------------------------------------" << std::endl;
    err += test<oneapi::dpl::exponential_distribution<double>>();
#    if TEST_LONG_RUN
    err += test_vec<oneapi::dpl::exponential_distribution<sycl::vec<double, 16>>>();
    err += test_vec<oneapi::dpl::exponential_distribution<sycl::vec<double, 8>>>();
    err += test_vec<oneapi::dpl::exponential_distribution<sycl::vec<double, 4>>>();
    err += test_vec<oneapi::dpl::exponential_distribution<sycl::vec<double, 3>>>();
    err += test_vec<oneapi::dpl::exponential_distribution<sycl::vec<double, 2>>>();
    err += test_vec<oneapi::dpl::exponential_distribution<sycl::vec<double, 1>>>();
#    endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

#endif // TEST_DPCPP_BACKEND_PRESENT && TEST_UNNAMED_LAMBDAS

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT && TEST_UNNAMED_LAMBDAS);
}
