// -*- C++ -*-
//===-- engines_methods.hpp ----------------------             -------------===//
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
// Testing of different engines' methods

#include "support/utils.h"
#include <iostream>

#if _ONEDPL_BACKEND_SYCL
#    include <vector>
#    include <CL/sycl.hpp>
#    include <oneapi/dpl/random>

#    define SEED 777
#    define N_GEN 960
#    define MINSTD_A 48271
#    define MINSTD_C 0
#    define MINSTD_M 2147483647
#    define MINSTD_MIN 1
#    define MINSTD_MAX 2147483646
#    define RANLUX24_BASE_W 24
#    define RANLUX24_BASE_S 10
#    define RANLUX24_BASE_R 24
#    define RANLUX24_BASE_MIN 0
#    define RANLUX24_BASE_MAX 16777215
#    define RANLUX24_P 223
#    define RANLUX24_R 23

template <class Engine, int N>
std::int32_t
check_params(Engine& engine)
{
    if constexpr (std::is_same<Engine, oneapi::dpl::minstd_rand>::value)
    {
        return ((oneapi::dpl::minstd_rand::multiplier != MINSTD_A) ||
                (oneapi::dpl::minstd_rand::increment != MINSTD_C) || (oneapi::dpl::minstd_rand::modulus != MINSTD_M) ||
                (engine.min() != MINSTD_MIN) || (engine.max() != MINSTD_MAX));
    }
    if constexpr (std::is_same<Engine, oneapi::dpl::minstd_rand_vec<N>>::value)
    {
        return ((oneapi::dpl::minstd_rand_vec<N>::multiplier != MINSTD_A) ||
                (oneapi::dpl::minstd_rand_vec<N>::increment != MINSTD_C) ||
                (oneapi::dpl::minstd_rand_vec<N>::modulus != MINSTD_M) || (engine.min() != MINSTD_MIN) ||
                (engine.max() != MINSTD_MAX));
    }
    if constexpr (std::is_same<Engine, oneapi::dpl::ranlux24_base>::value)
    {
        return ((oneapi::dpl::ranlux24_base::word_size != RANLUX24_BASE_W) ||
                (oneapi::dpl::ranlux24_base::short_lag != RANLUX24_BASE_S) ||
                (oneapi::dpl::ranlux24_base::long_lag != RANLUX24_BASE_R) || (engine.min() != RANLUX24_BASE_MIN) ||
                (engine.max() != RANLUX24_BASE_MAX));
    }
    if constexpr (std::is_same<Engine, oneapi::dpl::ranlux24_base_vec<N>>::value)
    {
        return ((oneapi::dpl::ranlux24_base_vec<N>::word_size != RANLUX24_BASE_W) ||
                (oneapi::dpl::ranlux24_base_vec<N>::short_lag != RANLUX24_BASE_S) ||
                (oneapi::dpl::ranlux24_base_vec<N>::long_lag != RANLUX24_BASE_R) ||
                (engine.min() != RANLUX24_BASE_MIN) || (engine.max() != RANLUX24_BASE_MAX));
    }
    if constexpr (std::is_same<Engine, oneapi::dpl::ranlux24>::value)
    {
        return ((oneapi::dpl::ranlux24::block_size != RANLUX24_P) ||
                (oneapi::dpl::ranlux24::used_block != RANLUX24_R) || (engine.min() != RANLUX24_BASE_MIN) ||
                (engine.max() != RANLUX24_BASE_MAX));
    }
    return 0;
}

template <class Engine>
bool
test()
{
    using result_type = typename Engine::scalar_type;

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

    sycl::queue queue(sycl::default_selector{}, exception_handler);
    int sum = 0;

    // Memory allocation
    std::vector<std::int32_t> dpstd_res(N_GEN);
    constexpr std::int32_t num_elems =
        oneapi::dpl::internal::type_traits_t<typename Engine::result_type>::num_elems == 0
            ? 1
            : oneapi::dpl::internal::type_traits_t<typename Engine::result_type>::num_elems;

    // Random number generation
    {
        sycl::buffer<std::int32_t, 1> dpstd_buffer(dpstd_res.data(), dpstd_res.size());

        try
        {
            queue.submit([&](sycl::handler& cgh) {
                auto dpstd_acc = dpstd_buffer.template get_access<sycl::access::mode::write>(cgh);

                cgh.parallel_for<>(sycl::range<1>(N_GEN), [=](sycl::item<1> idx) {
                    unsigned long long offset = idx.get_linear_id() * num_elems;
                    Engine engine0(SEED);
                    Engine engine1;
                    engine1.seed(SEED);
                    engine0.discard(offset);
                    engine1.discard(offset);
                    typename Engine::result_type res0;
                    if constexpr (std::is_same<Engine, oneapi::dpl::ranlux24>::value)
                    {
                        Engine engine(engine1);
                        auto eng = engine.base();
                        res0 = engine();
                    }
                    else
                    {
                        res0 = engine0();
                    }
                    typename Engine::result_type res1 = engine1();
                    if constexpr ((num_elems > 1) || (std::is_same<typename Engine::result_type,
                                                                   sycl::vec<typename Engine::scalar_type, 1>>::value))
                    {
                        std::int32_t is_inequal = 0;
                        for (std::int32_t i = 0; i < num_elems; ++i)
                        {
                            if (res0[i] != res1[i])
                            {
                                is_inequal = 1;
                            }
                        }
                        dpstd_acc[offset] = is_inequal;
                    }
                    else
                    {
                        if (res0 != res1)
                        {
                            dpstd_acc[offset] = 1;
                        }
                        else
                        {
                            dpstd_acc[offset] = 0;
                        }
                    }
                });
            });
            auto dpstd_acc = dpstd_buffer.template get_access<sycl::access::mode::write>();
            for (int i = 0; i < N_GEN; ++i)
            {
                sum += dpstd_acc[i];
            }
            if (sum)
            {
                std::cout << "Error occured in " << sum << " elements" << std::endl;
            }
        }
        catch (sycl::exception const& e)
        {
            std::cout << "\t\tSYCL exception during generation\n"
                      << e.what() << std::endl
                      << "OpenCL status: " << e.get_cl_code() << std::endl;
            return 1;
        }

        queue.wait_and_throw();
        Engine engine;
        sum += check_params<Engine, num_elems>(engine);
    }

    return sum;
}

#endif // _ONEDPL_BACKEND_SYCL

int
main()
{

#if _ONEDPL_BACKEND_SYCL

    std::int32_t err = 0;
    std::int32_t global_err = 0;

    std::cout << "---------------------------------------------------" << std::endl;
    std::cout << "linear_congruential_engine<48271, 0, 2147483647>" << std::endl;
    std::cout << "---------------------------------------------------" << std::endl;
    err += test<oneapi::dpl::minstd_rand>();
    err += test<oneapi::dpl::minstd_rand_vec<1>>();
    err += test<oneapi::dpl::minstd_rand_vec<2>>();
    err += test<oneapi::dpl::minstd_rand_vec<3>>();
    err += test<oneapi::dpl::minstd_rand_vec<4>>();
    err += test<oneapi::dpl::minstd_rand_vec<8>>();
    err += test<oneapi::dpl::minstd_rand_vec<16>>();
    if (err)
    {
        std::cout << "Test FAILED" << std::endl;
    }
    global_err += err;
    err = 0;

    std::cout << "---------------------------------------------------" << std::endl;
    std::cout << "subtract_with_carry_engine<24, 10, 24>" << std::endl;
    std::cout << "---------------------------------------------------" << std::endl;
    err += test<oneapi::dpl::ranlux24_base>();
    err += test<oneapi::dpl::ranlux24_base_vec<1>>();
    err += test<oneapi::dpl::ranlux24_base_vec<2>>();
    err += test<oneapi::dpl::ranlux24_base_vec<3>>();
    err += test<oneapi::dpl::ranlux24_base_vec<4>>();
    err += test<oneapi::dpl::ranlux24_base_vec<8>>();
    err += test<oneapi::dpl::ranlux24_base_vec<16>>();
    if (err)
    {
        std::cout << "Test FAILED" << std::endl;
    }
    global_err += err;
    err = 0;

    std::cout << "---------------------------------------------------" << std::endl;
    std::cout << "discard_block_engine<ranlux24_base, 223, 23>" << std::endl;
    std::cout << "---------------------------------------------------" << std::endl;
    err += test<oneapi::dpl::ranlux24>();
    err += test<oneapi::dpl::ranlux24_vec<1>>();
    err += test<oneapi::dpl::ranlux24_vec<2>>();
    err += test<oneapi::dpl::ranlux24_vec<3>>();
    err += test<oneapi::dpl::ranlux24_vec<4>>();
    err += test<oneapi::dpl::ranlux24_vec<8>>();
    err += test<oneapi::dpl::ranlux24_vec<16>>();
    if (err)
    {
        std::cout << "Test FAILED" << std::endl;
    }
    global_err += err;

    if (global_err)
    {
        return 1;
    }

#endif // _ONEDPL_BACKEND_SYCL

    return TestUtils::done(_ONEDPL_BACKEND_SYCL);
}