// -*- C++ -*-
//===----------------------------------------------------------------------===//
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

// oneDPL headers should be included before standard headers
#include <oneapi/dpl/random>

#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>

template <typename RealType>
void
scalar_example(sycl::queue& queue, std::uint32_t seed, std::vector<RealType>& x)
{
    {
        sycl::buffer<RealType> x_buffer(x.data(), x.size());

        queue.submit(
            [&](sycl::handler& cgh)
            {
                sycl::accessor x_acc(x_buffer, cgh, sycl::write_only);

                cgh.parallel_for(sycl::range<1>(x.size()),
                                 [=](sycl::item<1> idx)
                                 {
                                     unsigned long long offset = idx.get_linear_id();

                                     // Create minstd_rand engine
                                     oneapi::dpl::minstd_rand engine(seed, offset);

                                     // Create float uniform_real_distribution distribution with a = 0, b = 1
                                     oneapi::dpl::uniform_real_distribution<float> distr;

                                     // Generate float random number
                                     auto res = distr(engine);

                                     // Store results to x_acc
                                     x_acc[idx] = res;
                                 });
            });
    }

    std::cout << "\nsuccess for scalar generation" << std::endl;
    std::cout << "First 5 samples of minstd_rand with scalar generation" << std::endl;
    for (int i = 0; i < 5; i++)
    {
        std::cout << x.begin()[i] << std::endl;
    }

    std::cout << "\nLast 5 samples of minstd_rand with scalar generation" << std::endl;
    for (int i = 0; i < 5; i++)
    {
        std::cout << x.rbegin()[i] << std::endl;
    }
}

template <int VecSize, typename RealType>
void
vector_example(sycl::queue& queue, std::uint32_t seed, std::vector<RealType>& x)
{
    {
        sycl::buffer<RealType> x_buffer(x.data(), x.size());

        queue.submit(
            [&](sycl::handler& cgh)
            {
                sycl::accessor x_acc(x_buffer, cgh, sycl::write_only);

                cgh.parallel_for(
                    sycl::range<1>(x.size() / VecSize),
                    [=](sycl::item<1> idx)
                    {
                        unsigned long long offset = idx.get_linear_id() * VecSize;

                        // Create minstd_rand engine
                        oneapi::dpl::minstd_rand_vec<VecSize> engine(seed);
                        engine.discard(offset);

                        // Create float uniform_real_distribution distribution
                        oneapi::dpl::uniform_real_distribution<sycl::vec<float, VecSize>> distr;

                        // Generate sycl::vec<float, VecSize> of random numbers
                        auto res = distr(engine);

                        // Store results from res to VecSize * offset position of x_acc
                        res.store(idx.get_linear_id(), x_acc.get_pointer());
                    });
            });
    }

    std::cout << "\nsuccess for vector generation" << std::endl;
    std::cout << "First 5 samples of minstd_rand with vector generation" << std::endl;
    for (int i = 0; i < 5; i++)
    {
        std::cout << x.begin()[i] << std::endl;
    }

    std::cout << "\nLast 5 samples of minstd_rand with vector generation" << std::endl;
    for (int i = 0; i < 5; i++)
    {
        std::cout << x.rbegin()[i] << std::endl;
    }
}

int
main()
{
    auto async_handler = [](sycl::exception_list ex_list)
    {
        for (auto& ex : ex_list)
        {
            try
            {
                std::rethrow_exception(ex);
            }
            catch (sycl::exception& ex)
            {
                std::cerr << ex.what() << std::endl;
                std::exit(1);
            }
        }
    };

    sycl::queue queue(sycl::default_selector_v, async_handler);

    constexpr std::int64_t nsamples = 100;
    constexpr int vec_size = 4;
    constexpr std::uint32_t seed = 777;

    std::vector<float> x(nsamples);

    // Scalar random number generation
    scalar_example(queue, seed, x);

    // Vector random number generation
    vector_example<vec_size>(queue, seed, x);

    return 0;
}
