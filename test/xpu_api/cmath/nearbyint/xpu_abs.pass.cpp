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

#include <oneapi/dpl/cmath>

#include "support/test_config.h"
#include "support/utils.h"
#include "support/utils_invoke.h"

#include <iostream>
#include <vector>

#if TEST_DPCPP_BACKEND_PRESENT
#include <CL/sycl.hpp>

constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;

template <typename KernelClass, typename Function, typename ValueType>
void
test(Function fnc, const std::vector<ValueType>& args, const char* message)
{
    const auto args_count = args.size();

    std::vector<ValueType> output(args_count);

    cl::sycl::range<1> numOfItems1{args_count};
    cl::sycl::range<1> numOfItems2{args_count};

    // Evaluate results in Kernel
    {
        cl::sycl::queue deviceQueue(TestUtils::async_handler);

        if (!TestUtils::has_type_support<ValueType>(deviceQueue.get_device()))
        {
            std::cout << deviceQueue.get_device().template get_info<sycl::info::device::name>()
                      << " does not support " << typeid(ValueType).name() << " type,"
                      << " affected test case have been skipped" << std::endl;
            return;
        }

        cl::sycl::buffer<ValueType> buffer1(args.data(), args_count);
        cl::sycl::buffer<ValueType> buffer2(output.data(), args_count);

        deviceQueue.submit(
            [&](cl::sycl::handler& cgh)
            {
                auto in = buffer1.template get_access<sycl_read>(cgh);
                auto out = buffer2.template get_access<sycl_write>(cgh);
                cgh.single_task<KernelClass>(
                    [=]()
                    {
                        for (size_t i = 0; i < args_count; ++i)
                            out[i] = fnc(in[i]);
                    });
            });
    }

    // Check results: compare resuls evaluated in Kernel and on host
    for (size_t i = 0; i < args_count; ++i)
    {
        auto host_result = fnc(args[i]);
        EXPECT_EQ(host_result, output[i], message);
    }
}

class Test;

#endif // TEST_DPCPP_BACKEND_PRESENT

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT

    // functions from https://en.cppreference.com/w/cpp/numeric/math/abs :
    //     int       abs( int n );
    //     long      abs( long n );
    //     long long abs( long long n );

    ////////////////////////////////////////////////////////
    // int       abs( int n );
    auto f_abs_int = [](int arg) -> int { return oneapi::dpl::abs(arg); };
    const std::vector<int> f_args_int = { -2, 0, 2};
    test<TestUtils::unique_kernel_name<Test, 1>>(f_abs_int, f_args_int, "int abs(int)");

    ////////////////////////////////////////////////////////
    // long       abs( long n );
    auto f_abs_long = [](long arg) -> long { return oneapi::dpl::abs(arg); };
    const std::vector<long> f_args_long = {-2, 0, 2};
    test<TestUtils::unique_kernel_name<Test, 2>>(f_abs_int, f_args_int, "long abs(long)");

    ////////////////////////////////////////////////////////
    // long long abs( long long n );
    auto f_abs_long_long = [](long long arg) -> long long { return oneapi::dpl::abs(arg); };
    const std::vector<long long> f_args_long_long = {-2, 0, 2};
    test<TestUtils::unique_kernel_name<Test, 3>>(f_abs_long_long, f_args_long_long, "long long abs(long long)");

#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
