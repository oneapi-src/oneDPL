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
#include <memory>

#if TEST_DPCPP_BACKEND_PRESENT
#include <CL/sycl.hpp>

constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;

template <typename KernelClass, typename Function, typename ValueType>
struct TestImpl
{
    void operator()(cl::sycl::queue deviceQueue, Function fnc, const std::vector<ValueType>& args, const char* message)
    {
        const auto args_count = args.size();

        std::vector<ValueType> output(args_count);

        cl::sycl::range<1> numOfItems1{args_count};
        cl::sycl::range<1> numOfItems2{args_count};

        // Evaluate results in Kernel
        {
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
};

template <typename KernelClass, typename Function, typename ValueType>
bool
test(cl::sycl::queue deviceQueue, Function fnc, const std::vector<ValueType>& args, const char* message)
{
    if (TestUtils::has_type_support<ValueType>(deviceQueue.get_device()))
    {
        auto test_obj = ::std::make_unique<TestImpl<KernelClass, Function, ValueType> >();
        (*test_obj)(deviceQueue, fnc, args, message);
        return true;
    }
    else
    {
        std::cout << deviceQueue.get_device().template get_info<sycl::info::device::name>() << " does not support "
                  << typeid(ValueType).name() << " type,"
                  << " affected test case have been skipped" << std::endl;
        return false;
    }
}

class Test;

#endif // TEST_DPCPP_BACKEND_PRESENT

int
main()
{
    int result = 0;

#if TEST_DPCPP_BACKEND_PRESENT

    // functions from https://en.cppreference.com/w/cpp/numeric/math/nearbyint :
    //     long double nearbyintl(long double arg);

    cl::sycl::queue deviceQueue(TestUtils::async_handler);

    ////////////////////////////////////////////////////////
    // long double nearbyintl(long double arg);
    const std::vector<long double> f_args_ld = {+2.3, +2.5, +3.5, -2.3, -2.5, -3.5};
    auto f_nearbyintl_ld = [](long double arg) -> long double { return oneapi::dpl::nearbyintl(arg); };
    if (test<TestUtils::unique_kernel_name<Test, 11>>(deviceQueue, f_nearbyintl_ld, f_args_ld, "long double nearbyintl(long double)"))
        result = TEST_DPCPP_BACKEND_PRESENT;

#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(result);
}
