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

#ifndef __UTILS_COMPLEX_TEST_SUPPORT_H
#define __UTILS_COMPLEX_TEST_SUPPORT_H

#include <oneapi/dpl/execution>
#include <oneapi/dpl/complex>

#include "test_config.h"
#include "utils.h"
#include "utils_err_eng.h"

#if TEST_DPCPP_BACKEND_PRESENT
#    include <CL/sycl.hpp>
#endif

//
// https://stackoverflow.com/questions/2324658/how-to-determine-the-version-of-the-c-standard-used-by-the-compiler
//      C++11 : __cplusplus is 201103L
//      C++14 : __cplusplus is 201402L
//      C++20 : __cplusplus is 202002L

namespace TestUtils
{
namespace Complex
{
    template <typename TComplexDataType>
    struct InitConst;

    template <>
    struct InitConst<float>
    {
        using DestComplexFieldType = float;

        static constexpr float kPartReal = 1.5f;
        static constexpr float kPartImag = 2.25f;
        static constexpr float kZero     = 0.0f;

        static constexpr float kExpectedResReal = kPartReal;
    };

    template <>
    struct InitConst<double>
    {
        using DestComplexFieldType = double;

        static constexpr double kPartReal = 1.5;
        static constexpr double kPartImag = 2.25;
        static constexpr double kZero     = 0.0;

        static constexpr double kExpectedResReal = kPartReal;
    };

    template <>
    struct InitConst<long double>
    {
        using DestComplexFieldType = long double;

        static constexpr long double kPartReal = 1.5L;
        static constexpr long double kPartImag = 2.25L;
        static constexpr long double kZero     = 0.0L;

        static constexpr long double kExpectedResReal = kPartReal;
    };

    template <>
    struct InitConst<int>
    {
        using DestComplexFieldType = double;

        static constexpr int kPartReal = 1;
        static constexpr int kPartImag = 1;
        static constexpr int kZero     = 0;

        static constexpr double kExpectedResReal = 1.0;
    };

    template <typename TRequiredType, typename TVal>
    void check_type(TVal val)
    {
        static_assert(::std::is_same<typename ::std::decay<TVal>::type, TRequiredType>::value, "Types should be equals");
    }

    // Run test TComplexTestName on host
    template <template <typename TErrorEngine, typename IsSupportedDouble, typename IsSupportedLongDouble> class TComplexTestName>
    void test_on_host()
    {
        // Prepare host error engine
        TestUtils::ErrorEngineHost error_engine_host;

        // Run test on host
        TComplexTestName<TestUtils::ErrorEngineHost, ::std::true_type, ::std::true_type> tcc(error_engine_host);
        tcc.run_test(::std::true_type{});
    }

#if TEST_DPCPP_BACKEND_PRESENT
    // Run TComplexTestName test in Kernel
    template <template <typename TErrorEngine, typename IsSupportedDouble, typename IsSupportedLongDouble> class TComplexTestName>
    void test_in_kernel(sycl::queue& deviceQueue)
    {
        TestUtils::ErrorEngine_HostPart error_engine_host_part;

        const auto& device = deviceQueue.get_device();

        {
            auto sycl_buf_host_errors = error_engine_host_part.get_sycl_buffer();

            if (device.has(sycl::aspect::fp64))
            {
                deviceQueue.submit(
                    [&](cl::sycl::handler& cgh)
                    {
                        auto accessor_to_sycl_buf_host_errors = sycl_buf_host_errors.template get_access<cl::sycl::access::mode::read_write>(cgh);
                        using ErrorEngine_KernelPart_Impl = ::TestUtils::ErrorEngine_KernelPart<decltype(accessor_to_sycl_buf_host_errors)>;
                        using TestType = TComplexTestName<ErrorEngine_KernelPart_Impl, ::std::true_type, ::std::false_type>;

                        cgh.single_task<new_kernel_name<TestType, 0>>(
                            [=]()
                            {
                                // Prepare kernel part of error engine
                                ErrorEngine_KernelPart_Impl error_engine_kernel_part(accessor_to_sycl_buf_host_errors);

                                // Run test in kernel
                                TestType tcc(error_engine_kernel_part);
                                tcc.run_test(::std::false_type{});
                            });
                    });
            }
            else
            {
                deviceQueue.submit(
                    [&](cl::sycl::handler& cgh)
                    {
                        auto accessor_to_sycl_buf_host_errors = sycl_buf_host_errors.template get_access<cl::sycl::access::mode::read_write>(cgh);
                        using ErrorEngine_KernelPart_Impl = ::TestUtils::ErrorEngine_KernelPart<decltype(accessor_to_sycl_buf_host_errors)>;
                        using TestType = TComplexTestName<ErrorEngine_KernelPart_Impl, ::std::false_type, ::std::false_type>;

                        cgh.single_task<new_kernel_name<TestType, 1>>(
                            [=]()
                            {
                                // Prepare kernel part of error engine
                                ErrorEngine_KernelPart_Impl error_engine_kernel_part(accessor_to_sycl_buf_host_errors);

                                // Run test in kernel
                                TestType tcc(error_engine_kernel_part);
                                tcc.run_test(::std::false_type{});
                            });
                    });
            }
        }

        error_engine_host_part.process_errors();
    }
#endif // TEST_DPCPP_BACKEND_PRESENT

} /* namespace Complex */

} /* namespace TestUtils */

#endif // __UTILS_COMPLEX_TEST_SUPPORT_H
