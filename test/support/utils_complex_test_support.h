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

#include <oneapi/dpl/pstl/utils.h>

// C++ compiler versions:
//      C++11 : __cplusplus is 201103L
//      C++14 : __cplusplus is 201402L
//      C++20 : __cplusplus is 202002L

namespace TestUtils
{
namespace Complex
{
    template <typename TComplexDataType>
    struct TestConstants;

    template <>
    struct TestConstants<float>
    {
        using DestComplexFieldType = float;

        static constexpr float kPartReal = 1.5f;
        static constexpr float kPartImag = 2.25f;
        static constexpr float kZero     = 0.0f;

        static constexpr float kExpectedResReal = kPartReal;
    };

    template <>
    struct TestConstants<double>
    {
        using DestComplexFieldType = double;

        static constexpr double kPartReal = 1.5;
        static constexpr double kPartImag = 2.25;
        static constexpr double kZero     = 0.0;

        static constexpr double kExpectedResReal = kPartReal;
    };

    template <>
    struct TestConstants<long double>
    {
        using DestComplexFieldType = long double;

        static constexpr long double kPartReal = 1.5L;
        static constexpr long double kPartImag = 2.25L;
        static constexpr long double kZero     = 0.0L;

        static constexpr long double kExpectedResReal = kPartReal;
    };

    template <>
    struct TestConstants<int>
    {
        using DestComplexFieldType = double;

        static constexpr int kPartReal = 1;
        static constexpr int kPartImag = 1;
        static constexpr int kZero     = 0;

        static constexpr double kExpectedResReal = 1.0;
    };

    // Run test TComplexTestName on host
    /*
     * @return bool - true if no errors occurred, false - otherwise.
     */
    template <template <typename TErrorsContainer, typename IsSupportedDouble, typename IsSupportedLongDouble> class TComplexTestName>
    bool test_on_host()
    {
        // Prepare host error container
        TestUtils::ErrorsContainerOnHost errors;

        // Run test on host
        TComplexTestName<TestUtils::ErrorsContainerOnHost, ::std::true_type, ::std::true_type> tcc(errors);
        tcc.run_test(::std::true_type{});

        return !errors.bHaveErrors;
    }

#if TEST_DPCPP_BACKEND_PRESENT
    // Run TComplexTestName test in Kernel
    /*
     * @return bool - true if no errors occurred, false - otherwise.
     */
    template <template <typename TErrorsContainer, typename IsSupportedDouble, typename IsSupportedLongDouble> class TComplexTestName>
    bool test_in_kernel(sycl::queue& deviceQueue, ::std::size_t max_errors_count)
    {
        TestUtils::ErrorContainer_HostPart error_container_host_part(max_errors_count);

        const auto& device = deviceQueue.get_device();

        {
            auto sycl_buf_host_errors = TestUtils::ErrorContainer_HostPart::get_sycl_buffer(error_container_host_part);

            if (device.has(sycl::aspect::fp64))
            {
                deviceQueue.submit(
                    [&](cl::sycl::handler& cgh)
                    {
                        auto accessor_to_sycl_buf_host_errors = sycl_buf_host_errors.template get_access<cl::sycl::access::mode::read_write>(cgh);
                        using ErrorContainer_KernelPart_Impl = ::TestUtils::ErrorContairer_KernelPart<decltype(accessor_to_sycl_buf_host_errors)>;
                        using TestType = TComplexTestName<ErrorContainer_KernelPart_Impl, ::std::true_type, ::std::false_type>;

                        cgh.single_task<new_kernel_name<TestType, 0>>(
                            [=]()
                            {
                                // Prepare kernel part of error container
                                ErrorContainer_KernelPart_Impl error_container_kernel_part(accessor_to_sycl_buf_host_errors, max_errors_count);

                                // Run test in kernel
                                TestType tcc(error_container_kernel_part);
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
                        using ErrorContainer_KernelPart_Impl = ::TestUtils::ErrorContairer_KernelPart<decltype(accessor_to_sycl_buf_host_errors)>;
                        using TestType = TComplexTestName<ErrorContainer_KernelPart_Impl, ::std::false_type, ::std::false_type>;

                        cgh.single_task<new_kernel_name<TestType, 1>>(
                            [=]()
                            {
                                // Prepare kernel part of error container
                                ErrorContainer_KernelPart_Impl error_container_kernel_part(accessor_to_sycl_buf_host_errors, max_errors_count);

                                // Run test in kernel
                                TestType tcc(error_container_kernel_part);
                                tcc.run_test(::std::false_type{});
                            });
                    });
            }
        }

        error_container_host_part.process_errors();

        return !error_container_host_part.have_errors();
    }
#endif // TEST_DPCPP_BACKEND_PRESENT

} /* namespace Complex */

} /* namespace TestUtils */

#endif // __UTILS_COMPLEX_TEST_SUPPORT_H
