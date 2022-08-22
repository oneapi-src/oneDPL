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

#ifndef _TEST_COMPLEX_H
#define _TEST_COMPLEX_H

#include <oneapi/dpl/complex>
#include <oneapi/dpl/pstl/utils.h>

#include "utils.h"
#include "utils_invoke.h"

#include <type_traits>
#include <cassert>

#include "test_macros.h"

#define ONEDPL_TEST_NUM_MAIN                                                                    \
template <typename EnableDouble, typename EnableLongDouble>                                     \
int                                                                                             \
run_test();                                                                                     \
                                                                                                \
int main(int, char**)                                                                           \
{                                                                                               \
    run_test<::std::true_type, ::std::true_type>();                                             \
                                                                                                \
    TestUtils::run_test_in_kernel([&]() { run_test<::std::true_type, ::std::false_type>(); },   \
                                  [&]() { run_test<::std::false_type, ::std::false_type>(); }); \
                                                                                                \
    return TestUtils::done();                                                                   \
}                                                                                               \
                                                                                                \
template <typename EnableDouble, typename EnableLongDouble>                                     \
int                                                                                             \
run_test()

#define INVOKE_IF_DOUBLE_SUPPORT(x)                                                             \
    if constexpr (EnableDouble::value) { x; }

#define INVOKE_IF_LONG_DOUBLE_SUPPORT(x)                                                        \
    if constexpr (EnableLongDouble::value) { x; }


namespace TestUtils
{
    /**
     * Run test in Kernel as single task.
     * 
     * If an exception occurred, it logged into std::cerr and application finish with EXIT_FAILURE return code.
     * 
     * @param TFncTest1 fncDoubleSupported - lambda for call if double is supported on device
     * @param TFncTest1 fncDoubleNotSupported - lambda for call if double is not supported on device
     */
    template <typename TFncTest1, typename TFncTest2>
    void
    run_test_in_kernel(TFncTest1 fncDoubleSupported, TFncTest2 fncDoubleNotSupported)
    {
#if TEST_DPCPP_BACKEND_PRESENT
        try
        {
            sycl::queue deviceQueue{TestUtils::default_selector};

            const auto device = deviceQueue.get_device();
            const bool double_supported = has_type_support<double>(device);

            deviceQueue.submit(
                [&](cl::sycl::handler& cgh)
                {
                    cgh.single_task<TestUtils::new_kernel_name<class TestType, 0>>(
                        [=]()
                        { 
                            if (double_supported)
                                fncDoubleSupported();
                            else
                                fncDoubleNotSupported();
                        });
                });
        }
        catch (const std::exception& exc)
        {
            std::stringstream str;

            str << "Exception occurred";
            if (exc.what())
                str << " : " << exc.what();

            TestUtils::issue_error_message(str);
        }
#endif // TEST_DPCPP_BACKEND_PRESENT
    }

} /* namespace TestUtils */

#endif /* _TEST_COMPLEX_H */
