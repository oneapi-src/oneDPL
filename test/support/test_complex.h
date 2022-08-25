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
#include "test_config.h"

#include <type_traits>
#include <cassert>

#include "test_macros.h"

#define ONEDPL_TEST_NUM_MAIN                                                                          \
template <typename HasDoubleSupportInRuntime, typename HasLongDoubleSupportInCompiletime>             \
int                                                                                                   \
run_test();                                                                                           \
                                                                                                      \
int main(int, char**)                                                                                 \
{                                                                                                     \
    run_test<::std::true_type, ::std::true_type>();                                                   \
                                                                                                      \
    /* Sometimes we may start test on device, which don't support type double. */                     \
    /* In this case generates run-time error.                                  */                     \
    /* This two types allow us to avoid this situation.                        */                     \
    using HasDoubleTypeSupportInRuntime = ::std::true_type;                                           \
    using HasntDoubleTypeSupportInRuntime = ::std::false_type;                                        \
                                                                                                      \
    /* long double type generate compile-time error in Kernel code             */                     \
    /* and we never can use this type inside Kernel                            */                     \
    using HasntLongDoubleSupportInCompiletime = ::std::false_type;                                    \
                                                                                                      \
    TestUtils::run_test_in_kernel(                                                                    \
        /* labbda for the case when we have support of double type on device */                       \
        [&]() { run_test<HasDoubleTypeSupportInRuntime, HasntLongDoubleSupportInCompiletime>(); },    \
        /* labbda for the case when we haven't support of double type on device */                    \
        [&]() { run_test<HasntDoubleTypeSupportInRuntime, HasntLongDoubleSupportInCompiletime>(); }); \
                                                                                                      \
    return TestUtils::done();                                                                         \
}                                                                                                     \
                                                                                                      \
template <typename HasDoubleSupportInRuntime, typename HasLongDoubleSupportInCompiletime>             \
int                                                                                                   \
run_test()

// We should use this macros to avoid runtime-error if type double doesn't supported on device.
// 
// Example:
//     template <class T>
//     void
//     test(T x, typename std::enable_if<std::is_integral<T>::value>::type* = 0)
//     {
//         static_assert((std::is_same<decltype(dpl::conj(x)), dpl::complex<double>>::value), "");
// 
//         // HERE IS THE CODE WHICH CALL WE SHOULD AVOID IF DOUBLE IS NOT SUPPORTED ON DEVICE
//         assert(dpl::conj(x) == dpl::conj(dpl::complex<double>(x, 0)));
//     }
//
//     template <class T>
//     void test()
//     {
//         // ...
//         test<T>(1);
//         // ...
//     }
//
//     ONEDPL_TEST_NUM_MAIN
//     {
//         // ...
//         IF_DOUBLE_SUPPORT_IN_RUNTIME(test<int>())
//         // ...
//     }
#define IF_DOUBLE_SUPPORT_IN_RUNTIME(x)                                                               \
    if constexpr (HasDoubleSupportInRuntime::value) { x; }

// We should use this macros to avoid compile-time error in code with long double type in Kernel.
#define IF_LONG_DOUBLE_SUPPORT_IN_COMPILETIME(x)                                                      \
    if constexpr (HasLongDoubleSupportInCompiletime::value) { x; }

namespace TestUtils
{
    // Run test in Kernel as single task
    template <typename TFncDoubleHasSupportInRuntime, typename TFncDoubleHasntSupportInRuntime>
    void
    run_test_in_kernel(TFncDoubleHasSupportInRuntime fncDoubleHasSupportInRuntime,
                       TFncDoubleHasntSupportInRuntime fncDoubleHasntSupportInRuntime)
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
                                fncDoubleHasSupportInRuntime();
                            else
                                fncDoubleHasntSupportInRuntime();
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
