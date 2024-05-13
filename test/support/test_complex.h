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

#include "test_config.h"

#include <oneapi/dpl/complex>

#include "utils.h"
#include "utils_invoke.h"

#include <type_traits>
#include <cassert>
#include <vector>

#if !_PSTL_MSVC_LESS_THAN_CPP20_COMPLEX_CONSTEXPR_BROKEN
#    define STD_COMPLEX_TESTS_STATIC_ASSERT(arg) static_assert(arg)
#else
#    define STD_COMPLEX_TESTS_STATIC_ASSERT(arg) assert(arg)
#endif // !_PSTL_MSVC_LESS_THAN_CPP20_COMPLEX_CONSTEXPR_BROKEN

struct HostChecker
{
    struct ErrorInfo
    {
        int i = 0;
        int j = 0;
        int lineNo = 0;

        bool
        operator==(const ErrorInfo& other) const
        {
            return i == other.i && j == other.j && lineNo == other.lineNo;
        }

        bool
        operator<(const ErrorInfo& other) const
        {
            if (lineNo != other.lineNo)
                return lineNo < other.lineNo;

            if (i != other.i)
                return i < other.i;

            return j < other.j;
        }
    };
    std::vector<ErrorInfo> errors;

    inline void
    operator()(int i, bool bResult, int lineNo)
    {
        if (!bResult)
        {
            ErrorInfo info;
            info.i = i;
            info.lineNo = lineNo;
            errors.push_back(info);
        }
    }

    inline void
    operator()(int i, int j, bool bResult, int lineNo)
    {
        if (!bResult)
        {
            ErrorInfo info;
            info.i = i;
            info.lineNo = lineNo;
            errors.push_back(info);
        }
    }
};

struct KernelChecker
{
    inline void
    operator()(int /*i*/, bool bResult, int /*lineNo*/) const
    {
        assert(bResult);
    }

    inline void
    operator()(int /*i*/, int /*j*/, bool bResult, int /*lineNo*/) const
    {
        assert(bResult);
    }
};

#define ONEDPL_TEST_NUM_MAIN                                                                                            \
template <typename HasDoubleSupportInRuntime, typename HasLongDoubleSupportInCompiletime,                               \
          class TChecker>                                                                                               \
int                                                                                                                     \
run_test(TChecker& check_obj);                                                                                          \
                                                                                                                        \
int main(int, char**)                                                                                                   \
{                                                                                                                       \
    HostChecker check_obj_host;                                                                                         \
    run_test<::std::true_type, ::std::true_type>(check_obj_host);                                                       \
    if (!check_obj_host.errors.empty())                                                                                 \
    {                                                                                                                   \
        std::sort(check_obj_host.errors.begin(), check_obj_host.errors.end());                                          \
        std::cout << "Errors on host:";                                                                                 \
        bool bFirst = true;                                                                                             \
        int nLastLine = -1;                                                                                             \
        HostChecker::ErrorInfo last;                                                                                    \
        for (const auto& errorInfo : check_obj_host.errors)                                                             \
        {                                                                                                               \
            if (errorInfo == last)                                                                                      \
                continue;                                                                                               \
                                                                                                                        \
            last = errorInfo;                                                                                           \
                                                                                                                        \
            if (errorInfo.lineNo != nLastLine)                                                                          \
            {                                                                                                           \
                std::cout << "\n\t";                                                                                    \
                nLastLine = errorInfo.lineNo;                                                                           \
                                                                                                                        \
                std::cout << "line " << errorInfo.lineNo << " : ";                                                      \
                bFirst = true;                                                                                          \
            }                                                                                                           \
            if (!bFirst)                                                                                                \
                std::cout << ", ";                                                                                      \
            else                                                                                                        \
                bFirst = false;                                                                                         \
            std::cout << "(" << errorInfo.i << ", " << errorInfo.j << ")";                                              \
        }                                                                                                               \
        std::cout << std::endl;                                                                                         \
        std::exit(EXIT_FAILURE);                                                                                        \
    }                                                                                                                   \
                                                                                                                        \
    /* Sometimes we may start test on device, which don't support type double. */                                       \
    /* In this case generates run-time error.                                  */                                       \
    /* This two types allow us to avoid this situation.                        */                                       \
    using HasDoubleTypeSupportInRuntime = ::std::true_type;                                                             \
    using HasntDoubleTypeSupportInRuntime = ::std::false_type;                                                          \
                                                                                                                        \
    /* long double type generate compile-time error in Kernel code             */                                       \
    /* and we never can use this type inside Kernel                            */                                       \
    using HasntLongDoubleSupportInCompiletime = ::std::false_type;                                                      \
                                                                                                                        \
    std::cout << "Run test on device" << std::endl;                                                                     \
    KernelChecker check_obj_kernel;                                                                                     \
    TestUtils::run_test_in_kernel(                                                                                      \
        /* lambda for the case when we have support of double type on device */                                         \
        [check_obj_kernel]() { run_test<HasDoubleTypeSupportInRuntime, HasntLongDoubleSupportInCompiletime>(check_obj_kernel); },       \
        /* lambda for the case when we haven't support of double type on device */                                                      \
        [check_obj_kernel]() { run_test<HasntDoubleTypeSupportInRuntime, HasntLongDoubleSupportInCompiletime>(check_obj_kernel); });    \
                                                                                                                        \
    return TestUtils::done();                                                                                           \
}                                                                                                                       \
                                                                                                                        \
template <typename HasDoubleSupportInRuntime, typename HasLongDoubleSupportInCompiletime,                               \
          class TChecker>                                                                                               \
int                                                                                                                     \
run_test(TChecker& check_obj)

// We should use this macros to avoid runtime-error if type double doesn't supported on device.
//
// Example:
//     template <class T>
//     void
//     test(T x, ::std::enable_if_t<std::is_integral_v<T>>* = 0)
//     {
//         static_assert((std::is_same_v<decltype(dpl::conj(x)), dpl::complex<double>>));
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
//         IF_DOUBLE_SUPPORT(test<int>())
//         // ...
//     }
#define IF_DOUBLE_SUPPORT(...)                                                                        \
    TestUtils::invoke_test_if(HasDoubleSupportInRuntime(), []() { __VA_ARGS__; });

#define IF_DOUBLE_SUPPORT_REF_CAPT(...)                                                               \
    TestUtils::invoke_test_if(HasDoubleSupportInRuntime(), [&]() { __VA_ARGS__; });

#define IF_DOUBLE_SUPPORT_L(...)                                                                      \
    TestUtils::invoke_test_if(HasDoubleSupportInRuntime(), __VA_ARGS__);

// We should use this macros to avoid compile-time error in code with long double type in Kernel.
#define IF_LONG_DOUBLE_SUPPORT(...)                                                                   \
    TestUtils::invoke_test_if(HasLongDoubleSupportInCompiletime(), []() { __VA_ARGS__; });
#define IF_LONG_DOUBLE_SUPPORT_L(...)                                                                 \
    TestUtils::invoke_test_if(HasLongDoubleSupportInCompiletime(), __VA_ARGS__);

#define CALL_CHECK_OBJ_I(OBJ, I, COND) OBJ(I, COND, __LINE__)
#define CALL_CHECK_OBJ_I_J(OBJ, I, J, COND) OBJ(I, J, COND, __LINE__)

namespace TestUtils
{
    template <typename _FncTest>
    void
    invoke_test_if(::std::true_type, _FncTest __fncTest)
    {
        __fncTest();
    }

    template <typename _FncTest>
    void
    invoke_test_if(::std::false_type, _FncTest)
    {
    }

    // Run test in Kernel as single task
    template <typename TFncDoubleHasSupportInRuntime, typename TFncDoubleHasntSupportInRuntime>
    void
    run_test_in_kernel(TFncDoubleHasSupportInRuntime fncDoubleHasSupportInRuntime,
                       TFncDoubleHasntSupportInRuntime fncDoubleHasntSupportInRuntime)
    {
#if TEST_DPCPP_BACKEND_PRESENT
        try
        {
            sycl::queue deviceQueue = TestUtils::get_test_queue();

            const auto device = deviceQueue.get_device();

            // We should run fncDoubleHasSupportInRuntime and fncDoubleHasntSupportInRuntime
            // in two separate Kernels to have ability compile these Kernels separatelly
            // by using Intel(R) oneAPI DPC++/C++ Compiler option -fsycl-device-code-split=per_kernel
            // which described at
            // https://www.intel.com/content/www/us/en/develop/documentation/oneapi-gpu-optimization-guide/top/compilation/jitting.html
            if (has_type_support<double>(device))
            {
                deviceQueue.submit(
                    [&](sycl::handler& cgh) {
                        cgh.single_task<TestUtils::new_kernel_name<class TestType, 0>>(
                            [fncDoubleHasSupportInRuntime]() { fncDoubleHasSupportInRuntime(); });
                    });
            }
            else
            {
                deviceQueue.submit(
                    [&](sycl::handler& cgh) {
                        cgh.single_task<TestUtils::new_kernel_name<class TestType, 1>>(
                            [fncDoubleHasntSupportInRuntime]() { fncDoubleHasntSupportInRuntime(); });
                    });
            }
            deviceQueue.wait_and_throw();
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

#endif // _TEST_COMPLEX_H
