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

#include "support/utils_complex_test_support.h"

#include "complex_fnc_abs_cases.h"

////////////////////////////////////////////////////////////////////////////////
// class TestComplexAbs - testing of std::abs from <complex>
// 
// Function std::conj described https://en.cppreference.com/w/cpp/numeric/complex/abs :
//      template< class T >
//      T abs(const complex<T>& z);
template <typename TErrorEngine, typename IsSupportedDouble, typename IsSupportedLongDouble>
class TestComplexAbs
{
public:

    TestComplexAbs(TErrorEngine& ee)
        : errorEngine(ee)
    {
    }

    template <typename RunOnHost>
    void run_test(RunOnHost /*runOnHost*/)
    {
        test_abs<float>();

        // Sometimes device, on which SYCL::queue work, may not support double type
        TestUtils::invoke_test_if<IsSupportedDouble>()([&](){ test_abs<double>(); });

        // Type "long double" not specified in https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html#table.types.fundamental
        TestUtils::invoke_test_if<IsSupportedLongDouble>()([&](){ test_abs<long double>(); });

        // Test cases from libxcxx checks
        TestUtils::invoke_test_if<IsSupportedDouble>()([&]() { test_edges(); });
    }

protected:

    template <typename T>
    void test_abs()
    {
        const auto cv = dpl::complex<T>(TestUtils::Complex::InitConst<T>::kPartReal, TestUtils::Complex::InitConst<T>::kPartImag);
        const auto abs_res = dpl::abs(cv);
        const auto abs_res_expected = ::std::abs(cv);
        EXPECT_TRUE_EE(errorEngine, abs_res == abs_res_expected, "Wrong result in dpl::abs(dpl::complex<T>()) function");
    }

    void test_edges()
    {
        const unsigned N = sizeof(testcases) / sizeof(testcases[0]);
        for (unsigned i = 0; i < N; ++i)
        {
            const auto cv = testcases[i];
            const auto abs_res = dpl::abs(cv);
            const auto abs_res_expected = ::std::abs(cv);
            EXPECT_TRUE_EE(errorEngine, abs_res == abs_res_expected, "Wrong result in dpl::abs(dpl::complex<T>()) function #1");
        }
    }

private:

    TErrorEngine& errorEngine;
};

int
main()
{
    bool bSuccess = true;

    if (!TestUtils::Complex::test_on_host<TestComplexAbs>())
        bSuccess = false;

#if TEST_DPCPP_BACKEND_PRESENT
    try
    {
        sycl::queue deviceQueue{ TestUtils::default_selector };

        if (!TestUtils::Complex::test_in_kernel<TestComplexAbs>(deviceQueue))
            bSuccess = false;
    }
    catch (const std::exception& exc)
    {
        bSuccess = false;

        std::string errorMsg = "Exception occurred";
        if (exc.what())
        {
            errorMsg += " : ";
            errorMsg += exc.what();
        }

        EXPECT_TRUE(false, errorMsg.c_str());
    }
#endif // TEST_DPCPP_BACKEND_PRESENT

    if (!bSuccess)
        TestUtils::exit_on_error();

    return TestUtils::done();
}
