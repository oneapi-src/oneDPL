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

////////////////////////////////////////////////////////////////////////////////
// class TestComplexExp - testing of std::exp from <complex>
// 
// Function std::conj described https://en.cppreference.com/w/cpp/numeric/complex/exp :
// 
//      template< class T >
//      complex<T> exp(const complex<T>& z);
template <typename TErrorsContainer, typename IsSupportedDouble, typename IsSupportedLongDouble>
class TestComplexExp
{
public:

    TestComplexExp(TErrorsContainer& ee)
        : errors(ee)
    {
    }

    template <typename RunOnHost>
    void run_test(RunOnHost /*runOnHost*/)
    {
        test_exp<float>("Wrong result in dpl::exp(dpl::complex<T>()) function (float)");

        // Sometimes device, on which SYCL::queue work, may not support double type
        TestUtils::invoke_test_if<IsSupportedDouble>()([&](){ test_exp<double>("Wrong result in dpl::exp(dpl::complex<T>()) function (double)"); });

        // Type "long double" not specified in https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html#table.types.fundamental
        TestUtils::invoke_test_if<IsSupportedLongDouble>()([&](){ test_exp<long double>("Wrong result in dpl::exp(dpl::complex<T>()) function (long double)"); });
    }

protected:

    template <typename T>
    void test_exp(const char* msg)
    {
        const auto cv = dpl::complex<T>(TestUtils::Complex::InitConst<T>::kPartReal, TestUtils::Complex::InitConst<T>::kPartImag);
        const auto exp_res = dpl::exp(cv);
        auto exp_res_expected = ::std::exp(cv);
        EXPECT_TRUE_EE(errors, exp_res == exp_res_expected, msg);
    }

private:

    TErrorsContainer& errors;
};

int
main()
{
    bool bSuccess = TestUtils::Complex::test_on_host<TestComplexExp>();

#if TEST_DPCPP_BACKEND_PRESENT
    try
    {
        sycl::queue deviceQueue{ TestUtils::default_selector };

        if (!TestUtils::Complex::test_in_kernel<TestComplexExp>(deviceQueue))
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

        TestUtils::issue_error_message(errorMsg);
    }
#endif // TEST_DPCPP_BACKEND_PRESENT

    if (!bSuccess)
        TestUtils::exit_on_error();

    return TestUtils::done();
}
