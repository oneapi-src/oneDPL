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
// class TextComplexArith2 - testing of some std::complex operators
// 
// described https://en.cppreference.com/w/cpp/numeric/complex/operator_arith2
//
//  (1)
//      template <class T>
//      std::complex<T>
//      operator+(const std::complex<T>& val);      (until C++ 20)
// 
//      template <class T>
//      constexpr std::complex<T>
//      operator+(const std::complex<T>& val);      (since C++ 20)
// 
//  (2)
//      template <class T>
//      std::complex<T>
//      operator-(const std::complex<T>& val);      (until C++ 20)
// 
//      template <class T>
//      constexpr std::complex<T>
//      operator-(const std::complex<T>& val);      (since C++ 20)
//
template <typename TErrorsContainer, typename IsSupportedDouble, typename IsSupportedLongDouble>
class TextComplexArith2
{
public:

    TextComplexArith2(TErrorsContainer& ee)
        : errors(ee)
    {
    }

    template <typename RunOnHost>
    void run_test(RunOnHost /*runOnHost*/)
    {
        // 202002L
        test_primary<float>();

        oneapi::dpl::__internal::__invoke_if(IsSupportedDouble(), [&]() { test_primary<double>(); });

        oneapi::dpl::__internal::__invoke_if(IsSupportedLongDouble(), [&]() { test_primary<long double>(); });
    }

protected:

    template <class T>
    void test_primary()
    {
        test_primary_form_1<T>();
        test_primary_form_2<T>();
    }

    template <class T>
    void test_primary_form_1()
    {
        COMPLEX_TEST_CONSTEXPR dpl::complex<T> cv1(TestUtils::Complex::InitConst<T>::kPartReal, TestUtils::Complex::InitConst<T>::kPartImag);
        COMPLEX_TEST_CONSTEXPR dpl::complex<T> cv2(TestUtils::Complex::InitConst<T>::kPartReal, TestUtils::Complex::InitConst<T>::kPartImag);

        COMPLEX_TEST_CONSTEXPR auto cv3 = cv1 + cv2;
        static_assert(::std::is_same<typename ::std::decay<decltype(cv3.real())>::type, T>::value);
        static_assert(::std::is_same<typename ::std::decay<decltype(cv3.imag())>::type, T>::value);

        EXPECT_TRUE_EE(errors, cv3.real() == cv1.real() + cv2.real(), "Wrong effect in operator+");
        EXPECT_TRUE_EE(errors, cv3.imag() == cv1.imag() + cv2.imag(), "Wrong effect in operator+");
    }

    template <class T>
    void test_primary_form_2()
    {
        COMPLEX_TEST_CONSTEXPR dpl::complex<T> cv1(TestUtils::Complex::InitConst<T>::kPartReal, TestUtils::Complex::InitConst<T>::kPartImag);
        COMPLEX_TEST_CONSTEXPR dpl::complex<T> cv2(TestUtils::Complex::InitConst<T>::kPartReal, TestUtils::Complex::InitConst<T>::kPartImag);

        COMPLEX_TEST_CONSTEXPR auto cv3 = cv1 - cv2;
        static_assert(::std::is_same<typename ::std::decay<decltype(cv3.real())>::type, T>::value);
        static_assert(::std::is_same<typename ::std::decay<decltype(cv3.imag())>::type, T>::value);

        EXPECT_TRUE_EE(errors, cv3.real() == cv1.real() - cv2.real(), "Wrong effect in operator-");
        EXPECT_TRUE_EE(errors, cv3.imag() == cv1.imag() - cv2.imag(), "Wrong effect in operator-");
    }

private:

    TErrorsContainer& errors;
};

int
main()
{
    bool bSuccess = TestUtils::Complex::test_on_host<TextComplexArith2>();

#if TEST_DPCPP_BACKEND_PRESENT
    try
    {
        sycl::queue deviceQueue{ TestUtils::default_selector };

        if (!TestUtils::Complex::test_in_kernel<TextComplexArith2>(deviceQueue, TestUtils::kMaxKernelErrorsCount))
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
