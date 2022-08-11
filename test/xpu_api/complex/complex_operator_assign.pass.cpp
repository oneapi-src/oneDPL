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
// class TextComplexAssign - testing of some std::complex operators
// 
// described https://en.cppreference.com/w/cpp/numeric/complex/operator%3D
//
// (1)
//      Primary template complex<T>
//           complex& operator=( const T& x );                   (until C++20)
//           constexpr complex& operator=( const T& x );         (since C++20)
//      
//      Specialization complex<float>
//           complex& operator=( float x );                      (until C++20)
//           constexpr complex& operator=( float x );            (since C++20)
//      
//      Specialization complex<double>
//           complex& operator=( double x );                     (until C++20)
//           constexpr complex& operator=( double x );           (since C++20)
//      
//      Specialization complex<long double>
//           complex& operator=( long double x );                (until C++20)
//           constexpr complex& operator=( long double x );      (since C++20)
// 
// All specializations
//  (2)
//      complex& operator=( const complex& cx );                 (until C++20)
//      constexpr complex& operator=( const complex& cx );       (since C++20)
// 
//  (3)
//      template< class X >
//      complex&
//      operator=(const std::complex<X>& cx);                    (until C++20)
//      template< class X >
//      constexpr complex&
//      operator=(const std::complex<X>& cx);                    (since C++20)
//
template <typename TErrorsContainer, typename IsSupportedDouble, typename IsSupportedLongDouble>
class TextComplexAssign
{
public:

    TextComplexAssign(TErrorsContainer& ee)
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
        dpl::complex<T> cv1;
        constexpr T kConst = TestUtils::Complex::TestConstants<T>::kPartReal;

        cv1 = kConst;

        EXPECT_TRUE_EE(errors, cv1.real() == kConst, "Wrong effect in std::complex<T> complex& operator=( const T& x )");
        EXPECT_TRUE_EE(errors, cv1.imag() == T{}, "Wrong effect in std::complex<T> complex& operator=( const T& x )");
    }


    template <class T>
    void test_primary_form_2()
    {
        dpl::complex<T> cv1(TestUtils::Complex::TestConstants<T>::kPartReal, TestUtils::Complex::TestConstants<T>::kPartImag);
        dpl::complex<T> cv2;

        cv2 = cv1;

        EXPECT_TRUE_EE(errors, cv2.real() == cv1.real(), "Wrong effect in std::complex<T> complex& operator=( const std::complex<X>& cx )");
        EXPECT_TRUE_EE(errors, cv2.imag() == cv1.imag(), "Wrong effect in std::complex<T> complex& operator=( const std::complex<X>& cx )");
    }

private:

    TErrorsContainer& errors;
};

int
main()
{
    bool bSuccess = TestUtils::Complex::test_on_host<TextComplexAssign>();

#if TEST_DPCPP_BACKEND_PRESENT
    try
    {
        sycl::queue deviceQueue{ TestUtils::default_selector };

        if (!TestUtils::Complex::test_in_kernel<TextComplexAssign>(deviceQueue, TestUtils::kMaxKernelErrorsCount))
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
