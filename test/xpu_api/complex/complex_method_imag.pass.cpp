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
// class TextComplexMethodImag - testing of constexpr std::complex<T>::imag()
// 
// described https://en.cppreference.com/w/cpp/numeric/complex/imag
// 
//      primary template complex<T>
//          (1)
//              (until C++14)
//                  T imag() const;
//              (since C++14)
//                  constexpr T imag() const;
//          (2)
//              (until C++20)
//                  void imag( T value );
//              (since C++20)
//                  constexpr void imag( T value );
// 
//      specialization complex<float>
//          (1)
//              (until C++11)
//                  float imag() const;
//              (since C++11)
//                  constexpr float imag() const;
//          (2)
//              (until C++20)
//                  void imag( float value );
//              (since C++20)
//                  constexpr void imag( float value );
// 
//      specialization complex<double>
//          (1)
//              (until C++11)
//                  double imag() const;
//              (since C++11)
//                  constexpr double imag() const;
//          (2)
//              (until C++20)
//                  void imag( double value );
//              (since C++20)
//                  constexpr void imag( double value );
// 
//      specialization complex<long double>
//          (1)
//              (until C++11)
//                  long double imag() const;
//              (since C++11)
//                  constexpr long double imag() const;
//          (2)
//              (until C++20)
//                  void imag( long double value );
//              (since C++20)
//                  constexpr void imag( long double value );
//
template <typename TErrorsContainer, typename IsSupportedDouble, typename IsSupportedLongDouble>
class TextComplexMethodImag
{
public:

    TextComplexMethodImag(TErrorsContainer& ee)
        : errors(ee)
    {
    }

    template <typename RunOnHost>
    void run_test(RunOnHost /*runOnHost*/)
    {
        test_method_imag<float>();
        oneapi::dpl::__internal::__invoke_if(IsSupportedDouble(), [&]() { test_method_imag<double>(); });
        oneapi::dpl::__internal::__invoke_if(IsSupportedLongDouble(), [&]() { test_method_imag<long double>(); });
    }

protected:

    template <typename T>
    void test_method_imag()
    {
        test_method_imag_get<T>();
        test_method_imag_set<T>();
    }

    template <typename T>
    void test_method_imag_get()
    {
        test_method_imag_get_until_CPP14<T>();
        test_method_imag_get_since_CPP14<T>();
    }

    template <typename T>
    void test_method_imag_set()
    {
        test_method_imag_set_until_CPP20<T>();
        test_method_imag_set_since_CPP20<T>();
    }

    template <typename T>
    void test_method_imag_get_until_CPP14()
    {
#if __cplusplus < 201402L
        const dpl::complex<T> complex_val = dpl::complex<T>(TestUtils::Complex::TestConstants<T>::kPartReal, TestUtils::Complex::TestConstants<T>::kPartImag);
        EXPECT_TRUE_EE(errors, TestUtils::Complex::TestConstants<T>::kPartImag == complex_val.imag(), "Wrong effect of dpl::complex::imag() #1");
#endif
    }

    template <typename T>
    void test_method_imag_get_since_CPP14()
    {
#if __cplusplus >= 201402L
        COMPLEX_TEST_CONSTEXPR dpl::complex<T> complex_val = dpl::complex<T>(TestUtils::Complex::TestConstants<T>::kPartReal, TestUtils::Complex::TestConstants<T>::kPartImag);
        EXPECT_TRUE_EE(errors, TestUtils::Complex::TestConstants<T>::kPartImag == complex_val.imag(), "Wrong effect of dpl::complex::imag() #2");
#endif
    }

    template <typename T>
    void test_method_imag_set_until_CPP20()
    {
#if __cplusplus < 202002L
        dpl::complex<T> complex_val;
        complex_val.imag(TestUtils::Complex::TestConstants<T>::kPartImag);
        EXPECT_TRUE_EE(errors, TestUtils::Complex::TestConstants<T>::kPartImag == complex_val.imag(), "Wrong effect of dpl::complex::imag() #3");
#endif
    }

    template <typename T>
    void test_method_imag_set_since_CPP20()
    {
#if __cplusplus >= 202002L
        COMPLEX_TEST_CONSTEXPR dpl::complex<T> complex_val;
        complex_val.imag(TestUtils::Complex::TestConstants<T>::kPartImag);
        EXPECT_TRUE_EE(errors, TestUtils::Complex::TestConstants<T>::kPartImag == complex_val.imag(), "Wrong effect of dpl::complex::imag() #4");
#endif
    }

private:

    TErrorsContainer& errors;
};

int
main()
{
    bool bSuccess = TestUtils::Complex::test_on_host<TextComplexMethodImag>();

#if TEST_DPCPP_BACKEND_PRESENT
    try
    {
        sycl::queue deviceQueue{ TestUtils::default_selector };

        if (!TestUtils::Complex::test_in_kernel<TextComplexMethodImag>(deviceQueue, TestUtils::kMaxKernelErrorsCount))
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
