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
// class TextComplexMethodReal - testing of std::complex<T>::real
// 
// described https://en.cppreference.com/w/cpp/numeric/complex/real
// 
//      primary template complex<T>
//          (1)
//              (until C++14)
//                  T real() const;
//              (since C++14)
//                  constexpr T real() const;
//          (2)
//              (until C++20)
//                  void real( T value );
//              (since C++20)
//                  constexpr void real( T value );
// 
//      specialization complex<float>
//          (1)
//              (until C++11)
//                  float real() const;
//              (since C++11)
//                  constexpr float real() const;
//          (2)
//              (until C++20)
//                  void real( float value );
//              (since C++20)
//                  constexpr void real( float value );
// 
//      specialization complex<double>
//          (1)
//              (until C++11)
//                  double real() const;
//              (since C++11)
//                  constexpr double real() const;
//          (2)
//              (until C++20)
//                  void real( double value );
//              (since C++20)
//                  constexpr void real( double value );
// 
//      specialization complex<long double>
//          (1)
//              (until C++11)
//                  long double real() const;
//              (since C++11)
//                  constexpr long double real() const;
//          (2)
//              (until C++20)
//                  void real( long double value );
//              (since C++20)
//                  constexpr void real( long double value );
//
template <typename TErrorEngine, typename IsSupportedDouble, typename IsSupportedLongDouble>
class TextComplexMethodReal
{
public:

    TextComplexMethodReal(TErrorEngine& ee)
        : errorEngine(ee)
    {
    }

    template <typename RunOnHost>
    void run_test(RunOnHost /*runOnHost*/)
    {
        test_method_real<float>();
        TestUtils::invoke_test_if<IsSupportedDouble>()([&]() { test_method_real<double>(); });
        TestUtils::invoke_test_if<IsSupportedLongDouble>()([&]() { test_method_real<long double>(); });
    }

protected:

    template <typename T>
    void test_method_real()
    {
        test_method_real_get<T>();
        test_method_real_set<T>();
    }

    template <typename T>
    void test_method_real_get()
    {
        test_method_real_get_until_CPP14<T>();
        test_method_real_get_since_CPP14<T>();
    }

    template <typename T>
    void test_method_real_set()
    {
        test_method_real_set_until_CPP20<T>();
        test_method_real_set_since_CPP20<T>();
    }

    template <typename T>
    void test_method_real_get_until_CPP14()
    {
#if __cplusplus < 201402L
        const dpl::complex<T> complex_val = dpl::complex<T>(TestUtils::Complex::InitConst<T>::kPartReal);
        EXPECT_TRUE_EE(errorEngine, TestUtils::Complex::InitConst<T>::kPartReal == complex_val.real(), "Wrong effect of dpl::complex::real() #1");
#endif
    }

    template <typename T>
    void test_method_real_get_since_CPP14()
    {
#if __cplusplus >= 201402L
        COMPLEX_TEST_CONSTEXPR dpl::complex<T> complex_val = dpl::complex<T>(TestUtils::Complex::InitConst<T>::kPartReal);
        EXPECT_TRUE_EE(errorEngine, TestUtils::Complex::InitConst<T>::kPartReal == complex_val.real(), "Wrong effect of dpl::complex::real() #2");
#endif
    }

    template <typename T>
    void test_method_real_set_until_CPP20()
    {
#if __cplusplus < 202002L
        dpl::complex<T> complex_val;
        complex_val.real(TestUtils::Complex::InitConst<T>::kPartReal);
        EXPECT_TRUE_EE(errorEngine, TestUtils::Complex::InitConst<T>::kPartReal == complex_val.real(), "Wrong effect of dpl::complex::real() #3");
#endif
    }

    template <typename T>
    void test_method_real_set_since_CPP20()
    {
#if __cplusplus >= 202002L
        COMPLEX_TEST_CONSTEXPR dpl::complex<T> complex_val;
        complex_val.real(TestUtils::Complex::InitConst<T>::kPartReal);
        EXPECT_TRUE_EE(errorEngine, TestUtils::Complex::InitConst<T>::kPartReal == complex_val.real(), "Wrong effect of dpl::complex::real() #4");
#endif
    }

private:

    TErrorEngine& errorEngine;
};

int
main()
{
    bool bSuccess = true;

    if (!TestUtils::Complex::test_on_host<TextComplexMethodReal>())
        bSuccess = false;

#if TEST_DPCPP_BACKEND_PRESENT
    try
    {
        sycl::queue deviceQueue{ TestUtils::default_selector };

        if (!TestUtils::Complex::test_in_kernel<TextComplexMethodReal>(deviceQueue))
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
