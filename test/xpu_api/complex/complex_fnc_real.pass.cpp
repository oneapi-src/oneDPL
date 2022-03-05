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
// class TestComplexReal - testing of constexpr std::real
// 
// described https://en.cppreference.com/w/cpp/numeric/complex/real2
//      (1) 
//          (until C++14)
//              template< class T >
//              T real(const std::complex<T>& z);
//          (since C++14)
//              template< class T >
//              constexpr T real(const std::complex<T>& z);
//      (2)
//          (since C++11) (until C++14)
//              float real( float z );
//              template< class DoubleOrInteger >
//              double real(DoubleOrInteger z);
//              long double real(long double z);
//          (since C++14)
//              constexpr float real( float z );
//              template< class DoubleOrInteger >
//              constexpr double real(DoubleOrInteger z);
//              constexpr long double real(long double z);
//      
template <typename TErrorEngine, typename IsSupportedDouble, typename IsSupportedLongDouble>
class TestComplexFncReal
{
public:

    TestComplexFncReal(TErrorEngine& ee)
        : errorEngine(ee)
    {
    }

    template <typename RunOnHost>
    void run_test(RunOnHost /*runOnHost*/)
    {
        test_fnc_real<float>();
        TestUtils::invoke_test_if<IsSupportedDouble>()([&]() { test_fnc_real<double>(); });
        TestUtils::invoke_test_if<IsSupportedLongDouble>()([&]() { test_fnc_real<long double>(); });
    }

protected:

    template <typename T>
    void test_fnc_real()
    {
        test_fnc_real_form1<T>();
        test_fnc_real_form2<T>();
        test_fnc_real_form2<int>();     // DoubleOrInteger, result type checked
    }

    template <typename T>
    void test_fnc_real_form1()
    {
        test_fnc_real_form1_until_CPP14<T>();
        test_fnc_real_form1_since_CPP14<T>();
    }

    template <typename T>
    void test_fnc_real_form2()
    {
        test_fnc_real_form2_until_CPP14<T>();
        test_fnc_real_form2_since_CPP14<T>();
    }

    template <typename T>
    void test_fnc_real_form1_until_CPP14()
    {
#if __cplusplus < 201402L
        const dpl::complex<T> cv(TestUtils::Complex::InitConst<T>::kPartReal);
        EXPECT_TRUE_EE(errorEngine, TestUtils::Complex::InitConst<T>::kPartReal == dpl::real(cv), "Wrong effect of dpl::real() #1");
#endif
    }

    template <typename T>
    void test_fnc_real_form1_since_CPP14()
    {
#if __cplusplus >= 201402L
        COMPLEX_TEST_CONSTEXPR dpl::complex<T> cv(TestUtils::Complex::InitConst<T>::kPartReal);
        EXPECT_TRUE_EE(errorEngine, TestUtils::Complex::InitConst<T>::kPartReal == dpl::real(cv), "Wrong effect of dpl::real() #2");
#endif
    }

    template <typename T>
    void test_fnc_real_form2_until_CPP14()
    {
#if __cplusplus < 201402L
        const T z = TestUtils::Complex::InitConst<T>::kPartReal;
        const auto real_res = dpl::real(z);
        TestUtils::Complex::check_type<typename TestUtils::Complex::InitConst<T>::DestComplexFieldType>(real_res);
        
        EXPECT_TRUE_EE(errorEngine, TestUtils::Complex::InitConst<T>::kExpectedResReal == real_res, "Wrong effect of dpl::real() #3");
#endif
    }

    template <typename T>
    void test_fnc_real_form2_since_CPP14()
    {
#if __cplusplus >= 201402L
        COMPLEX_TEST_CONSTEXPR T z = TestUtils::Complex::InitConst<T>::kPartReal;
        COMPLEX_TEST_CONSTEXPR auto real_res = dpl::real(z);
        TestUtils::Complex::check_type<typename TestUtils::Complex::InitConst<T>::DestComplexFieldType>(real_res);

        EXPECT_TRUE_EE(errorEngine, TestUtils::Complex::InitConst<T>::kExpectedResReal == real_res, "Wrong effect of dpl::real() #4");
#endif
    }

private:

    TErrorEngine& errorEngine;
};

int
main()
{
    TestUtils::Complex::test_on_host<TestComplexFncReal>();

#if TEST_DPCPP_BACKEND_PRESENT
    try
    {
        sycl::queue deviceQueue{ TestUtils::default_selector };

        TestUtils::Complex::test_in_kernel<TestComplexFncReal>(deviceQueue);
    }
    catch (const std::exception& exc)
    {
        std::string errorMsg = "Exception occurred";
        if (exc.what())
        {
            errorMsg += " : ";
            errorMsg += exc.what();
        }

        EXPECT_TRUE(false, errorMsg.c_str());
    }
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done();
}
