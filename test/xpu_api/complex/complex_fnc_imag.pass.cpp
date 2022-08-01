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
// class TestComplexFncImag - testing of std::imag
// 
// described https://en.cppreference.com/w/cpp/numeric/complex/imag2
//      (1)
//          (until C++14)
//              template< class T >
//              T imag(const std::complex<T>& z);
//          (since C++14)
//              template< class T >
//              constexpr T imag(const std::complex<T>& z);
//      (2)
//          (since C++11)(until C++14)
//              float imag( float z );
//              template< class DoubleOrInteger >
//              double imag(DoubleOrInteger z);
//              long double imag(long double z);
//          (since C++14)
//              constexpr float imag( float z );
//              template< class DoubleOrInteger >
//              constexpr double imag(DoubleOrInteger z);
//              constexpr long double imag(long double z);
//
template <typename TErrorEngine, typename IsSupportedDouble, typename IsSupportedLongDouble>
class TestComplexFncImag
{
public:

    TestComplexFncImag(TErrorEngine& ee)
        : errorEngine(ee)
    {
    }

    template <typename RunOnHost>
    void run_test(RunOnHost /*runOnHost*/)
    {
        test_fnc_imag<float>();
        TestUtils::invoke_test_if<IsSupportedDouble>()([&]() { test_fnc_imag<double>(); });
        TestUtils::invoke_test_if<IsSupportedLongDouble>()([&]() { test_fnc_imag<long double>(); });
    }

protected:

    template <typename T>
    void test_fnc_imag()
    {
        test_fnc_imag_form1<T>();
        test_fnc_imag_form2<T>();
        test_fnc_imag_form2<int>();     // DoubleOrInteger, result type checked
    }

    template <typename T>
    void test_fnc_imag_form1()
    {
        test_fnc_imag_form1_until_CPP14<T>();
        test_fnc_imag_form1_since_CPP14<T>();
    }

    template <typename T>
    void test_fnc_imag_form2()
    {
        test_fnc_imag_form2_until_CPP14<T>();
        test_fnc_imag_form2_since_CPP14<T>();
    }

    template <typename T>
    void test_fnc_imag_form1_until_CPP14()
    {
#if __cplusplus < 201402L
        const dpl::complex<T> cv(TestUtils::Complex::InitConst<T>::kPartReal, TestUtils::Complex::InitConst<T>::kPartImag);
        EXPECT_TRUE_EE(errorEngine, TestUtils::Complex::InitConst<T>::kPartImag == dpl::imag(cv), "Wrong effect of dpl::imag() #1");
#endif
    }

    template <typename T>
    void test_fnc_imag_form1_since_CPP14()
    {
#if __cplusplus >= 201402L
        COMPLEX_TEST_CONSTEXPR dpl::complex<T> cv(TestUtils::Complex::InitConst<T>::kPartReal, TestUtils::Complex::InitConst<T>::kPartImag);
        EXPECT_TRUE_EE(errorEngine, TestUtils::Complex::InitConst<T>::kPartImag == dpl::imag(cv), "Wrong effect of dpl::imag() #2");
#endif
    }

    template <typename T>
    void test_fnc_imag_form2_until_CPP14()
    {
#if __cplusplus < 201402L
        const T z = TestUtils::Complex::InitConst<T>::kPartImag;
        const auto imag_res = dpl::imag(z);
        EXPECT_EQ_TYPE_EE(errorEngine, typename TestUtils::Complex::InitConst<T>::DestComplexFieldType, imag_res);

        const typename TestUtils::Complex::InitConst<T>::DestComplexFieldType imag_res_expected = { };
        EXPECT_TRUE_EE(errorEngine, imag_res_expected == imag_res, "Wrong effect of dpl::imag() #3");
#endif
    }

    template <typename T>
    void test_fnc_imag_form2_since_CPP14()
    {
#if __cplusplus >= 201402L
        COMPLEX_TEST_CONSTEXPR T z = TestUtils::Complex::InitConst<T>::kPartImag;
        COMPLEX_TEST_CONSTEXPR auto imag_res = dpl::imag(z);
        EXPECT_EQ_TYPE_EE(errorEngine, typename TestUtils::Complex::InitConst<T>::DestComplexFieldType, imag_res);


        COMPLEX_TEST_CONSTEXPR typename TestUtils::Complex::InitConst<T>::DestComplexFieldType imag_res_expected = { };
        EXPECT_TRUE_EE(errorEngine, imag_res_expected == imag_res, "Wrong effect of dpl::imag() #4");
#endif
    }

private:

    TErrorEngine& errorEngine;
};

int
main()
{
    TestUtils::Complex::test_on_host<TestComplexFncImag>();

#if TEST_DPCPP_BACKEND_PRESENT
    try
    {
        sycl::queue deviceQueue{ TestUtils::default_selector };

        TestUtils::Complex::test_in_kernel<TestComplexFncImag>(deviceQueue);
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
