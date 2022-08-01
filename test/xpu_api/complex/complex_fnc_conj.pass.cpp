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
// class TestComplexConj - testing of std::conj from <complex>
// 
// Function std::conj described https://en.cppreference.com/w/cpp/numeric/complex/conj :
// (1)
//      (until C++20)
//          template< class T >
//          std::complex<T> conj(const std::complex<T>& z);
//      (since C++20)
//          template< class T >
//          constexpr std::complex<T> conj(const std::complex<T>& z);
// (2)
//      (since C++11) (until C++20)
//          std::complex<float> conj(float z);
//          template< class DoubleOrInteger >
//          std::complex<double> conj(DoubleOrInteger z);
//          std::complex<long double> conj(long double z);
//      (since C++20)
//          constexpr std::complex<float> conj(float z);
//          template< class DoubleOrInteger >
//          constexpr std::complex<double> conj(DoubleOrInteger z);
//          constexpr std::complex<long double> conj(long double z);
template <typename TErrorEngine, typename IsSupportedDouble, typename IsSupportedLongDouble>
class TestComplexConj
{
public:

    TestComplexConj(TErrorEngine& ee)
        : errorEngine(ee)
    {
    }

    template <typename RunOnHost>
    void run_test(RunOnHost runOnHost)
    {
        test_form_1<float>();

        // Sometimes device, on which SYCL::queue work, may not support double type
        TestUtils::invoke_test_if<IsSupportedDouble>()([&](){ test_form_1<double>(); });

        // Type "long double" not specified in https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html#table.types.fundamental
        TestUtils::invoke_test_if<IsSupportedLongDouble>()([&](){ test_form_1<long double>(); });

        test_form_2<float>();

        // Sometimes device, on which SYCL::queue work, may not support double type
        TestUtils::invoke_test_if<IsSupportedDouble>()([&]() { test_form_2<double>(); });

        run_test_integer(runOnHost);

        // Type "long double" not specified in https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html#table.types.fundamental
        TestUtils::invoke_test_if<IsSupportedLongDouble>()([&]() { test_form_2<long double>(); });
    }

protected:

    void run_test_integer(::std::true_type)
    {
        // Test in type on host
        test_form_2<int>();                 // DoubleOrInteger, result type checked
    }

    void run_test_integer(::std::false_type)
    {
        // Test int type in Kernel
        test_form_2<int>();                 // DoubleOrInteger, result type checked
    }

    template <typename T>
    void test_form_1()
    {
        test_form_1_until_CPP20<T>();
        test_form_1_since_CPP20<T>();
    }

    template <typename T>
    void test_form_2()
    {
        test_form_2_until_CPP20<T>();
        test_form_2_since_CPP20<T>();
    }

    template <typename T>
    void test_form_1_until_CPP20()
    {
#if __cplusplus < 202002L
        const dpl::complex<T> cv(TestUtils::Complex::InitConst<T>::kPartReal, TestUtils::Complex::InitConst<T>::kPartImag);
        EXPECT_EQ_TYPE_EE(errorEngine, typename TestUtils::Complex::InitConst<T>::DestComplexFieldType, cv.real());
        EXPECT_EQ_TYPE_EE(errorEngine, typename TestUtils::Complex::InitConst<T>::DestComplexFieldType, cv.imag());

        const dpl::complex<T> cv_conj = dpl::conj(cv);
        EXPECT_EQ_TYPE_EE(errorEngine, typename TestUtils::Complex::InitConst<T>::DestComplexFieldType, cv_conj.real());
        EXPECT_EQ_TYPE_EE(errorEngine, typename TestUtils::Complex::InitConst<T>::DestComplexFieldType, cv_conj.imag());

        EXPECT_TRUE_EE(errorEngine, cv.real() == cv_conj.real(), "Wrong effect of conj #1");
        EXPECT_TRUE_EE(errorEngine, cv.imag() == cv_conj.imag() * -1, "Wrong effect of conj #2");
#endif
    }

    template <typename T>
    void test_form_1_since_CPP20()
    {
#if __cplusplus >= 202002L
        const dpl::complex<T> cv(TestUtils::Complex::InitConst<T>::kPartReal, TestUtils::Complex::InitConst<T>::kPartImag);
        EXPECT_EQ_TYPE_EE(errorEngine, typename TestUtils::Complex::InitConst<T>::DestComplexFieldType, cv.real());
        EXPECT_EQ_TYPE_EE(errorEngine, typename TestUtils::Complex::InitConst<T>::DestComplexFieldType, cv.imag());

        COMPLEX_TEST_CONSTEXPR dpl::complex<T> cv_conj = dpl::conj(cv);
        EXPECT_EQ_TYPE_EE(errorEngine, typename TestUtils::Complex::InitConst<T>::DestComplexFieldType, cv_conj.real());
        EXPECT_EQ_TYPE_EE(errorEngine, typename TestUtils::Complex::InitConst<T>::DestComplexFieldType, cv_conj.imag());

        EXPECT_TRUE_EE(errorEngine, cv.real() == cv_conj.real(), "Wrong effect of conj #3");
        EXPECT_TRUE_EE(errorEngine, cv.imag() == cv_conj.imag() * -1, "Wrong effect of conj #4");
#endif
    }

    template <typename T>
    void test_form_2_until_CPP20()
    {
#if __cplusplus < 202002L
        T z = TestUtils::Complex::InitConst<T>::kPartReal;

        dpl::complex<T> cv_conj = dpl::conj(z);
        EXPECT_EQ_TYPE_EE(errorEngine, typename TestUtils::Complex::InitConst<T>::DestComplexFieldType, cv_conj.real());
        EXPECT_EQ_TYPE_EE(errorEngine, typename TestUtils::Complex::InitConst<T>::DestComplexFieldType, cv_conj.imag());

        EXPECT_TRUE_EE(errorEngine, cv_conj.real() == z, "Wrong effect of conj #5");
        EXPECT_TRUE_EE(errorEngine, cv_conj.imag() == TestUtils::Complex::InitConst<T>::kZero, "Wrong effect of conj #6");
#endif
    }

    template <typename T>
    void test_form_2_since_CPP20()
    {
#if __cplusplus >= 202002L
        T z = TestUtils::Complex::InitConst<T>::kPartReal;

        COMPLEX_TEST_CONSTEXPR dpl::complex<T> cv_conj = dpl::conj(z);
        EXPECT_EQ_TYPE_EE(errorEngine, typename TestUtils::Complex::InitConst<T>::DestComplexFieldType, cv_conj.real());
        EXPECT_EQ_TYPE_EE(errorEngine, typename TestUtils::Complex::InitConst<T>::DestComplexFieldType, cv_conj.imag());

        EXPECT_TRUE_EE(errorEngine, cv_conj.real() == z, "Wrong effect of conj #7");
        EXPECT_TRUE_EE(errorEngine, cv_conj.imag() == 0, "Wrong effect of conj #8");
#endif
    }

private:

    TErrorEngine& errorEngine;
};

int
main()
{
    bool bSuccess = true;

    if (!TestUtils::Complex::test_on_host<TestComplexConj>())
        bSuccess = false;

#if TEST_DPCPP_BACKEND_PRESENT
    try
    {
        sycl::queue deviceQueue{ TestUtils::default_selector };

        if (!TestUtils::Complex::test_in_kernel<TestComplexConj>(deviceQueue))
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
