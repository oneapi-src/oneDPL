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
// class TextComplexArith - testing of some std::complex operators
// 
// described https://en.cppreference.com/w/cpp/numeric/complex/operator_arith
// 
//  Primary template complex<T>
//      (1)
//          complex& operator+=(const T& other);                (until C++ 20)
//          constexpr complex& operator+=(const T& other);      (since C++ 20)
//      (2)
//          complex& operator-=(const T& other);                (until C++ 20) 
//          constexpr complex& operator-=(const T& other);      (since C++ 20)
//      (3)
//          complex& operator*=(const T& other);                (until C++ 20)
//          constexpr complex& operator*=(const T& other);      (since C++ 20)
//      (4)
//          complex& operator/=(const T& other);                (until C++ 20)
//          constexpr complex& operator/=(const T& other);      (since C++ 20)
//  Specialization complex<float>
//      (1)
//          complex& operator+=(float other);                   (until C++ 20)
//          constexpr complex& operator+=(float other);         (since C++ 20)
//      (2)
//          complex& operator-=(float other);                   (until C++ 20)
//          constexpr complex& operator-=(float other);         (since C++ 20)
//      (3)
//          complex& operator*=(float other);                   (until C++ 20)
//          constexpr complex& operator*=(float other);         (since C++ 20)
//      (4)
//          complex& operator/=(float other);                   (until C++ 20)
//          constexpr complex& operator/=(float other);         (since C++ 20)
//  Specialization complex<double>
//      (1)
//          complex& operator+=(double other);                  (until C++ 20)
//          constexpr complex& operator+=(double other);        (since C++ 20)
//      (2)
//          complex& operator-=(double other);                  (until C++ 20)
//          constexpr complex& operator-=(double other);        (since C++ 20)
//      (3)
//          complex& operator*=(double other);                  (until C++ 20)
//          constexpr complex& operator*=(double other);        (since C++ 20)
//      (4)
//          complex& operator/=(double other);                  (until C++ 20)
//          constexpr complex& operator/=(double other);        (since C++ 20)
//  Specialization complex<long double>
//      (1)
//          complex& operator+=(long double other);             (until C++ 20)
//          constexpr complex& operator+=(long double other);   (since C++ 20)
//      (2)
//          complex& operator-=(long double other);             (until C++ 20)
//          constexpr complex& operator-=(long double other);   (since C++ 20)
//      (3)
//          complex& operator*=(long double other);             (until C++ 20)
//          constexpr complex& operator*=(long double other);   (since C++ 20)
//      (4)
//          complex& operator/=(long double other);             (until C++ 20)
//          constexpr complex& operator/=(long double other);   (since C++ 20) 
//
// ATTENTION: all cases (until C++11) not covered by this test
//
template <typename TErrorsContainer, typename IsSupportedDouble, typename IsSupportedLongDouble>
class TextComplexArith
{
public:

    TextComplexArith(TErrorsContainer& ee)
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

        test_specialization<float>();

        oneapi::dpl::__internal::__invoke_if(IsSupportedDouble(), [&]() { test_specialization<double>(); });

        // Type "long double" not specified in https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html#table.types.fundamental
        oneapi::dpl::__internal::__invoke_if(IsSupportedLongDouble(), [&]() { test_specialization<long double>(); });
    }

protected:

    template <class T>
    void test_primary()
    {
        test_primary_form_1<T>();
        test_primary_form_2<T>();
        test_primary_form_3<T>();
        test_primary_form_4<T>();
    }

    template <class T>
    void test_specialization()
    {
        test_specialization_form_1<T>();
        test_specialization_form_2<T>();
        test_specialization_form_3<T>();
        test_specialization_form_4<T>();
    }

    template <class T>
    void test_primary_form_1()
    {
        COMPLEX_TEST_CONSTEXPR dpl::complex<T> cv1(TestUtils::Complex::TestConstants<T>::kPartReal, TestUtils::Complex::TestConstants<T>::kPartImag);
        COMPLEX_TEST_CONSTEXPR dpl::complex<T> cv2(TestUtils::Complex::TestConstants<T>::kPartReal, TestUtils::Complex::TestConstants<T>::kPartImag);

        cv2 += cv1;
        EXPECT_TRUE_EE(errors, cv2.real() == 2 * TestUtils::Complex::TestConstants<T>::kPartReal, "Wrong effect in complex& operator+=( const T& other )");
        EXPECT_TRUE_EE(errors, cv2.imag() == 2 * TestUtils::Complex::TestConstants<T>::kPartImag, "Wrong effect in complex& operator+=( const T& other )");
    }

    template <class T>
    void test_primary_form_2()
    {
        COMPLEX_TEST_CONSTEXPR dpl::complex<T> cv1(TestUtils::Complex::TestConstants<T>::kPartReal, TestUtils::Complex::TestConstants<T>::kPartImag);
        COMPLEX_TEST_CONSTEXPR dpl::complex<T> cv2(TestUtils::Complex::TestConstants<T>::kPartReal, TestUtils::Complex::TestConstants<T>::kPartImag);

        cv2 -= cv1;
        EXPECT_TRUE_EE(errors, cv2.real() == dpl::complex<T>().real(), "Wrong effect in complex& operator-=( const T& other )");
        EXPECT_TRUE_EE(errors, cv2.imag() == dpl::complex<T>().imag(), "Wrong effect in complex& operator-=( const T& other )");
    }

    template <class T>
    void test_primary_form_3()
    {
        COMPLEX_TEST_CONSTEXPR dpl::complex<T> cv1(TestUtils::Complex::TestConstants<T>::kPartReal, TestUtils::Complex::TestConstants<T>::kPartImag);
        COMPLEX_TEST_CONSTEXPR dpl::complex<T> cv2(TestUtils::Complex::TestConstants<T>::kPartReal, TestUtils::Complex::TestConstants<T>::kPartImag);

        const auto a1 = cv1.real();
        const auto a2 = cv1.imag();
        const auto b1 = cv2.real();
        const auto b2 = cv2.imag();

        cv2 *= cv1;
        EXPECT_TRUE_EE(errors, cv2.real() == a1 * b1 - a2 * b2, "Wrong effect in complex& operator*=( const T& other )");
        EXPECT_TRUE_EE(errors, cv2.imag() == a1 * b2 + a2 * b1, "Wrong effect in complex& operator*=( const T& other )");
    }

    template <class T>
    void test_primary_form_4()
    {
        COMPLEX_TEST_CONSTEXPR dpl::complex<T> cv1(TestUtils::Complex::TestConstants<T>::kPartReal, TestUtils::Complex::TestConstants<T>::kPartImag);
        COMPLEX_TEST_CONSTEXPR dpl::complex<T> cv2(TestUtils::Complex::TestConstants<T>::kPartReal, TestUtils::Complex::TestConstants<T>::kPartImag);

        const T expectedReal = (cv1.real() * cv2.real() + cv1.imag() * cv2.imag()) / (cv2.real() * cv2.real() + cv2.imag() * cv2.imag());
        const T expectedImag = (cv2.real() * cv1.imag() - cv1.real() * cv2.imag()) / (cv2.real() * cv2.real() + cv2.imag() * cv2.imag());

        cv2 /= cv1;

        EXPECT_TRUE_EE(errors, cv2.real() == expectedReal, "Wrong effect in complex& operator/=( const T& other )");
        EXPECT_TRUE_EE(errors, cv2.imag() == expectedImag, "Wrong effect in complex& operator/=( const T& other )");
    }

    template <class T>
    void test_specialization_form_1()
    {
        const T kConst = 2;

        dpl::complex<T> cv1(TestUtils::Complex::TestConstants<T>::kPartReal, TestUtils::Complex::TestConstants<T>::kPartImag);
        dpl::complex<T> cv2(TestUtils::Complex::TestConstants<T>::kPartReal, TestUtils::Complex::TestConstants<T>::kPartImag);

        cv1 += kConst;

        EXPECT_TRUE_EE(errors, cv1.real() == cv2.real() + kConst, "Wrong effect in operator+=");
        EXPECT_TRUE_EE(errors, cv1.imag() == cv2.imag(), "Wrong effect in operator+=");
    }

    template <class T>
    void test_specialization_form_2()
    {
        const T kConst = 2;

        dpl::complex<T> cv1(TestUtils::Complex::TestConstants<T>::kPartReal, TestUtils::Complex::TestConstants<T>::kPartImag);
        dpl::complex<T> cv2(TestUtils::Complex::TestConstants<T>::kPartReal, TestUtils::Complex::TestConstants<T>::kPartImag);

        cv1 -= kConst;

        EXPECT_TRUE_EE(errors, cv1.real() == cv2.real() - kConst, "Wrong effect in operator+=");
        EXPECT_TRUE_EE(errors, cv1.imag() == cv2.imag(), "Wrong effect in operator+=");
    }

    template <class T>
    void test_specialization_form_3()
    {
        const T kConst = 2;

        dpl::complex<T> cv1(TestUtils::Complex::TestConstants<T>::kPartReal, TestUtils::Complex::TestConstants<T>::kPartImag);
        dpl::complex<T> cv2(TestUtils::Complex::TestConstants<T>::kPartReal, TestUtils::Complex::TestConstants<T>::kPartImag);

        cv1 *= kConst;

        EXPECT_TRUE_EE(errors, cv1.real() == cv2.real() * kConst, "Wrong effect in operator*=");
        EXPECT_TRUE_EE(errors, cv1.imag() == cv2.imag() * kConst, "Wrong effect in operator*=");
    }

    template <class T>
    void test_specialization_form_4()
    {
        const T kConst = 2;

        dpl::complex<T> cv1(TestUtils::Complex::TestConstants<T>::kPartReal, TestUtils::Complex::TestConstants<T>::kPartImag);
        dpl::complex<T> cv2(TestUtils::Complex::TestConstants<T>::kPartReal, TestUtils::Complex::TestConstants<T>::kPartImag);

        cv1 /= kConst;

        EXPECT_TRUE_EE(errors, cv1.real() == cv2.real() / kConst, "Wrong effect in operator/=");
        EXPECT_TRUE_EE(errors, cv1.imag() == cv2.imag() / kConst, "Wrong effect in operator/=");
    }

private:

    TErrorsContainer& errors;
};

int
main()
{
    bool bSuccess = TestUtils::Complex::test_on_host<TextComplexArith>();

#if TEST_DPCPP_BACKEND_PRESENT
    try
    {
        sycl::queue deviceQueue{ TestUtils::default_selector };

        if (!TestUtils::Complex::test_in_kernel<TextComplexArith>(deviceQueue, TestUtils::kMaxKernelErrorsCount))
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
