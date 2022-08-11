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
// class TextComplexArith3 - testing of some std::complex operators
// 
// described https://en.cppreference.com/w/cpp/numeric/complex/operator_arith3
//
// (1)
//      template< class T >
//      std::complex<T>
//      operator+(const std::complex<T>& lhs, const std::complex<T>& rhs);      (until C++ 20)
//
//      template <class T>
//      constexpr std::complex<T>
//      operator+(const std::complex<T>& lhs, const std::complex<T>& rhs);      (since C++ 20)
// 
// (2)
//
//      template <class T>
//      std::complex<T>
//      operator+(const std::complex<T>& lhs, const T& rhs);                    (until C++ 20)
//
//      template <class T>
//      constexpr std::complex<T>
//      operator+(const std::complex<T>& lhs, const T& rhs);                    (since C++ 20)
// 
// (3)
//
//      template <class T>
//      std::complex<T>
//      operator+(const T& lhs, const std::complex<T>& rhs);                    (until C++ 20)
//
//      template <class T>
//      constexpr std::complex<T>
//      operator+(const T& lhs, const std::complex<T>& rhs);                    (since C++ 20)
// 
// (4)
//
//      template <class T>
//      std::complex<T>
//      operator-(const std::complex<T>& lhs, const std::complex<T>& rhs);      (until C++ 20)
//
//      template <class T>
//      constexpr std::complex<T>
//      operator-(const std::complex<T>& lhs, const std::complex<T>& rhs);      (since C++ 20)
// 
// (5)
//
//      template <class T>
//      std::complex<T>
//      operator-(const std::complex<T>& lhs, const T& rhs);                    (until C++ 20)
//
//      template <class T>
//      constexpr std::complex<T>
//      operator-(const std::complex<T>& lhs, const T& rhs);                    (since C++ 20)
// 
// (6)
//
//      template <class T>
//      std::complex<T>
//      operator-(const T& lhs, const std::complex<T>& rhs);                    (until C++ 20)
//
//      template <class T>
//      constexpr std::complex<T>
//      operator-(const T& lhs, const std::complex<T>& rhs);                    (since C++ 20)
// 
// (7)
//
//      template <class T>
//      std::complex<T>
//      operator*(const std::complex<T>& lhs, const std::complex<T>& rhs);      (until C++ 20)
//
//      template <class T>
//      constexpr std::complex<T>
//      operator*(const std::complex<T>& lhs, const std::complex<T>& rhs);      (since C++ 20)
// 
// (8)
//
//      template <class T>
//      std::complex<T>
//      operator*(const std::complex<T>& lhs, const T& rhs);                    (until C++ 20)
//
//      template <class T>
//      constexpr std::complex<T>
//      operator*(const std::complex<T>& lhs, const T& rhs);                    (since C++ 20)
// 
// (9)
//
//      template <class T>
//      std::complex<T>
//      operator*(const T& lhs, const std::complex<T>& rhs);                    (until C++ 20)
//
//      template <class T>
//      constexpr std::complex<T>
//      operator*(const T& lhs, const std::complex<T>& rhs);                    (since C++ 20)
// 
// (10)
//
//      template <class T>
//      std::complex<T>
//      operator/(const std::complex<T>& lhs, const std::complex<T>& rhs);      (until C++ 20)
//
//      template <class T>
//      constexpr std::complex<T>
//      operator/(const std::complex<T>& lhs, const std::complex<T>& rhs);      (since C++ 20)
// 
// (11)
//
//      template <class T>
//      std::complex<T>
//      operator/(const std::complex<T>& lhs, const T& rhs);                    (until C++ 20)
//
//      template <class T>
//      constexpr std::complex<T>
//      operator/(const std::complex<T>& lhs, const T& rhs);                    (since C++ 20)
// 
// (12)
//
//      template <class T>
//      std::complex<T>
//      operator/(const T& lhs, const std::complex<T>& rhs);                    (until C++ 20)
//
//      template <class T>
//      constexpr std::complex<T>
//      operator/(const T& lhs, const std::complex<T>& rhs);                    (since C++ 20)
//
template <typename TErrorsContainer, typename IsSupportedDouble, typename IsSupportedLongDouble>
class TextComplexArith3
{
public:

    TextComplexArith3(TErrorsContainer& ee)
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
        test_primary_form_3<T>();
        test_primary_form_4<T>();
        test_primary_form_5<T>();
        test_primary_form_6<T>();
        test_primary_form_7<T>();
        test_primary_form_8<T>();
        test_primary_form_9<T>();
        test_primary_form_10<T>();
        test_primary_form_11<T>();
        test_primary_form_12<T>();
    }

    template <class T>
    void test_primary_form_1()
    {
        COMPLEX_TEST_CONSTEXPR dpl::complex<T> cv1(TestUtils::Complex::TestConstants<T>::kPartReal, TestUtils::Complex::TestConstants<T>::kPartImag);
        COMPLEX_TEST_CONSTEXPR dpl::complex<T> cv2(TestUtils::Complex::TestConstants<T>::kPartReal, TestUtils::Complex::TestConstants<T>::kPartImag);

        COMPLEX_TEST_CONSTEXPR auto cv3 = cv1 + cv2;
        static_assert(::std::is_same<typename ::std::decay<decltype(cv3.real())>::type, T>::value);
        static_assert(::std::is_same<typename ::std::decay<decltype(cv3.imag())>::type, T>::value);

        EXPECT_TRUE_EE(errors, cv3.real() == cv1.real() + cv2.real(), "Wrong effect in std::complex<T> operator+( const std::complex<T>& lhs, const std::complex<T>& rhs )");
        EXPECT_TRUE_EE(errors, cv3.imag() == cv1.imag() + cv2.imag(), "Wrong effect in std::complex<T> operator+( const std::complex<T>& lhs, const std::complex<T>& rhs )");
    }

    template <class T>
    void test_primary_form_2()
    {
        constexpr T kConst = TestUtils::Complex::TestConstants<T>::kPartReal;

        COMPLEX_TEST_CONSTEXPR dpl::complex<T> cv1(TestUtils::Complex::TestConstants<T>::kPartReal, TestUtils::Complex::TestConstants<T>::kPartImag);

        COMPLEX_TEST_CONSTEXPR auto cv2 = cv1 + kConst;
        static_assert(::std::is_same<typename ::std::decay<decltype(cv2.real())>::type, T>::value);
        static_assert(::std::is_same<typename ::std::decay<decltype(cv2.imag())>::type, T>::value);

        EXPECT_TRUE_EE(errors, cv2.real() == cv1.real() + kConst, "Wrong effect in std::complex<T> operator+( const std::complex<T>& lhs, const T& rhs )");
        EXPECT_TRUE_EE(errors, cv2.imag() == cv1.imag(), "Wrong effect in std::complex<T> operator+( const std::complex<T>& lhs, const T& rhs )");
    }

    template <class T>
    void test_primary_form_3()
    {
        constexpr T kConst = TestUtils::Complex::TestConstants<T>::kPartReal;

        COMPLEX_TEST_CONSTEXPR dpl::complex<T> cv1(TestUtils::Complex::TestConstants<T>::kPartReal, TestUtils::Complex::TestConstants<T>::kPartImag);

        COMPLEX_TEST_CONSTEXPR auto cv2 = kConst + cv1;
        static_assert(::std::is_same<typename ::std::decay<decltype(cv2.real())>::type, T>::value);
        static_assert(::std::is_same<typename ::std::decay<decltype(cv2.imag())>::type, T>::value);

        EXPECT_TRUE_EE(errors, cv2.real() == cv1.real() + kConst, "Wrong effect in std::complex<T> operator+( const T& lhs, const std::complex<T>& rhs )");
        EXPECT_TRUE_EE(errors, cv2.imag() == cv1.imag(), "Wrong effect in std::complex<T> operator+( const T& lhs, const std::complex<T>& rhs )");
    }

    template <class T>
    void test_primary_form_4()
    {
        COMPLEX_TEST_CONSTEXPR dpl::complex<T> cv1(TestUtils::Complex::TestConstants<T>::kPartReal, TestUtils::Complex::TestConstants<T>::kPartImag);
        COMPLEX_TEST_CONSTEXPR dpl::complex<T> cv2(TestUtils::Complex::TestConstants<T>::kPartReal, TestUtils::Complex::TestConstants<T>::kPartImag);

        COMPLEX_TEST_CONSTEXPR auto cv3 = cv1 - cv2;
        static_assert(::std::is_same<typename ::std::decay<decltype(cv3.real())>::type, T>::value);
        static_assert(::std::is_same<typename ::std::decay<decltype(cv3.imag())>::type, T>::value);

        EXPECT_TRUE_EE(errors, cv3.real() == cv1.real() - cv2.real(), "Wrong effect in operator-(const std::complex<T>& lhs, const std::complex<T>& rhs)");
        EXPECT_TRUE_EE(errors, cv3.imag() == cv1.imag() - cv2.imag(), "Wrong effect in operator-(const std::complex<T>& lhs, const std::complex<T>& rhs)");
    }

    template <class T>
    void test_primary_form_5()
    {
        constexpr T kConst = TestUtils::Complex::TestConstants<T>::kPartReal;

        COMPLEX_TEST_CONSTEXPR dpl::complex<T> cv1(TestUtils::Complex::TestConstants<T>::kPartReal, TestUtils::Complex::TestConstants<T>::kPartImag);

        COMPLEX_TEST_CONSTEXPR auto cv2 = cv1 - kConst;
        static_assert(::std::is_same<typename ::std::decay<decltype(cv2.real())>::type, T>::value);
        static_assert(::std::is_same<typename ::std::decay<decltype(cv2.imag())>::type, T>::value);

        EXPECT_TRUE_EE(errors, cv2.real() == cv1.real() - kConst, "Wrong effect in operator-(const std::complex<T>& lhs, const T& rhs)");
        EXPECT_TRUE_EE(errors, cv2.imag() == cv1.imag(), "Wrong effect in operator-(const std::complex<T>& lhs, const T& rhs)");
    }

    template <class T>
    void test_primary_form_6()
    {
        constexpr T kConst = TestUtils::Complex::TestConstants<T>::kPartReal;

        COMPLEX_TEST_CONSTEXPR dpl::complex<T> cv1(TestUtils::Complex::TestConstants<T>::kPartReal, TestUtils::Complex::TestConstants<T>::kPartImag);

        COMPLEX_TEST_CONSTEXPR auto cv2 = kConst - cv1;
        static_assert(::std::is_same<typename ::std::decay<decltype(cv2.real())>::type, T>::value);
        static_assert(::std::is_same<typename ::std::decay<decltype(cv2.imag())>::type, T>::value);

        EXPECT_TRUE_EE(errors, cv2.real() == kConst - cv1.real(), "Wrong effect in operator-(const std::complex<T>& lhs, const T& rhs)");
        EXPECT_TRUE_EE(errors, cv2.imag() == -1 * cv1.imag(), "Wrong effect in operator-(const std::complex<T>& lhs, const T& rhs)");
    }

    template <class T>
    void test_primary_form_7()
    {
        COMPLEX_TEST_CONSTEXPR dpl::complex<T> cv1(TestUtils::Complex::TestConstants<T>::kPartReal, TestUtils::Complex::TestConstants<T>::kPartImag);
        COMPLEX_TEST_CONSTEXPR dpl::complex<T> cv2(TestUtils::Complex::TestConstants<T>::kPartReal, TestUtils::Complex::TestConstants<T>::kPartImag);

        COMPLEX_TEST_CONSTEXPR auto cv3 = cv1 - cv2;
        static_assert(::std::is_same<typename ::std::decay<decltype(cv3.real())>::type, T>::value);
        static_assert(::std::is_same<typename ::std::decay<decltype(cv3.imag())>::type, T>::value);

        EXPECT_TRUE_EE(errors, cv3.real() == cv1.real() - cv2.real(), "Wrong effect in operator*(const std::complex<T>& lhs, const std::complex<T>& rhs)");
        EXPECT_TRUE_EE(errors, cv3.imag() == cv1.imag() - cv2.imag(), "Wrong effect in operator*(const std::complex<T>& lhs, const std::complex<T>& rhs)");
    }

    template <class T>
    void test_primary_form_8()
    {
        COMPLEX_TEST_CONSTEXPR dpl::complex<T> cv1(TestUtils::Complex::TestConstants<T>::kPartReal, TestUtils::Complex::TestConstants<T>::kPartImag);
        COMPLEX_TEST_CONSTEXPR dpl::complex<T> cv2(TestUtils::Complex::TestConstants<T>::kPartReal, TestUtils::Complex::TestConstants<T>::kPartImag);

        COMPLEX_TEST_CONSTEXPR auto cv3 = cv1 * cv2;
        static_assert(::std::is_same<typename ::std::decay<decltype(cv3.real())>::type, T>::value);
        static_assert(::std::is_same<typename ::std::decay<decltype(cv3.imag())>::type, T>::value);

        const auto a1 = cv1.real();
        const auto a2 = cv1.imag();
        const auto b1 = cv2.real();
        const auto b2 = cv2.imag();

        EXPECT_TRUE_EE(errors, cv3.real() == a1 * b1 - a2 * b2, "Wrong effect in operator*(const std::complex<T>& lhs, const T& rhs)");
        EXPECT_TRUE_EE(errors, cv3.imag() == a1 * b2 + a2 * b1, "Wrong effect in operator*(const std::complex<T>& lhs, const T& rhs)");
    }

    template <class T>
    void test_primary_form_9()
    {
        constexpr T kConst = TestUtils::Complex::TestConstants<T>::kPartReal;

        COMPLEX_TEST_CONSTEXPR dpl::complex<T> cv1(TestUtils::Complex::TestConstants<T>::kPartReal, TestUtils::Complex::TestConstants<T>::kPartImag);

        COMPLEX_TEST_CONSTEXPR auto cv2 = kConst * cv1;
        static_assert(::std::is_same<typename ::std::decay<decltype(cv2.real())>::type, T>::value);
        static_assert(::std::is_same<typename ::std::decay<decltype(cv2.imag())>::type, T>::value);

        const auto a1 = kConst;
        const auto b1 = cv1.real();
        const auto b2 = cv1.imag();

        EXPECT_TRUE_EE(errors, cv2.real() == a1 * b1, "Wrong effect in operator*(const T& lhs, const std::complex<T>& rhs)");
        EXPECT_TRUE_EE(errors, cv2.imag() == a1 * b2, "Wrong effect in operator*(const T& lhs, const std::complex<T>& rhs)");
    }

    template <class T>
    void test_primary_form_10()
    {
        COMPLEX_TEST_CONSTEXPR dpl::complex<T> cv1(TestUtils::Complex::TestConstants<T>::kPartReal, TestUtils::Complex::TestConstants<T>::kPartImag);
        COMPLEX_TEST_CONSTEXPR dpl::complex<T> cv2(TestUtils::Complex::TestConstants<T>::kPartReal, TestUtils::Complex::TestConstants<T>::kPartImag);

        COMPLEX_TEST_CONSTEXPR auto cv3 = cv1 / cv2;
        static_assert(::std::is_same<typename ::std::decay<decltype(cv3.real())>::type, T>::value);
        static_assert(::std::is_same<typename ::std::decay<decltype(cv3.imag())>::type, T>::value);

        COMPLEX_TEST_CONSTEXPR T expectedReal = (cv1.real() * cv2.real() + cv1.imag() * cv2.imag()) / (cv2.real() * cv2.real() + cv2.imag() * cv2.imag());
        COMPLEX_TEST_CONSTEXPR T expectedImag = (cv2.real() * cv1.imag() - cv1.real() * cv2.imag()) / (cv2.real() * cv2.real() + cv2.imag() * cv2.imag());

        EXPECT_TRUE_EE(errors, cv3.real() == expectedReal, "Wrong effect in operator/(const std::complex<T>& lhs, const std::complex<T>& rhs)");
        EXPECT_TRUE_EE(errors, cv3.imag() == expectedImag, "Wrong effect in operator/(const std::complex<T>& lhs, const std::complex<T>& rhs)");
    }

    template <class T>
    void test_primary_form_11()
    {
        constexpr T kConst = TestUtils::Complex::TestConstants<T>::kPartReal;

        COMPLEX_TEST_CONSTEXPR dpl::complex<T> cv1(TestUtils::Complex::TestConstants<T>::kPartReal, TestUtils::Complex::TestConstants<T>::kPartImag);

        COMPLEX_TEST_CONSTEXPR auto cv2 = cv1 / kConst;
        static_assert(::std::is_same<typename ::std::decay<decltype(cv2.real())>::type, T>::value);
        static_assert(::std::is_same<typename ::std::decay<decltype(cv2.imag())>::type, T>::value);

        EXPECT_TRUE_EE(errors, cv2.real() == cv1.real() / kConst, "Wrong effect in operator/(const std::complex<T>& lhs, const T& rhs)");
        EXPECT_TRUE_EE(errors, cv2.imag() == cv1.imag() / kConst, "Wrong effect in operator/(const std::complex<T>& lhs, const T& rhs)");
    }

    template <class T>
    void test_primary_form_12()
    {
        constexpr T kConst = TestUtils::Complex::TestConstants<T>::kPartReal;

        COMPLEX_TEST_CONSTEXPR dpl::complex<T> cv1(TestUtils::Complex::TestConstants<T>::kPartReal, TestUtils::Complex::TestConstants<T>::kPartImag);

        COMPLEX_TEST_CONSTEXPR auto cv2 = kConst / cv1;
        static_assert(::std::is_same<typename ::std::decay<decltype(cv2.real())>::type, T>::value);
        static_assert(::std::is_same<typename ::std::decay<decltype(cv2.imag())>::type, T>::value);

        COMPLEX_TEST_CONSTEXPR T expectedReal = (kConst * cv1.real()) / (cv1.real() * cv1.real() + cv1.imag() * cv1.imag());
        COMPLEX_TEST_CONSTEXPR T expectedImag = (-1 * kConst * cv1.imag()) / (cv1.real() * cv1.real() + cv1.imag() * cv1.imag());

        EXPECT_TRUE_EE(errors, cv2.real() == expectedReal, "Wrong effect in operator/(const T& lhs, const std::complex<T>& rhs)");
        EXPECT_TRUE_EE(errors, cv2.imag() == expectedImag, "Wrong effect in operator/(const T& lhs, const std::complex<T>& rhs)");
    }

private:

    TErrorsContainer& errors;
};

int
main()
{
    bool bSuccess = TestUtils::Complex::test_on_host<TextComplexArith3>();

#if TEST_DPCPP_BACKEND_PRESENT
    try
    {
        sycl::queue deviceQueue{ TestUtils::default_selector };

        if (!TestUtils::Complex::test_in_kernel<TextComplexArith3>(deviceQueue, TestUtils::kMaxKernelErrorsCount))
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
