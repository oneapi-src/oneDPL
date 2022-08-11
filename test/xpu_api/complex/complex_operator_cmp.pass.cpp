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
// class TextComplexCmp - testing of some std::complex operators
// 
// described https://en.cppreference.com/w/cpp/numeric/complex/operator_cmp
//
// (1)
//      template< class T >
//      bool
//      operator==(const complex<T>& lhs, const complex<T>& rhs);       (until C++14)
// 
//      template< class T >
//      constexpr bool
//      operator==(const complex<T>& lhs, const complex<T>& rhs);       (since C++14)
// 
// (2)
//      template< class T >
//      bool
//      operator==(const complex<T>& lhs, const T& rhs);                (until C++14)
// 
//      template< class T >
//      constexpr bool
//      operator==(const complex<T>& lhs, const T& rhs);                (since C++14)
// 
// (3)
//      template< class T >
//      bool
//      operator==(const T& lhs, const complex<T>& rhs);                (until C++14)
// 
//      template< class T >
//      constexpr bool                                                  (since C++14)
//      operator==(const T& lhs, const complex<T>& rhs);                (until C++20)
//
// (4)
//      template< class T >
//      bool
//      operator!=(const complex<T>& lhs, const complex<T>& rhs);       (until C++14)
// 
//      template< class T >
//      constexpr bool                                                  (since C++14)
//      operator!=(const complex<T>& lhs, const complex<T>& rhs);       (until C++20)
// 
// (5)
//      template< class T >
//      bool
//      operator!=(const complex<T>& lhs, const T& rhs);                (until C++14)
// 
//      template< class T >
//      constexpr bool                                                  (since C++14)
//      operator!=(const complex<T>& lhs, const T& rhs);                (until C++20)
// 
// (6)
//      template< class T >
//      bool
//      operator!=(const T& lhs, const complex<T>& rhs);                (until C++14)
// 
//      template< class T >
//      constexpr bool                                                  (since C++14)
//      operator!=(const T& lhs, const complex<T>& rhs);                (until C++20)
//
template <typename TErrorsContainer, typename IsSupportedDouble, typename IsSupportedLongDouble>
class TextComplexCmp
{
public:

    TextComplexCmp(TErrorsContainer& ee)
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
        test_primary_form_1_until_CPP20<T>();
        test_primary_form_2_until_CPP20<T>();
        test_primary_form_3_until_CPP20<T>();
        test_primary_form_4_until_CPP20<T>();
        test_primary_form_5_until_CPP20<T>();
        test_primary_form_6_until_CPP20<T>();
    }

    template <class T>
    void test_primary_form_1_until_CPP20()
    {
#if __cplusplus < 202002L
        TEST_KW_CONSTEXPR dpl::complex<T> cv1(TestUtils::Complex::TestConstants<T>::kPartReal, TestUtils::Complex::TestConstants<T>::kPartImag);

        TEST_KW_CONSTEXPR bool isEq1 = cv1 == cv1;
        EXPECT_TRUE_EE(errors, isEq1, "Wrong effect in std::complex<T> operator==(const complex<T>& lhs, const complex<T>& rhs)");
#endif
    }

    template <class T>
    void test_primary_form_2_until_CPP20()
    {
#if __cplusplus < 202002L
        TEST_KW_CONSTEXPR dpl::complex<T> cv1(TestUtils::Complex::TestConstants<T>::kPartReal);
        constexpr T kConst = TestUtils::Complex::TestConstants<T>::kPartReal;

        TEST_KW_CONSTEXPR bool isEq = cv1 == kConst;

        EXPECT_TRUE_EE(errors, isEq, "Wrong effect in std::complex<T> operator==(const complex<T>& lhs, const T& rhs)");
#endif
    }

    template <class T>
    void test_primary_form_3_until_CPP20()
    {
#if __cplusplus < 202002L
        constexpr T kConst = TestUtils::Complex::TestConstants<T>::kPartReal;
        TEST_KW_CONSTEXPR dpl::complex<T> cv1(TestUtils::Complex::TestConstants<T>::kPartReal);

        TEST_KW_CONSTEXPR bool isEq = kConst == cv1;

        EXPECT_TRUE_EE(errors, isEq, "Wrong effect in std::complex<T> operator==(const T& lhs, const complex<T>& rhs)");
#endif
    }

    template <class T>
    void test_primary_form_4_until_CPP20()
    {
#if __cplusplus < 202002L
        TEST_KW_CONSTEXPR dpl::complex<T> cv1(TestUtils::Complex::TestConstants<T>::kPartReal, TestUtils::Complex::TestConstants<T>::kPartImag);

        TEST_KW_CONSTEXPR bool isNotEq = cv1 != cv1;

        EXPECT_TRUE_EE(errors, !isNotEq, "Wrong effect in std::complex<T> operator!=(const complex<T>& lhs, const complex<T>& rhs)");
#endif
    }

    template <class T>
    void test_primary_form_5_until_CPP20()
    {
#if __cplusplus < 202002L
        TEST_KW_CONSTEXPR dpl::complex<T> cv1(TestUtils::Complex::TestConstants<T>::kPartReal, TestUtils::Complex::TestConstants<T>::kPartImag);
        constexpr T kConst = TestUtils::Complex::TestConstants<T>::kPartReal;

        TEST_KW_CONSTEXPR bool isNotEq = cv1 != kConst;

        EXPECT_TRUE_EE(errors, isNotEq, "Wrong effect in std::complex<T> operator!=(const complex<T>& lhs, const T& rhs)");
#endif
    }

    template <class T>
    void test_primary_form_6_until_CPP20()
    {
#if __cplusplus < 202002L
        constexpr T kConst = TestUtils::Complex::TestConstants<T>::kPartReal;
        TEST_KW_CONSTEXPR dpl::complex<T> cv1(TestUtils::Complex::TestConstants<T>::kPartReal, TestUtils::Complex::TestConstants<T>::kPartImag);

        TEST_KW_CONSTEXPR bool isNotEq = kConst != cv1;

        EXPECT_TRUE_EE(errors, isNotEq, "Wrong effect in std::complex<T> operator!=(const T& lhs, const complex<T>& rhs)");
#endif
    }

private:

    TErrorsContainer& errors;
};

int
main()
{
    bool bSuccess = TestUtils::Complex::test_on_host<TextComplexCmp>();

#if TEST_DPCPP_BACKEND_PRESENT
    try
    {
        sycl::queue deviceQueue{ TestUtils::default_selector };

        if (!TestUtils::Complex::test_in_kernel<TextComplexCmp>(deviceQueue, TestUtils::kMaxKernelErrorsCount))
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

#if __cplusplus < 202002L
    const int result = 1;
#else
    // Since C++20 test not implemented
    const int result = 0;
#endif

    return TestUtils::done(result);
}
