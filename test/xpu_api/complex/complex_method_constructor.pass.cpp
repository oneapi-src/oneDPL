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
// class TextComplexMethodConstructor - testing of std::complex constructor
// 
// described https://en.cppreference.com/w/cpp/numeric/complex/complex
// 
//      primary template complex<T>
//          (1)
//              (until C++14)
//                  complex( const T& re = T(), const T& im = T() );
//              (since C++14)
//                  constexpr complex( const T& re = T(), const T& im = T() );
//          (2)
//              (until C++14)
//                  complex( const complex& other );
//              (since C++14)
//                  constexpr complex( const complex& other );
//          (3)
//              (until C++14)
//                  template< class X >
//                  complex(const complex<X>& other);
//              (since C++14)
//                  template< class X >
//                  constexpr complex(const complex<X>& other);
// 
//      specialization complex<float>
//          (1)
//              (until C++11)
//                  complex(float re = 0.0f, float im = 0.0f);
//              (since C++11)
//                  constexpr complex(float re = 0.0f, float im = 0.0f);
//          (3)
//              (until C++11)
//                  explicit complex(const complex<double>& other);
//                  explicit complex(const complex<long double>& other);
//              (since C++11)
//                  explicit constexpr complex(const complex<double>& other);
//                  explicit constexpr complex(const complex<long double>& other);
// 
//      specialization complex<double>
//          (1)
//              (until C++11)
//                  complex(double re = 0.0, double im = 0.0);
//              (since C++11)
//                  constexpr complex(double re = 0.0, double im = 0.0);
//          (3)
//              (until C++11)
//                  complex(const complex<float>& other);
//                  explicit complex(const complex<long double>& other);
//              (since C++11)
//                  constexpr complex(const complex<float>& other);
//                  explicit constexpr complex(const complex<long double>& other);
// 
//      specialization complex<long double>
//          (1)
//              (until C++11)
//                  complex(long double re = 0.0L, long double im = 0.0L);
//              (since C++11)
//                  constexpr complex(long double re = 0.0L, long double im = 0.0L);
//          (3)
//              (until C++11)
//                  complex(const complex<float>& other);
//                  complex(const complex<double>& other);
//              (since C++11)
//                  constexpr complex(const complex<float>& other);
//                  constexpr complex(const complex<double>& other);
//
// ATTENTION: all cases (until C++11) not covered by this test
//
template <typename TErrorsContainer, typename IsSupportedDouble, typename IsSupportedLongDouble>
class TextComplexMethodConstructor
{
public:

    TextComplexMethodConstructor(TErrorsContainer& ee)
        : errors(ee)
    {
    }

    template <typename RunOnHost>
    void run_test(RunOnHost /*runOnHost*/)
    {
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
    }

    template <class T>
    void test_primary_form_1()
    {
        test_primary_form_1_until_CPP14<T>();
        test_primary_form_1_since_CPP14<T>();
    }

    template <class T>
    void test_primary_form_2()
    {
        test_primary_form_2_until_CPP14<T>();
        test_primary_form_2_since_CPP14<T>();
    }

    template <class T>
    void test_primary_form_3()
    {
        test_primary_form_3_until_CPP14<T>();
        test_primary_form_3_since_CPP14<T>();
    }

    template <class T>
    void test_primary_form_1_until_CPP14()
    {
#if __cplusplus < 201402L
        // complex( const T& re = T(), const T& im = T() );

        dpl::complex<T> cv1;
        static_assert(::std::is_same<typename TestUtils::Complex::TestConstants<T>::DestComplexFieldType, decltype(cv1.real())>::value);
        static_assert(::std::is_same<typename TestUtils::Complex::TestConstants<T>::DestComplexFieldType, decltype(cv1.imag())>::value);
        EXPECT_TRUE_EE(errors, cv1.real() == TestUtils::Complex::TestConstants<T>::kZero, "Wrong effect in constructor #1");
        EXPECT_TRUE_EE(errors, cv1.imag() == TestUtils::Complex::TestConstants<T>::kZero, "Wrong effect in constructor #2");

        dpl::complex<T> cv2(TestUtils::Complex::TestConstants<T>::kPartReal);
        static_assert(::std::is_same<typename TestUtils::Complex::TestConstants<T>::DestComplexFieldType, decltype(cv2.real())>::value);
        static_assert(::std::is_same<typename TestUtils::Complex::TestConstants<T>::DestComplexFieldType, decltype(cv2.imag())>::value);
        EXPECT_TRUE_EE(errors, cv2.real() == TestUtils::Complex::TestConstants<T>::kPartReal, "Wrong effect in constructor #3");
        EXPECT_TRUE_EE(errors, cv2.imag() == TestUtils::Complex::TestConstants<T>::kZero, "Wrong effect in constructor #4");

        dpl::complex<T> cv3(TestUtils::Complex::TestConstants<T>::kPartReal, TestUtils::Complex::TestConstants<T>::kPartImag);
        static_assert(::std::is_same<typename TestUtils::Complex::TestConstants<T>::DestComplexFieldType, decltype(cv3.real())>::value);
        static_assert(::std::is_same<typename TestUtils::Complex::TestConstants<T>::DestComplexFieldType, decltype(cv3.imag())>::value);
        EXPECT_TRUE_EE(errors, cv3.real() == TestUtils::Complex::TestConstants<T>::kPartReal, "Wrong effect in constructor #5");
        EXPECT_TRUE_EE(errors, cv3.imag() == TestUtils::Complex::TestConstants<T>::kPartImag, "Wrong effect in constructor #6");
#endif
    }

    template <class T>
    void test_primary_form_1_since_CPP14()
    {
#if __cplusplus >= 201402L
        // constexpr complex( const T& re = T(), const T& im = T() );

        COMPLEX_TEST_CONSTEXPR dpl::complex<T> cv1;
        static_assert(::std::is_same<typename TestUtils::Complex::TestConstants<T>::DestComplexFieldType, decltype(cv1.real())>::value);
        static_assert(::std::is_same<typename TestUtils::Complex::TestConstants<T>::DestComplexFieldType, decltype(cv1.imag())>::value);
        EXPECT_TRUE_EE(errors, cv1.real() == TestUtils::Complex::TestConstants<T>::kZero, "Wrong effect in constructor #7");
        EXPECT_TRUE_EE(errors, cv1.imag() == TestUtils::Complex::TestConstants<T>::kZero, "Wrong effect in constructor #8");

        COMPLEX_TEST_CONSTEXPR dpl::complex<T> cv2(TestUtils::Complex::TestConstants<T>::kPartReal);
        static_assert(::std::is_same<typename TestUtils::Complex::TestConstants<T>::DestComplexFieldType, decltype(cv2.real())>::value);
        static_assert(::std::is_same<typename TestUtils::Complex::TestConstants<T>::DestComplexFieldType, decltype(cv2.imag())>::value);
        EXPECT_TRUE_EE(errors, cv2.real() == TestUtils::Complex::TestConstants<T>::kPartReal, "Wrong effect in constructor #9");
        EXPECT_TRUE_EE(errors, cv2.imag() == TestUtils::Complex::TestConstants<T>::kZero, "Wrong effect in constructor #10");

        COMPLEX_TEST_CONSTEXPR dpl::complex<T> cv3(TestUtils::Complex::TestConstants<T>::kPartReal, TestUtils::Complex::TestConstants<T>::kPartImag);
        static_assert(::std::is_same<typename TestUtils::Complex::TestConstants<T>::DestComplexFieldType, decltype(cv3.real())>::value);
        static_assert(::std::is_same<typename TestUtils::Complex::TestConstants<T>::DestComplexFieldType, decltype(cv3.imag())>::value);
        EXPECT_TRUE_EE(errors, cv3.real() == TestUtils::Complex::TestConstants<T>::kPartReal, "Wrong effect in constructor #11");
        EXPECT_TRUE_EE(errors, cv3.imag() == TestUtils::Complex::TestConstants<T>::kPartImag, "Wrong effect in constructor #12");
#endif
    }

    template <class T>
    void test_primary_form_2_until_CPP14()
    {
#if __cplusplus < 201402L
        // complex( const complex& other );
        const dpl::complex<T> cv_src(TestUtils::Complex::TestConstants<T>::kPartReal, TestUtils::Complex::TestConstants<T>::kPartImag);
        const dpl::complex<T> cv_copy(cv_src);
        static_assert(::std::is_same<typename TestUtils::Complex::TestConstants<T>::DestComplexFieldType, decltype(cv_copy.real())>::value);
        static_assert(::std::is_same<typename TestUtils::Complex::TestConstants<T>::DestComplexFieldType, decltype(cv_copy.imag())>::value);
        EXPECT_TRUE_EE(errors, cv_copy.real() == TestUtils::Complex::TestConstants<T>::kPartReal, "Wrong effect in constructor #13");
        EXPECT_TRUE_EE(errors, cv_copy.imag() == TestUtils::Complex::TestConstants<T>::kPartImag, "Wrong effect in constructor #14");
#endif
    }

    template <class T>
    void test_primary_form_2_since_CPP14()
    {
#if __cplusplus >= 201402L
        // constexpr complex( const complex& other );
        const dpl::complex<T> cv_src(TestUtils::Complex::TestConstants<T>::kPartReal, TestUtils::Complex::TestConstants<T>::kPartImag);
        COMPLEX_TEST_CONSTEXPR dpl::complex<T> cv_copy(cv_src);
        static_assert(::std::is_same<typename TestUtils::Complex::TestConstants<T>::DestComplexFieldType, decltype(cv_copy.real())>::value);
        static_assert(::std::is_same<typename TestUtils::Complex::TestConstants<T>::DestComplexFieldType, decltype(cv_copy.imag())>::value);
        EXPECT_TRUE_EE(errors, cv_copy.real() == TestUtils::Complex::TestConstants<T>::kPartReal, "Wrong effect in constructor #15");
        EXPECT_TRUE_EE(errors, cv_copy.imag() == TestUtils::Complex::TestConstants<T>::kPartImag, "Wrong effect in constructor #16");
#endif
    }

    template <class T>
    void test_primary_form_3_until_CPP14()
    {
#if __cplusplus < 201402L
        // template< class X >
        // complex(const complex<X>& other);
        const dpl::complex<T> cv_src(TestUtils::Complex::TestConstants<T>::kPartReal, TestUtils::Complex::TestConstants<T>::kPartImag);
        const dpl::complex<T> cv_copy(cv_src);
        EXPECT_TRUE_EE(errors, cv_copy.real() == TestUtils::Complex::TestConstants<T>::kPartReal, "Wrong effect in constructor #17");
        EXPECT_TRUE_EE(errors, cv_copy.imag() == TestUtils::Complex::TestConstants<T>::kPartImag, "Wrong effect in constructor #18");
#endif
    }

    template <class T>
    void test_primary_form_3_since_CPP14()
    {
#if __cplusplus >= 201402L
        // template< class X >
        // constexpr complex(const complex<X>& other);
        const dpl::complex<T> cv_src(TestUtils::Complex::TestConstants<T>::kPartReal, TestUtils::Complex::TestConstants<T>::kPartImag);
        COMPLEX_TEST_CONSTEXPR dpl::complex<T> cv_copy(cv_src);
        static_assert(::std::is_same<typename TestUtils::Complex::TestConstants<T>::DestComplexFieldType, decltype(cv_copy.real())>::value);
        static_assert(::std::is_same<typename TestUtils::Complex::TestConstants<T>::DestComplexFieldType, decltype(cv_copy.imag())>::value);
        EXPECT_TRUE_EE(errors, cv_copy.real() == TestUtils::Complex::TestConstants<T>::kPartReal, "Wrong effect in constructor #19");
        EXPECT_TRUE_EE(errors, cv_copy.imag() == TestUtils::Complex::TestConstants<T>::kPartImag, "Wrong effect in constructor #20");
#endif
    }

    template <class T>
    void test_specialization()
    {
        test_specialization_form1<T>();
        test_specialization_form3<T>();
    }

    template <class T>
    void test_specialization_form1()
    {
        // constexpr complex(float re = 0.0f, float im = 0.0f);
        // constexpr complex(double re = 0.0, double im = 0.0);
        // constexpr complex(long double re = 0.0L, long double im = 0.0L);

        COMPLEX_TEST_CONSTEXPR dpl::complex<T> cv1;
        static_assert(::std::is_same<typename TestUtils::Complex::TestConstants<T>::DestComplexFieldType, decltype(cv1.real())>::value);
        static_assert(::std::is_same<typename TestUtils::Complex::TestConstants<T>::DestComplexFieldType, decltype(cv1.imag())>::value);
        EXPECT_TRUE_EE(errors, cv1.real() == TestUtils::Complex::TestConstants<T>::kZero, "Wrong effect in constructor #7");
        EXPECT_TRUE_EE(errors, cv1.imag() == TestUtils::Complex::TestConstants<T>::kZero, "Wrong effect in constructor #8");

        COMPLEX_TEST_CONSTEXPR dpl::complex<T> cv2(TestUtils::Complex::TestConstants<T>::kPartReal);
        static_assert(::std::is_same<typename TestUtils::Complex::TestConstants<T>::DestComplexFieldType, decltype(cv2.real())>::value);
        static_assert(::std::is_same<typename TestUtils::Complex::TestConstants<T>::DestComplexFieldType, decltype(cv2.imag())>::value);
        EXPECT_TRUE_EE(errors, cv2.real() == TestUtils::Complex::TestConstants<T>::kPartReal, "Wrong effect in constructor #9");
        EXPECT_TRUE_EE(errors, cv2.imag() == TestUtils::Complex::TestConstants<T>::kZero, "Wrong effect in constructor #10");

        COMPLEX_TEST_CONSTEXPR dpl::complex<T> cv3(TestUtils::Complex::TestConstants<T>::kPartReal, TestUtils::Complex::TestConstants<T>::kPartImag);
        static_assert(::std::is_same<typename TestUtils::Complex::TestConstants<T>::DestComplexFieldType, decltype(cv3.real())>::value);
        static_assert(::std::is_same<typename TestUtils::Complex::TestConstants<T>::DestComplexFieldType, decltype(cv3.imag())>::value);
        EXPECT_TRUE_EE(errors, cv3.real() == TestUtils::Complex::TestConstants<T>::kPartReal, "Wrong effect in constructor #11");
        EXPECT_TRUE_EE(errors, cv3.imag() == TestUtils::Complex::TestConstants<T>::kPartImag, "Wrong effect in constructor #12");
    }

    template <class T>
    void test_specialization_form3()
    {
        // explicit constexpr complex(const complex<double>& other);
        // explicit constexpr complex(const complex<long double>& other);

        // constexpr complex(const complex<float>& other);
        // explicit constexpr complex(const complex<long double>& other);

        // constexpr complex(const complex<float>& other);
        // constexpr complex(const complex<double>& other);

        // TODO required to implement?
    }

private:

    TErrorsContainer& errors;
};

int
main()
{
    bool bSuccess = TestUtils::Complex::test_on_host<TextComplexMethodConstructor>();

#if TEST_DPCPP_BACKEND_PRESENT
    try
    {
        sycl::queue deviceQueue{ TestUtils::default_selector };

        if (!TestUtils::Complex::test_in_kernel<TextComplexMethodConstructor>(deviceQueue, TestUtils::kMaxKernelErrorsCount))
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
