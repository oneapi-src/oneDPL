//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// XFAIL: LIBCXX-AIX-FIXME

// <cmath>

#include "support/test_complex.h"

#include <oneapi/dpl/cmath>
#include <oneapi/dpl/limits>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

// convertible to int/float/double/etc
template <class T, int N=0>
struct Value {
    operator T () { return T(N); }
};

// See PR21083
// Ambiguous is a user-defined type that defines its own overloads of cmath
// functions. When the std overloads are candidates too (by using or adl),
// they should not interfere.
struct Ambiguous : std::true_type { // ADL
    operator float () { return 0.f; }
    operator double () { return 0.; }
};
Ambiguous abs(Ambiguous){ return Ambiguous(); }
Ambiguous ceil(Ambiguous){ return Ambiguous(); }
Ambiguous exp(Ambiguous){ return Ambiguous(); }
Ambiguous fabs(Ambiguous){ return Ambiguous(); }
Ambiguous floor(Ambiguous){ return Ambiguous(); }
Ambiguous isgreater(Ambiguous, Ambiguous){ return Ambiguous(); }
Ambiguous isgreaterequal(Ambiguous, Ambiguous){ return Ambiguous(); }
Ambiguous isless(Ambiguous, Ambiguous){ return Ambiguous(); }
Ambiguous islessequal(Ambiguous, Ambiguous){ return Ambiguous(); }
Ambiguous copysign(Ambiguous, Ambiguous){ return Ambiguous(); }
Ambiguous fmax(Ambiguous, Ambiguous){ return Ambiguous(); }
Ambiguous fmin(Ambiguous, Ambiguous){ return Ambiguous(); }
Ambiguous round(Ambiguous){ return Ambiguous(); }
Ambiguous trunc(Ambiguous){ return Ambiguous(); }

template <class T, class = decltype(std::abs(std::declval<T>()))>
std::true_type has_abs_imp(int);
template <class T>
std::false_type has_abs_imp(...);

template <class T>
struct has_abs : decltype(has_abs_imp<T>(0)) {};

void test_abs()
{
    // See also "abs.pass.cpp"

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wabsolute-value"
#endif
    static_assert((std::is_same<decltype(dpl::abs((float)0)), float>::value), "");
    if constexpr (HasDoubleSupportInRuntime::value)
    {
        static_assert((std::is_same<decltype(dpl::abs((double)0)), double>::value), "");
        if constexpr (HasLongDoubleSupportInCompiletime::value)
            static_assert((std::is_same<decltype(dpl::abs((long double)0)), long double>::value), "");
    }
    static_assert((std::is_same<decltype(dpl::abs((int)0)), int>::value), "");
    static_assert((std::is_same<decltype(dpl::abs((long)0)), long>::value), "");
    static_assert((std::is_same<decltype(dpl::abs((long long)0)), long long>::value), "");
    static_assert((std::is_same<decltype(dpl::abs((unsigned char)0)), int>::value), "");
    static_assert((std::is_same<decltype(dpl::abs((unsigned short)0)), int>::value), "");
    static_assert((std::is_same<decltype(dpl::abs((signed char)0)), int>::value), "");
    static_assert((std::is_same<decltype(dpl::abs((short)0)), int>::value), "");
    static_assert((std::is_same<decltype(dpl::abs((unsigned char)0)), int>::value), "");
    static_assert((std::is_same<decltype(dpl::abs((char)0)), int>::value), "");
    static_assert((std::is_same<decltype(abs(Ambiguous())), Ambiguous>::value), "");

    static_assert(!has_abs<unsigned>::value, "");
    static_assert(!has_abs<unsigned long>::value, "");
    static_assert(!has_abs<unsigned long long>::value, "");
    static_assert(!has_abs<size_t>::value, "");
#ifdef __clang__
#pragma clang diagnostic pop
#endif

    assert(dpl::abs(-1.) == 1);
}

void test_ceil()
{
    static_assert((std::is_same<decltype(dpl::ceil((float)0)), float>::value), "");
    if constexpr (HasDoubleSupportInRuntime::value)
    {
        static_assert((std::is_same<decltype(dpl::ceil((bool)0)), double>::value), "");
        static_assert((std::is_same<decltype(dpl::ceil((unsigned short)0)), double>::value), "");
        static_assert((std::is_same<decltype(dpl::ceil((int)0)), double>::value), "");
        static_assert((std::is_same<decltype(dpl::ceil((unsigned int)0)), double>::value), "");
        static_assert((std::is_same<decltype(dpl::ceil((long)0)), double>::value), "");
        static_assert((std::is_same<decltype(dpl::ceil((unsigned long)0)), double>::value), "");
        static_assert((std::is_same<decltype(dpl::ceil((long long)0)), double>::value), "");
        static_assert((std::is_same<decltype(dpl::ceil((unsigned long long)0)), double>::value), "");
        static_assert((std::is_same<decltype(dpl::ceil((double)0)), double>::value), "");
        if constexpr (HasLongDoubleSupportInCompiletime::value)
            static_assert((std::is_same<decltype(dpl::ceil((long double)0)), long double>::value), "");
    }
    static_assert((std::is_same<decltype(dpl::ceilf(0)), float>::value), "");
    if constexpr (HasLongDoubleSupportInCompiletime::value)
        static_assert((std::is_same<decltype(dpl::ceill(0)), long double>::value), "");
    static_assert((std::is_same<decltype(ceil(Ambiguous())), Ambiguous>::value), "");
    assert(dpl::ceil(0) == 0);
}

void test_exp()
{
    static_assert((std::is_same<decltype(dpl::exp((float)0)), float>::value), "");
    if constexpr (HasDoubleSupportInRuntime::value)
    {
        static_assert((std::is_same<decltype(dpl::exp((bool)0)), double>::value), "");
        static_assert((std::is_same<decltype(dpl::exp((unsigned short)0)), double>::value), "");
        static_assert((std::is_same<decltype(dpl::exp((int)0)), double>::value), "");
        static_assert((std::is_same<decltype(dpl::exp((unsigned int)0)), double>::value), "");
        static_assert((std::is_same<decltype(dpl::exp((long)0)), double>::value), "");
        static_assert((std::is_same<decltype(dpl::exp((unsigned long)0)), double>::value), "");
        static_assert((std::is_same<decltype(dpl::exp((long long)0)), double>::value), "");
        static_assert((std::is_same<decltype(dpl::exp((unsigned long long)0)), double>::value), "");
        static_assert((std::is_same<decltype(dpl::exp((double)0)), double>::value), "");
        if constexpr (HasLongDoubleSupportInCompiletime::value)
            static_assert((std::is_same<decltype(dpl::exp((long double)0)), long double>::value), "");
    }
    static_assert((std::is_same<decltype(dpl::expf(0)), float>::value), "");
    if constexpr (HasLongDoubleSupportInCompiletime::value)
        static_assert((std::is_same<decltype(dpl::expl(0)), long double>::value), "");
    static_assert((std::is_same<decltype(exp(Ambiguous())), Ambiguous>::value), "");
    assert(dpl::exp(0) == 1);
}

void test_fabs()
{
    static_assert((std::is_same<decltype(dpl::fabs((float)0)), float>::value), "");
    if constexpr (HasDoubleSupportInRuntime::value)
    {
        static_assert((std::is_same<decltype(dpl::fabs((bool)0)), double>::value), "");
        static_assert((std::is_same<decltype(dpl::fabs((unsigned short)0)), double>::value), "");
        static_assert((std::is_same<decltype(dpl::fabs((int)0)), double>::value), "");
        static_assert((std::is_same<decltype(dpl::fabs((unsigned int)0)), double>::value), "");
        static_assert((std::is_same<decltype(dpl::fabs((long)0)), double>::value), "");
        static_assert((std::is_same<decltype(dpl::fabs((unsigned long)0)), double>::value), "");
        static_assert((std::is_same<decltype(dpl::fabs((long long)0)), double>::value), "");
        static_assert((std::is_same<decltype(dpl::fabs((unsigned long long)0)), double>::value), "");
        static_assert((std::is_same<decltype(dpl::fabs((double)0)), double>::value), "");
        if constexpr (HasLongDoubleSupportInCompiletime::value)
            static_assert((std::is_same<decltype(dpl::fabs((long double)0)), long double>::value), "");
    }
    static_assert((std::is_same<decltype(dpl::fabsf(0.0f)), float>::value), "");
    if constexpr (HasLongDoubleSupportInCompiletime::value)
        static_assert((std::is_same<decltype(dpl::fabsl(0.0L)), long double>::value), "");
    static_assert((std::is_same<decltype(fabs(Ambiguous())), Ambiguous>::value), "");
    assert(dpl::fabs(-1) == 1);
}

void test_floor()
{
    static_assert((std::is_same<decltype(dpl::floor((float)0)), float>::value), "");
    if constexpr (HasDoubleSupportInRuntime::value)
    {
        static_assert((std::is_same<decltype(dpl::floor((bool)0)), double>::value), "");
        static_assert((std::is_same<decltype(dpl::floor((unsigned short)0)), double>::value), "");
        static_assert((std::is_same<decltype(dpl::floor((int)0)), double>::value), "");
        static_assert((std::is_same<decltype(dpl::floor((unsigned int)0)), double>::value), "");
        static_assert((std::is_same<decltype(dpl::floor((long)0)), double>::value), "");
        static_assert((std::is_same<decltype(dpl::floor((unsigned long)0)), double>::value), "");
        static_assert((std::is_same<decltype(dpl::floor((long long)0)), double>::value), "");
        static_assert((std::is_same<decltype(dpl::floor((unsigned long long)0)), double>::value), "");
        static_assert((std::is_same<decltype(dpl::floor((double)0)), double>::value), "");
        if constexpr (HasLongDoubleSupportInCompiletime::value)
            static_assert((std::is_same<decltype(dpl::floor((long double)0)), long double>::value), "");
    }
    static_assert((std::is_same<decltype(dpl::floorf(0)), float>::value), "");
    if constexpr (HasLongDoubleSupportInCompiletime::value)
        static_assert((std::is_same<decltype(dpl::floorl(0)), long double>::value), "");        // TODO is it required?
    static_assert((std::is_same<decltype(floor(Ambiguous())), Ambiguous>::value), "");
    assert(dpl::floor(1) == 1);
}

void test_isgreater()
{
#ifdef isgreater
#error isgreater defined
#endif
    static_assert((std::is_same<decltype(dpl::isgreater((float)0, (float)0)), bool>::value), "");
    if constexpr (HasDoubleSupportInRuntime::value)
    {
        static_assert((std::is_same<decltype(dpl::isgreater((float)0, (double)0)), bool>::value), "");
        if constexpr (HasLongDoubleSupportInCompiletime::value)
            static_assert((std::is_same<decltype(dpl::isgreater((float)0, (long double)0)), bool>::value), "");
        static_assert((std::is_same<decltype(dpl::isgreater((double)0, (float)0)), bool>::value), "");
        static_assert((std::is_same<decltype(dpl::isgreater((double)0, (double)0)), bool>::value), "");
        static_assert((std::is_same<decltype(dpl::isgreater(0, (double)0)), bool>::value), "");
        if constexpr (HasLongDoubleSupportInCompiletime::value)
        {
            static_assert((std::is_same<decltype(dpl::isgreater((double)0, (long double)0)), bool>::value), "");
            static_assert((std::is_same<decltype(dpl::isgreater((long double)0, (float)0)), bool>::value), "");
            static_assert((std::is_same<decltype(dpl::isgreater((long double)0, (double)0)), bool>::value), "");
            static_assert((std::is_same<decltype(dpl::isgreater((long double)0, (long double)0)), bool>::value), "");
        }
    }
    static_assert((std::is_same<decltype(isgreater(Ambiguous(), Ambiguous())), Ambiguous>::value), "");
    assert(dpl::isgreater(-1.0, 0.F) == false);
}

void test_isgreaterequal()
{
#ifdef isgreaterequal
#error isgreaterequal defined
#endif
    static_assert((std::is_same<decltype(dpl::isgreaterequal((float)0, (float)0)), bool>::value), "");
    if constexpr (HasDoubleSupportInRuntime::value)
    {
        static_assert((std::is_same<decltype(dpl::isgreaterequal((float)0, (double)0)), bool>::value), "");
        if constexpr (HasLongDoubleSupportInCompiletime::value)
            static_assert((std::is_same<decltype(dpl::isgreaterequal((float)0, (long double)0)), bool>::value), "");
        static_assert((std::is_same<decltype(dpl::isgreaterequal((double)0, (float)0)), bool>::value), "");
        static_assert((std::is_same<decltype(dpl::isgreaterequal((double)0, (double)0)), bool>::value), "");
        static_assert((std::is_same<decltype(dpl::isgreaterequal(0, (double)0)), bool>::value), "");
        if constexpr (HasLongDoubleSupportInCompiletime::value)
        {
            static_assert((std::is_same<decltype(dpl::isgreaterequal((double)0, (long double)0)), bool>::value), "");
            static_assert((std::is_same<decltype(dpl::isgreaterequal((long double)0, (float)0)), bool>::value), "");
            static_assert((std::is_same<decltype(dpl::isgreaterequal((long double)0, (double)0)), bool>::value), "");
            static_assert((std::is_same<decltype(dpl::isgreaterequal((long double)0, (long double)0)), bool>::value), "");
        }
    }
    static_assert((std::is_same<decltype(isgreaterequal(Ambiguous(), Ambiguous())), Ambiguous>::value), "");
    assert(dpl::isgreaterequal(-1.0, 0.F) == false);
}

void test_isinf()
{
#ifdef isinf
#error isinf defined
#endif
    static_assert((std::is_same<decltype(dpl::isinf((float)0)), bool>::value), "");

    if constexpr (HasDoubleSupportInRuntime::value)
    {
        typedef decltype(dpl::isinf((double)0)) DoubleRetType;
#if !defined(__linux__) || defined(__clang__)
        static_assert((std::is_same<DoubleRetType, bool>::value), "");
#else
        // GLIBC < 2.23 defines 'isinf(double)' with a return type of 'int' in
        // all C++ dialects. The test should tolerate this when libc++ can't work
        // around it.
        // See: https://sourceware.org/bugzilla/show_bug.cgi?id=19439
        static_assert((std::is_same<DoubleRetType, bool>::value
                    || std::is_same<DoubleRetType, int>::value), "");
#endif
    }

    static_assert((std::is_same<decltype(dpl::isinf(0)), bool>::value), "");
    if constexpr (HasLongDoubleSupportInCompiletime::value)
        static_assert((std::is_same<decltype(dpl::isinf((long double)0)), bool>::value), "");
    assert(dpl::isinf(-1.0) == false);
    assert(dpl::isinf(0) == false);
    assert(dpl::isinf(1) == false);
    assert(dpl::isinf(-1) == false);
    assert(dpl::isinf(dpl::numeric_limits<int>::max()) == false);
    assert(dpl::isinf(dpl::numeric_limits<int>::min()) == false);
}

void test_isless()
{
#ifdef isless
#error isless defined
#endif
    static_assert((std::is_same<decltype(dpl::isless((float)0, (float)0)), bool>::value), "");
    if constexpr (HasDoubleSupportInRuntime::value)
    {
        static_assert((std::is_same<decltype(dpl::isless((float)0, (double)0)), bool>::value), "");
        if constexpr (HasLongDoubleSupportInCompiletime::value)
            static_assert((std::is_same<decltype(dpl::isless((float)0, (long double)0)), bool>::value), "");
        static_assert((std::is_same<decltype(dpl::isless((double)0, (float)0)), bool>::value), "");
        static_assert((std::is_same<decltype(dpl::isless((double)0, (double)0)), bool>::value), "");
        static_assert((std::is_same<decltype(dpl::isless(0, (double)0)), bool>::value), "");
        if constexpr (HasLongDoubleSupportInCompiletime::value)
        {
            static_assert((std::is_same<decltype(dpl::isless((double)0, (long double)0)), bool>::value), "");
            static_assert((std::is_same<decltype(dpl::isless((long double)0, (float)0)), bool>::value), "");
            static_assert((std::is_same<decltype(dpl::isless((long double)0, (double)0)), bool>::value), "");
            static_assert((std::is_same<decltype(dpl::isless((long double)0, (long double)0)), bool>::value), "");
        }
    }
    static_assert((std::is_same<decltype(isless(Ambiguous(), Ambiguous())), Ambiguous>::value), "");
    assert(dpl::isless(-1.0, 0.F) == true);
}

void test_islessequal()
{
#ifdef islessequal
#error islessequal defined
#endif
    static_assert((std::is_same<decltype(std::islessequal((float)0, (float)0)), bool>::value), "");
    static_assert((std::is_same<decltype(std::islessequal((float)0, (double)0)), bool>::value), "");
    static_assert((std::is_same<decltype(std::islessequal((float)0, (long double)0)), bool>::value), "");
    static_assert((std::is_same<decltype(std::islessequal((double)0, (float)0)), bool>::value), "");
    static_assert((std::is_same<decltype(std::islessequal((double)0, (double)0)), bool>::value), "");
    static_assert((std::is_same<decltype(std::islessequal(0, (double)0)), bool>::value), "");
    static_assert((std::is_same<decltype(std::islessequal((double)0, (long double)0)), bool>::value), "");
    static_assert((std::is_same<decltype(std::islessequal((long double)0, (float)0)), bool>::value), "");
    static_assert((std::is_same<decltype(std::islessequal((long double)0, (double)0)), bool>::value), "");
    static_assert((std::is_same<decltype(std::islessequal((long double)0, (long double)0)), bool>::value), "");
    static_assert((std::is_same<decltype(islessequal(Ambiguous(), Ambiguous())), Ambiguous>::value), "");
    assert(std::islessequal(-1.0, 0.F) == true);
}

void test_isnan()
{
#ifdef isnan
#error isnan defined
#endif
    static_assert((std::is_same<decltype(std::isnan((float)0)), bool>::value), "");

    typedef decltype(std::isnan((double)0)) DoubleRetType;
#if !defined(__linux__) || defined(__clang__)
    static_assert((std::is_same<DoubleRetType, bool>::value), "");
#else
    // GLIBC < 2.23 defines 'isinf(double)' with a return type of 'int' in
    // all C++ dialects. The test should tolerate this when libc++ can't work
    // around it.
    // See: https://sourceware.org/bugzilla/show_bug.cgi?id=19439
    static_assert((std::is_same<DoubleRetType, bool>::value
                || std::is_same<DoubleRetType, int>::value), "");
#endif

    static_assert((std::is_same<decltype(std::isnan(0)), bool>::value), "");
    static_assert((std::is_same<decltype(std::isnan((long double)0)), bool>::value), "");
    assert(std::isnan(-1.0) == false);
    assert(std::isnan(0) == false);
    assert(std::isnan(1) == false);
    assert(std::isnan(-1) == false);
    assert(std::isnan(std::numeric_limits<int>::max()) == false);
    assert(std::isnan(std::numeric_limits<int>::min()) == false);
}

void test_isunordered()
{
#ifdef isunordered
#error isunordered defined
#endif
    static_assert((std::is_same<decltype(std::isunordered((float)0, (float)0)), bool>::value), "");
    static_assert((std::is_same<decltype(std::isunordered((float)0, (double)0)), bool>::value), "");
    static_assert((std::is_same<decltype(std::isunordered((float)0, (long double)0)), bool>::value), "");
    static_assert((std::is_same<decltype(std::isunordered((double)0, (float)0)), bool>::value), "");
    static_assert((std::is_same<decltype(std::isunordered((double)0, (double)0)), bool>::value), "");
    static_assert((std::is_same<decltype(std::isunordered(0, (double)0)), bool>::value), "");
    static_assert((std::is_same<decltype(std::isunordered((double)0, (long double)0)), bool>::value), "");
    static_assert((std::is_same<decltype(std::isunordered((long double)0, (float)0)), bool>::value), "");
    static_assert((std::is_same<decltype(std::isunordered((long double)0, (double)0)), bool>::value), "");
    static_assert((std::is_same<decltype(std::isunordered((long double)0, (long double)0)), bool>::value), "");
    static_assert((std::is_same<decltype(isunordered(Ambiguous(), Ambiguous())), Ambiguous>::value), "");
    assert(std::isunordered(-1.0, 0.F) == false);
}

void test_copysign()
{
    static_assert((std::is_same<decltype(std::copysign((float)0, (float)0)), float>::value), "");
    static_assert((std::is_same<decltype(std::copysign((bool)0, (float)0)), double>::value), "");
    static_assert((std::is_same<decltype(std::copysign((unsigned short)0, (double)0)), double>::value), "");
    static_assert((std::is_same<decltype(std::copysign((int)0, (long double)0)), long double>::value), "");
    static_assert((std::is_same<decltype(std::copysign((float)0, (unsigned int)0)), double>::value), "");
    static_assert((std::is_same<decltype(std::copysign((double)0, (long)0)), double>::value), "");
    static_assert((std::is_same<decltype(std::copysign((long double)0, (unsigned long)0)), long double>::value), "");
    static_assert((std::is_same<decltype(std::copysign((int)0, (long long)0)), double>::value), "");
    static_assert((std::is_same<decltype(std::copysign((int)0, (unsigned long long)0)), double>::value), "");
    static_assert((std::is_same<decltype(std::copysign((double)0, (double)0)), double>::value), "");
    static_assert((std::is_same<decltype(std::copysign((long double)0, (long double)0)), long double>::value), "");
    static_assert((std::is_same<decltype(std::copysign((float)0, (double)0)), double>::value), "");
    static_assert((std::is_same<decltype(std::copysign((float)0, (long double)0)), long double>::value), "");
    static_assert((std::is_same<decltype(std::copysign((double)0, (long double)0)), long double>::value), "");
    static_assert((std::is_same<decltype(std::copysignf(0,0)), float>::value), "");
    static_assert((std::is_same<decltype(std::copysignl(0,0)), long double>::value), "");
    static_assert((std::is_same<decltype(std::copysign((int)0, (int)0)), double>::value), "");
    static_assert((std::is_same<decltype(copysign(Ambiguous(), Ambiguous())), Ambiguous>::value), "");
    assert(std::copysign(1,1) == 1);
}

void test_fmax()
{
    static_assert((std::is_same<decltype(std::fmax((float)0, (float)0)), float>::value), "");
    static_assert((std::is_same<decltype(std::fmax((bool)0, (float)0)), double>::value), "");
    static_assert((std::is_same<decltype(std::fmax((unsigned short)0, (double)0)), double>::value), "");
    static_assert((std::is_same<decltype(std::fmax((int)0, (long double)0)), long double>::value), "");
    static_assert((std::is_same<decltype(std::fmax((float)0, (unsigned int)0)), double>::value), "");
    static_assert((std::is_same<decltype(std::fmax((double)0, (long)0)), double>::value), "");
    static_assert((std::is_same<decltype(std::fmax((long double)0, (unsigned long)0)), long double>::value), "");
    static_assert((std::is_same<decltype(std::fmax((int)0, (long long)0)), double>::value), "");
    static_assert((std::is_same<decltype(std::fmax((int)0, (unsigned long long)0)), double>::value), "");
    static_assert((std::is_same<decltype(std::fmax((double)0, (double)0)), double>::value), "");
    static_assert((std::is_same<decltype(std::fmax((long double)0, (long double)0)), long double>::value), "");
    static_assert((std::is_same<decltype(std::fmax((float)0, (double)0)), double>::value), "");
    static_assert((std::is_same<decltype(std::fmax((float)0, (long double)0)), long double>::value), "");
    static_assert((std::is_same<decltype(std::fmax((double)0, (long double)0)), long double>::value), "");
    static_assert((std::is_same<decltype(std::fmaxf(0,0)), float>::value), "");
    static_assert((std::is_same<decltype(std::fmaxl(0,0)), long double>::value), "");
    static_assert((std::is_same<decltype(std::fmax((int)0, (int)0)), double>::value), "");
    static_assert((std::is_same<decltype(fmax(Ambiguous(), Ambiguous())), Ambiguous>::value), "");
    assert(std::fmax(1,0) == 1);
}

void test_fmin()
{
    static_assert((std::is_same<decltype(std::fmin((float)0, (float)0)), float>::value), "");
    static_assert((std::is_same<decltype(std::fmin((bool)0, (float)0)), double>::value), "");
    static_assert((std::is_same<decltype(std::fmin((unsigned short)0, (double)0)), double>::value), "");
    static_assert((std::is_same<decltype(std::fmin((int)0, (long double)0)), long double>::value), "");
    static_assert((std::is_same<decltype(std::fmin((float)0, (unsigned int)0)), double>::value), "");
    static_assert((std::is_same<decltype(std::fmin((double)0, (long)0)), double>::value), "");
    static_assert((std::is_same<decltype(std::fmin((long double)0, (unsigned long)0)), long double>::value), "");
    static_assert((std::is_same<decltype(std::fmin((int)0, (long long)0)), double>::value), "");
    static_assert((std::is_same<decltype(std::fmin((int)0, (unsigned long long)0)), double>::value), "");
    static_assert((std::is_same<decltype(std::fmin((double)0, (double)0)), double>::value), "");
    static_assert((std::is_same<decltype(std::fmin((long double)0, (long double)0)), long double>::value), "");
    static_assert((std::is_same<decltype(std::fmin((float)0, (double)0)), double>::value), "");
    static_assert((std::is_same<decltype(std::fmin((float)0, (long double)0)), long double>::value), "");
    static_assert((std::is_same<decltype(std::fmin((double)0, (long double)0)), long double>::value), "");
    static_assert((std::is_same<decltype(std::fminf(0,0)), float>::value), "");
    static_assert((std::is_same<decltype(std::fminl(0,0)), long double>::value), "");
    static_assert((std::is_same<decltype(std::fmin((int)0, (int)0)), double>::value), "");
    static_assert((std::is_same<decltype(fmin(Ambiguous(), Ambiguous())), Ambiguous>::value), "");
    assert(std::fmin(1,0) == 0);
}

void test_nan()
{
    static_assert((std::is_same<decltype(std::nan("")), double>::value), "");
    static_assert((std::is_same<decltype(std::nanf("")), float>::value), "");
    static_assert((std::is_same<decltype(std::nanl("")), long double>::value), "");
}

void test_round()
{
    static_assert((std::is_same<decltype(std::round((float)0)), float>::value), "");
    static_assert((std::is_same<decltype(std::round((bool)0)), double>::value), "");
    static_assert((std::is_same<decltype(std::round((unsigned short)0)), double>::value), "");
    static_assert((std::is_same<decltype(std::round((int)0)), double>::value), "");
    static_assert((std::is_same<decltype(std::round((unsigned int)0)), double>::value), "");
    static_assert((std::is_same<decltype(std::round((long)0)), double>::value), "");
    static_assert((std::is_same<decltype(std::round((unsigned long)0)), double>::value), "");
    static_assert((std::is_same<decltype(std::round((long long)0)), double>::value), "");
    static_assert((std::is_same<decltype(std::round((unsigned long long)0)), double>::value), "");
    static_assert((std::is_same<decltype(std::round((double)0)), double>::value), "");
    static_assert((std::is_same<decltype(std::round((long double)0)), long double>::value), "");
    static_assert((std::is_same<decltype(std::roundf(0)), float>::value), "");
    static_assert((std::is_same<decltype(std::roundl(0)), long double>::value), "");
    static_assert((std::is_same<decltype(round(Ambiguous())), Ambiguous>::value), "");
    assert(std::round(1) == 1);
}

void test_trunc()
{
    static_assert((std::is_same<decltype(std::trunc((float)0)), float>::value), "");
    static_assert((std::is_same<decltype(std::trunc((bool)0)), double>::value), "");
    static_assert((std::is_same<decltype(std::trunc((unsigned short)0)), double>::value), "");
    static_assert((std::is_same<decltype(std::trunc((int)0)), double>::value), "");
    static_assert((std::is_same<decltype(std::trunc((unsigned int)0)), double>::value), "");
    static_assert((std::is_same<decltype(std::trunc((long)0)), double>::value), "");
    static_assert((std::is_same<decltype(std::trunc((unsigned long)0)), double>::value), "");
    static_assert((std::is_same<decltype(std::trunc((long long)0)), double>::value), "");
    static_assert((std::is_same<decltype(std::trunc((unsigned long long)0)), double>::value), "");
    static_assert((std::is_same<decltype(std::trunc((double)0)), double>::value), "");
    static_assert((std::is_same<decltype(std::trunc((long double)0)), long double>::value), "");
    static_assert((std::is_same<decltype(std::truncf(0)), float>::value), "");
    static_assert((std::is_same<decltype(std::truncl(0)), long double>::value), "");
    static_assert((std::is_same<decltype(trunc(Ambiguous())), Ambiguous>::value), "");
    assert(std::trunc(1) == 1);
}

ONEDPL_TEST_NUM_MAIN
{
    test_abs();
    test_ceil();
    test_exp();
    test_fabs();
    test_floor();

    test_isgreater();
    test_isgreaterequal();
    test_isinf();
    test_isless();
    test_islessequal();

    test_isnan();
    test_isunordered();

    test_copysign();

    test_fma();
    test_fmax();
    test_fmin();

    test_nan();

    test_round();
    test_trunc();

  return 0;
}
