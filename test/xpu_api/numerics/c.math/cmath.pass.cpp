//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// XFAIL: LIBCXX-AIX-FIXME

// <cmath>

#include <cmath>
#include <limits>
#include <type_traits>
#include <cassert>

#include "support/test_complex.h"
#include "support/hexfloat.h"
#include "support/truncate_fp.h"

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
Ambiguous acos(Ambiguous){ return Ambiguous(); }
Ambiguous asin(Ambiguous){ return Ambiguous(); }
Ambiguous atan(Ambiguous){ return Ambiguous(); }
Ambiguous atan2(Ambiguous, Ambiguous){ return Ambiguous(); }
Ambiguous ceil(Ambiguous){ return Ambiguous(); }
Ambiguous cos(Ambiguous){ return Ambiguous(); }
Ambiguous cosh(Ambiguous){ return Ambiguous(); }
Ambiguous exp(Ambiguous){ return Ambiguous(); }
Ambiguous fabs(Ambiguous){ return Ambiguous(); }
Ambiguous floor(Ambiguous){ return Ambiguous(); }
Ambiguous fmod(Ambiguous, Ambiguous){ return Ambiguous(); }
Ambiguous frexp(Ambiguous, int*){ return Ambiguous(); }
Ambiguous ldexp(Ambiguous, int){ return Ambiguous(); }
Ambiguous log(Ambiguous){ return Ambiguous(); }
Ambiguous log10(Ambiguous){ return Ambiguous(); }
Ambiguous modf(Ambiguous, Ambiguous*){ return Ambiguous(); }
Ambiguous pow(Ambiguous, Ambiguous){ return Ambiguous(); }
Ambiguous sin(Ambiguous){ return Ambiguous(); }
Ambiguous sinh(Ambiguous){ return Ambiguous(); }
Ambiguous sqrt(Ambiguous){ return Ambiguous(); }
Ambiguous tan(Ambiguous){ return Ambiguous(); }
Ambiguous tanh(Ambiguous){ return Ambiguous(); }
Ambiguous signbit(Ambiguous){ return Ambiguous(); }
Ambiguous fpclassify(Ambiguous){ return Ambiguous(); }
Ambiguous isfinite(Ambiguous){ return Ambiguous(); }
Ambiguous isnormal(Ambiguous){ return Ambiguous(); }
Ambiguous isgreater(Ambiguous, Ambiguous){ return Ambiguous(); }
Ambiguous isgreaterequal(Ambiguous, Ambiguous){ return Ambiguous(); }
Ambiguous isless(Ambiguous, Ambiguous){ return Ambiguous(); }
Ambiguous islessequal(Ambiguous, Ambiguous){ return Ambiguous(); }
Ambiguous islessgreater(Ambiguous, Ambiguous){ return Ambiguous(); }
Ambiguous isunordered(Ambiguous, Ambiguous){ return Ambiguous(); }
Ambiguous acosh(Ambiguous){ return Ambiguous(); }
Ambiguous asinh(Ambiguous){ return Ambiguous(); }
Ambiguous atanh(Ambiguous){ return Ambiguous(); }
Ambiguous cbrt(Ambiguous){ return Ambiguous(); }
Ambiguous copysign(Ambiguous, Ambiguous){ return Ambiguous(); }
Ambiguous erf(Ambiguous){ return Ambiguous(); }
Ambiguous erfc(Ambiguous){ return Ambiguous(); }
Ambiguous exp2(Ambiguous){ return Ambiguous(); }
Ambiguous expm1(Ambiguous){ return Ambiguous(); }
Ambiguous fdim(Ambiguous, Ambiguous){ return Ambiguous(); }
Ambiguous fma(Ambiguous, Ambiguous, Ambiguous){ return Ambiguous(); }
Ambiguous fmax(Ambiguous, Ambiguous){ return Ambiguous(); }
Ambiguous fmin(Ambiguous, Ambiguous){ return Ambiguous(); }
Ambiguous hypot(Ambiguous, Ambiguous){ return Ambiguous(); }
Ambiguous hypot(Ambiguous, Ambiguous, Ambiguous){ return Ambiguous(); }
Ambiguous ilogb(Ambiguous){ return Ambiguous(); }
Ambiguous lgamma(Ambiguous){ return Ambiguous(); }
Ambiguous llrint(Ambiguous){ return Ambiguous(); }
Ambiguous llround(Ambiguous){ return Ambiguous(); }
Ambiguous log1p(Ambiguous){ return Ambiguous(); }
Ambiguous log2(Ambiguous){ return Ambiguous(); }
Ambiguous logb(Ambiguous){ return Ambiguous(); }
Ambiguous lrint(Ambiguous){ return Ambiguous(); }
Ambiguous lround(Ambiguous){ return Ambiguous(); }
Ambiguous nearbyint(Ambiguous){ return Ambiguous(); }
Ambiguous nextafter(Ambiguous, Ambiguous){ return Ambiguous(); }
Ambiguous nexttoward(Ambiguous, Ambiguous){ return Ambiguous(); }
Ambiguous remainder(Ambiguous, Ambiguous){ return Ambiguous(); }
Ambiguous remquo(Ambiguous, Ambiguous, int*){ return Ambiguous(); }
Ambiguous rint(Ambiguous){ return Ambiguous(); }
Ambiguous round(Ambiguous){ return Ambiguous(); }
Ambiguous scalbln(Ambiguous, Ambiguous){ return Ambiguous(); }
Ambiguous scalbn(Ambiguous, Ambiguous){ return Ambiguous(); }
Ambiguous tgamma(Ambiguous){ return Ambiguous(); }
Ambiguous trunc(Ambiguous){ return Ambiguous(); }

template <class T, class = decltype(dpl::abs(std::declval<T>()))>
std::true_type has_abs_imp(int);
template <class T>
std::false_type has_abs_imp(...);

template <class T>
struct has_abs : decltype(has_abs_imp<T>(0)) {};

ONEDPL_TEST_NUM_TEMPLATE
void test_abs()
{
    // See also "abs.pass.cpp"
    static_assert((std::is_same<decltype(dpl::abs((float)0)), float>::value), "");
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::abs((double)0)), double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::abs((long double)0)), long double>::value), ""))
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

    assert(dpl::abs(-1.) == 1);
}

ONEDPL_TEST_NUM_TEMPLATE
void test_acos()
{
    static_assert((std::is_same<decltype(dpl::acos((float)0)), float>::value), "");
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::acos((bool)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::acos((unsigned short)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::acos((int)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::acos((unsigned int)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::acos((long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::acos((unsigned long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::acos((long long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::acos((unsigned long long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::acos((double)0)), double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::acos((long double)0)), long double>::value), ""))
    static_assert((std::is_same<decltype(dpl::acosf(0)), float>::value), "");
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::acosl(0)), long double>::value), ""))
    static_assert((std::is_same<decltype(acos(Ambiguous())), Ambiguous>::value), "");
    assert(dpl::acos(1) == 0);
}

ONEDPL_TEST_NUM_TEMPLATE
void test_asin()
{
    static_assert((std::is_same<decltype(dpl::asin((float)0)), float>::value), "");
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::asin((bool)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::asin((unsigned short)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::asin((int)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::asin((unsigned int)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::asin((long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::asin((unsigned long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::asin((long long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::asin((unsigned long long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::asin((double)0)), double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::asin((long double)0)), long double>::value), ""))
    static_assert((std::is_same<decltype(dpl::asinf(0)), float>::value), "");
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::asinl(0)), long double>::value), ""))
    static_assert((std::is_same<decltype(asin(Ambiguous())), Ambiguous>::value), "");
    assert(dpl::asin(0) == 0);
}

ONEDPL_TEST_NUM_TEMPLATE
void test_atan()
{
    static_assert((std::is_same<decltype(dpl::atan((float)0)), float>::value), "");
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::atan((bool)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::atan((unsigned short)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::atan((int)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::atan((unsigned int)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::atan((long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::atan((unsigned long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::atan((long long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::atan((unsigned long long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::atan((double)0)), double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::atan((long double)0)), long double>::value), ""))
    static_assert((std::is_same<decltype(dpl::atanf(0)), float>::value), "");
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::atanl(0)), long double>::value), ""))
    static_assert((std::is_same<decltype(atan(Ambiguous())), Ambiguous>::value), "");
    assert(dpl::atan(0) == 0);
}

ONEDPL_TEST_NUM_TEMPLATE
void test_atan2()
{
    static_assert((std::is_same<decltype(dpl::atan2((float)0, (float)0)), float>::value), "");
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::atan2((bool)0, (float)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::atan2((unsigned short)0, (double)0)), double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::atan2((int)0, (long double)0)), long double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::atan2((float)0, (unsigned int)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::atan2((double)0, (long)0)), double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::atan2((long double)0, (unsigned long)0)), long double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::atan2((int)0, (long long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::atan2((int)0, (unsigned long long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::atan2((double)0, (double)0)), double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::atan2((long double)0, (long double)0)), long double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::atan2((float)0, (double)0)), double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::atan2((float)0, (long double)0)), long double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::atan2((double)0, (long double)0)), long double>::value), ""))
    static_assert((std::is_same<decltype(dpl::atan2f(0, 0)), float>::value), "");
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::atan2l(0,0)), long double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::atan2((int)0, (int)0)), double>::value), ""))
    static_assert((std::is_same<decltype(atan2(Ambiguous(), Ambiguous())), Ambiguous>::value), "");
    assert(dpl::atan2(0, 1) == 0);
}

ONEDPL_TEST_NUM_TEMPLATE
void test_ceil()
{
    static_assert((std::is_same<decltype(dpl::ceil((float)0)), float>::value), "");
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::ceil((bool)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::ceil((unsigned short)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::ceil((int)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::ceil((unsigned int)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::ceil((long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::ceil((unsigned long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::ceil((long long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::ceil((unsigned long long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::ceil((double)0)), double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::ceil((long double)0)), long double>::value), ""))
    static_assert((std::is_same<decltype(dpl::ceilf(0)), float>::value), "");
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::ceill(0)), long double>::value), ""))
    static_assert((std::is_same<decltype(ceil(Ambiguous())), Ambiguous>::value), "");
    assert(dpl::ceil(0) == 0);
}

ONEDPL_TEST_NUM_TEMPLATE
void test_cos()
{
    static_assert((std::is_same<decltype(dpl::cos((float)0)), float>::value), "");
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::cos((bool)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::cos((unsigned short)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::cos((int)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::cos((unsigned int)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::cos((long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::cos((unsigned long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::cos((long long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::cos((unsigned long long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::cos((double)0)), double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::cos((long double)0)), long double>::value), ""))
    static_assert((std::is_same<decltype(dpl::cosf(0)), float>::value), "");
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::cosl(0)), long double>::value), ""))
    static_assert((std::is_same<decltype(cos(Ambiguous())), Ambiguous>::value), "");
    assert(dpl::cos(0) == 1);
}

ONEDPL_TEST_NUM_TEMPLATE
void test_cosh()
{
    static_assert((std::is_same<decltype(dpl::cosh((float)0)), float>::value), "");
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::cosh((bool)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::cosh((unsigned short)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::cosh((int)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::cosh((unsigned int)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::cosh((long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::cosh((unsigned long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::cosh((long long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::cosh((unsigned long long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::cosh((double)0)), double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::cosh((long double)0)), long double>::value), ""))
    static_assert((std::is_same<decltype(dpl::coshf(0)), float>::value), "");
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::coshl(0)), long double>::value), ""))
    static_assert((std::is_same<decltype(cosh(Ambiguous())), Ambiguous>::value), "");
    assert(dpl::cosh(0) == 1);
}

ONEDPL_TEST_NUM_TEMPLATE
void test_exp()
{
    static_assert((std::is_same<decltype(dpl::exp((float)0)), float>::value), "");
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::exp((bool)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::exp((unsigned short)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::exp((int)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::exp((unsigned int)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::exp((long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::exp((unsigned long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::exp((long long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::exp((unsigned long long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::exp((double)0)), double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::exp((long double)0)), long double>::value), ""))
    static_assert((std::is_same<decltype(dpl::expf(0)), float>::value), "");
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::expl(0)), long double>::value), ""))
    static_assert((std::is_same<decltype(exp(Ambiguous())), Ambiguous>::value), "");
    assert(dpl::exp(0) == 1);
}

ONEDPL_TEST_NUM_TEMPLATE
void test_fabs()
{
    static_assert((std::is_same<decltype(dpl::fabs((float)0)), float>::value), "");
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::fabs((bool)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::fabs((unsigned short)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::fabs((int)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::fabs((unsigned int)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::fabs((long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::fabs((unsigned long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::fabs((long long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::fabs((unsigned long long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::fabs((double)0)), double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::fabs((long double)0)), long double>::value), ""))
    static_assert((std::is_same<decltype(dpl::fabsf(0.0f)), float>::value), "");
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::fabsl(0.0L)), long double>::value), ""))
    static_assert((std::is_same<decltype(fabs(Ambiguous())), Ambiguous>::value), "");
    assert(dpl::fabs(-1) == 1);
}

ONEDPL_TEST_NUM_TEMPLATE
void test_floor()
{
    static_assert((std::is_same<decltype(dpl::floor((float)0)), float>::value), "");
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::floor((bool)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::floor((unsigned short)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::floor((int)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::floor((unsigned int)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::floor((long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::floor((unsigned long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::floor((long long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::floor((unsigned long long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::floor((double)0)), double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::floor((long double)0)), long double>::value), ""))
    static_assert((std::is_same<decltype(dpl::floorf(0)), float>::value), "");
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::floorl(0)), long double>::value), ""))
    static_assert((std::is_same<decltype(floor(Ambiguous())), Ambiguous>::value), "");
    assert(dpl::floor(1) == 1);
}

ONEDPL_TEST_NUM_TEMPLATE
void test_fmod()
{
    static_assert((std::is_same<decltype(dpl::fmod((float)0, (float)0)), float>::value), "");
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::fmod((bool)0, (float)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::fmod((unsigned short)0, (double)0)), double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::fmod((int)0, (long double)0)), long double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::fmod((float)0, (unsigned int)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::fmod((double)0, (long)0)), double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::fmod((long double)0, (unsigned long)0)), long double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::fmod((int)0, (long long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::fmod((int)0, (unsigned long long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::fmod((double)0, (double)0)), double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::fmod((long double)0, (long double)0)), long double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::fmod((float)0, (double)0)), double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::fmod((float)0, (long double)0)), long double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::fmod((double)0, (long double)0)), long double>::value), ""))
    static_assert((std::is_same<decltype(dpl::fmodf(0,0)), float>::value), "");
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::fmodl(0,0)), long double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::fmod((int)0, (int)0)), double>::value), ""))
    static_assert((std::is_same<decltype(fmod(Ambiguous(), Ambiguous())), Ambiguous>::value), "");
    assert(dpl::fmod(1.5,1) == .5);
}

ONEDPL_TEST_NUM_TEMPLATE
void test_frexp()
{
    int ip;
    static_assert((std::is_same<decltype(dpl::frexp((float)0, &ip)), float>::value), "");
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::frexp((bool)0, &ip)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::frexp((unsigned short)0, &ip)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::frexp((int)0, &ip)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::frexp((unsigned int)0, &ip)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::frexp((long)0, &ip)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::frexp((unsigned long)0, &ip)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::frexp((long long)0, &ip)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::frexp((unsigned long long)0, &ip)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::frexp((double)0, &ip)), double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::frexp((long double)0, &ip)), long double>::value), ""))
    static_assert((std::is_same<decltype(dpl::frexpf(0, &ip)), float>::value), "");
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::frexpl(0, &ip)), long double>::value), ""))
    static_assert((std::is_same<decltype(frexp(Ambiguous(), &ip)), Ambiguous>::value), "");
    assert(dpl::frexp(0, &ip) == 0);
}

ONEDPL_TEST_NUM_TEMPLATE
void test_ldexp()
{
    int ip = 1;
    static_assert((std::is_same<decltype(dpl::ldexp((float)0, ip)), float>::value), "");
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::ldexp((bool)0, ip)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::ldexp((unsigned short)0, ip)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::ldexp((int)0, ip)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::ldexp((unsigned int)0, ip)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::ldexp((long)0, ip)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::ldexp((unsigned long)0, ip)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::ldexp((long long)0, ip)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::ldexp((unsigned long long)0, ip)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::ldexp((double)0, ip)), double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::ldexp((long double)0, ip)), long double>::value), ""))
    static_assert((std::is_same<decltype(dpl::ldexpf(0, ip)), float>::value), "");
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::ldexpl(0, ip)), long double>::value), ""))
    static_assert((std::is_same<decltype(ldexp(Ambiguous(), ip)), Ambiguous>::value), "");
    assert(dpl::ldexp(1, ip) == 2);
}

ONEDPL_TEST_NUM_TEMPLATE
void test_log()
{
    static_assert((std::is_same<decltype(dpl::log((float)0)), float>::value), "");
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::log((bool)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::log((unsigned short)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::log((int)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::log((unsigned int)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::log((long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::log((unsigned long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::log((long long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::log((unsigned long long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::log((double)0)), double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::log((long double)0)), long double>::value), ""))
    static_assert((std::is_same<decltype(dpl::logf(0)), float>::value), "");
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::logl(0)), long double>::value), ""))
    static_assert((std::is_same<decltype(log(Ambiguous())), Ambiguous>::value), "");
    assert(dpl::log(1) == 0);
}

ONEDPL_TEST_NUM_TEMPLATE
void test_log10()
{
    static_assert((std::is_same<decltype(dpl::log10((float)0)), float>::value), "");
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::log10((bool)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::log10((unsigned short)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::log10((int)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::log10((unsigned int)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::log10((long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::log10((unsigned long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::log10((long long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::log10((unsigned long long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::log10((double)0)), double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::log10((long double)0)), long double>::value), ""))
    static_assert((std::is_same<decltype(dpl::log10f(0)), float>::value), "");
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::log10l(0)), long double>::value), ""))
    static_assert((std::is_same<decltype(log10(Ambiguous())), Ambiguous>::value), "");
    assert(dpl::log10(1) == 0);
}

ONEDPL_TEST_NUM_TEMPLATE
void test_modf()
{
    static_assert((std::is_same<decltype(dpl::modf((float)0, (float*)0)), float>::value), "");
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::modf((double)0, (double*)0)), double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::modf((long double)0, (long double*)0)), long double>::value), ""))
    static_assert((std::is_same<decltype(dpl::modff(0, (float*)0)), float>::value), "");
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::modfl(0, (long double*)0)), long double>::value), ""))
    static_assert((std::is_same<decltype(modf(Ambiguous(), (Ambiguous*)0)), Ambiguous>::value), "");
    double i;
    assert(dpl::modf(1., &i) == 0);
}

ONEDPL_TEST_NUM_TEMPLATE
void test_pow()
{
    static_assert((std::is_same<decltype(dpl::pow((float)0, (float)0)), float>::value), "");
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::pow((bool)0, (float)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::pow((unsigned short)0, (double)0)), double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::pow((int)0, (long double)0)), long double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::pow((float)0, (unsigned int)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::pow((double)0, (long)0)), double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::pow((long double)0, (unsigned long)0)), long double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::pow((int)0, (long long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::pow((int)0, (unsigned long long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::pow((double)0, (double)0)), double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::pow((long double)0, (long double)0)), long double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::pow((float)0, (double)0)), double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::pow((float)0, (long double)0)), long double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::pow((double)0, (long double)0)), long double>::value), ""))
    static_assert((std::is_same<decltype(dpl::powf(0,0)), float>::value), "");
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::powl(0,0)), long double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::pow((int)0, (int)0)), double>::value), ""))
//     IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::pow(Value<int>(), (int)0)), double>::value), "");
//     IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::pow(Value<long double>(), (float)0)), long double>::value), "");
//     static_assert((std::is_same<decltype(dpl::pow((float) 0, Value<float>())), float>::value), "");
    static_assert((std::is_same<decltype(pow(Ambiguous(), Ambiguous())), Ambiguous>::value), "");
    assert(dpl::pow(1,1) == 1);
//     assert(dpl::pow(Value<int,1>(), Value<float,1>())  == 1);
//     assert(dpl::pow(1.0f, Value<double,1>()) == 1);
//     assert(dpl::pow(1.0, Value<int,1>()) == 1);
//     IF_LONG_DOUBLE_SUPPORT(assert(dpl::pow(Value<long double,1>(), 1LL) == 1);
}

ONEDPL_TEST_NUM_TEMPLATE
void test_sin()
{
    static_assert((std::is_same<decltype(dpl::sin((float)0)), float>::value), "");
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::sin((bool)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::sin((unsigned short)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::sin((int)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::sin((unsigned int)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::sin((long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::sin((unsigned long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::sin((long long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::sin((unsigned long long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::sin((double)0)), double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::sin((long double)0)), long double>::value), ""))
    static_assert((std::is_same<decltype(dpl::sinf(0)), float>::value), "");
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::sinl(0)), long double>::value), ""))
    static_assert((std::is_same<decltype(sin(Ambiguous())), Ambiguous>::value), "");
    assert(dpl::sin(0) == 0);
}

ONEDPL_TEST_NUM_TEMPLATE
void test_sinh()
{
    static_assert((std::is_same<decltype(dpl::sinh((float)0)), float>::value), "");
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::sinh((bool)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::sinh((unsigned short)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::sinh((int)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::sinh((unsigned int)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::sinh((long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::sinh((unsigned long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::sinh((long long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::sinh((unsigned long long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::sinh((double)0)), double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::sinh((long double)0)), long double>::value), ""))
    static_assert((std::is_same<decltype(dpl::sinhf(0)), float>::value), "");
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::sinhl(0)), long double>::value), ""))
    static_assert((std::is_same<decltype(sinh(Ambiguous())), Ambiguous>::value), "");
    assert(dpl::sinh(0) == 0);
}

ONEDPL_TEST_NUM_TEMPLATE
void test_sqrt()
{
    static_assert((std::is_same<decltype(dpl::sqrt((float)0)), float>::value), "");
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::sqrt((bool)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::sqrt((unsigned short)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::sqrt((int)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::sqrt((unsigned int)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::sqrt((long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::sqrt((unsigned long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::sqrt((long long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::sqrt((unsigned long long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::sqrt((double)0)), double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::sqrt((long double)0)), long double>::value), ""))
    static_assert((std::is_same<decltype(dpl::sqrtf(0)), float>::value), "");
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::sqrtl(0)), long double>::value), ""))
    static_assert((std::is_same<decltype(sqrt(Ambiguous())), Ambiguous>::value), "");
    assert(dpl::sqrt(4) == 2);
}

ONEDPL_TEST_NUM_TEMPLATE
void test_tan()
{
    static_assert((std::is_same<decltype(dpl::tan((float)0)), float>::value), "");
    static_assert((std::is_same<decltype(sqrt(Ambiguous())), Ambiguous>::value), "");
    static_assert((std::is_same<decltype(sqrt(Ambiguous())), Ambiguous>::value), "");
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::tan((bool)0)), double>::value), ""))
    static_assert((std::is_same<decltype(sqrt(Ambiguous())), Ambiguous>::value), "");
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::tan((unsigned short)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::tan((int)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::tan((unsigned int)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::tan((long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::tan((unsigned long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::tan((long long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::tan((unsigned long long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::tan((double)0)), double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::tan((long double)0)), long double>::value), ""))
    static_assert((std::is_same<decltype(dpl::tanf(0)), float>::value), "");
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::tanl(0)), long double>::value), ""))
    static_assert((std::is_same<decltype(tan(Ambiguous())), Ambiguous>::value), "");
    assert(dpl::tan(0) == 0);
}

ONEDPL_TEST_NUM_TEMPLATE
void test_tanh()
{
    static_assert((std::is_same<decltype(dpl::tanh((float)0)), float>::value), "");
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::tanh((bool)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::tanh((unsigned short)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::tanh((int)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::tanh((unsigned int)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::tanh((long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::tanh((unsigned long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::tanh((long long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::tanh((unsigned long long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::tanh((double)0)), double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::tanh((long double)0)), long double>::value), ""))
    static_assert((std::is_same<decltype(dpl::tanhf(0)), float>::value), "");
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::tanhl(0)), long double>::value), ""))
    static_assert((std::is_same<decltype(tanh(Ambiguous())), Ambiguous>::value), "");
    assert(dpl::tanh(0) == 0);
}

ONEDPL_TEST_NUM_TEMPLATE
void test_signbit()
{
#ifdef signbit
#error signbit defined
#endif
    static_assert((std::is_same<decltype(dpl::signbit((float)0)), bool>::value), "");
    static_assert((std::is_same<decltype(dpl::signbit((double)0)), bool>::value), "");
    static_assert((std::is_same<decltype(dpl::signbit(0)), bool>::value), "");
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::signbit((long double)0)), bool>::value), ""))
    static_assert((std::is_same<decltype(signbit(Ambiguous())), Ambiguous>::value), "");
    assert(dpl::signbit(-1.0) == true);
    assert(dpl::signbit(0u) == false);
    assert(dpl::signbit(std::numeric_limits<unsigned>::max()) == false);
    assert(dpl::signbit(0) == false);
    assert(dpl::signbit(1) == false);
    assert(dpl::signbit(-1) == true);
    assert(dpl::signbit(std::numeric_limits<int>::max()) == false);
    assert(dpl::signbit(std::numeric_limits<int>::min()) == true);
}

ONEDPL_TEST_NUM_TEMPLATE
void test_fpclassify()
{
#ifdef fpclassify
#error fpclassify defined
#endif
    static_assert((std::is_same<decltype(dpl::fpclassify((float)0)), int>::value), "");
    static_assert((std::is_same<decltype(dpl::fpclassify((double)0)), int>::value), "");
    static_assert((std::is_same<decltype(dpl::fpclassify(0)), int>::value), "");
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::fpclassify((long double)0)), int>::value), ""))
    static_assert((std::is_same<decltype(fpclassify(Ambiguous())), Ambiguous>::value), "");
    assert(dpl::fpclassify(-1.0) == FP_NORMAL);
    assert(dpl::fpclassify(0) == FP_ZERO);
    assert(dpl::fpclassify(1) == FP_NORMAL);
    assert(dpl::fpclassify(-1) == FP_NORMAL);
    assert(dpl::fpclassify(std::numeric_limits<int>::max()) == FP_NORMAL);
    assert(dpl::fpclassify(std::numeric_limits<int>::min()) == FP_NORMAL);
}

ONEDPL_TEST_NUM_TEMPLATE
void test_isfinite()
{
#ifdef isfinite
#error isfinite defined
#endif
    static_assert((std::is_same<decltype(dpl::isfinite((float)0)), bool>::value), "");
    static_assert((std::is_same<decltype(dpl::isfinite((double)0)), bool>::value), "");
    static_assert((std::is_same<decltype(dpl::isfinite(0)), bool>::value), "");
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::isfinite((long double)0)), bool>::value), ""))
    static_assert((std::is_same<decltype(isfinite(Ambiguous())), Ambiguous>::value), "");
    assert(dpl::isfinite(-1.0) == true);
    assert(dpl::isfinite(0) == true);
    assert(dpl::isfinite(1) == true);
    assert(dpl::isfinite(-1) == true);
    assert(dpl::isfinite(std::numeric_limits<int>::max()) == true);
    assert(dpl::isfinite(std::numeric_limits<int>::min()) == true);
}

ONEDPL_TEST_NUM_TEMPLATE
void test_isnormal()
{
#ifdef isnormal
#error isnormal defined
#endif
    static_assert((std::is_same<decltype(dpl::isnormal((float)0)), bool>::value), "");
    static_assert((std::is_same<decltype(dpl::isnormal((double)0)), bool>::value), "");
    static_assert((std::is_same<decltype(dpl::isnormal(0)), bool>::value), "");
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::isnormal((long double)0)), bool>::value), ""))
    static_assert((std::is_same<decltype(isnormal(Ambiguous())), Ambiguous>::value), "");
    assert(dpl::isnormal(-1.0) == true);
    assert(dpl::isnormal(0) == false);
    assert(dpl::isnormal(1) == true);
    assert(dpl::isnormal(-1) == true);
    assert(dpl::isnormal(std::numeric_limits<int>::max()) == true);
    assert(dpl::isnormal(std::numeric_limits<int>::min()) == true);
}

ONEDPL_TEST_NUM_TEMPLATE
void
test_isgreater()
{
#ifdef isgreater
#error isgreater defined
#endif
    static_assert((std::is_same<decltype(dpl::isgreater((float)0, (float)0)), bool>::value), "");
    static_assert((std::is_same<decltype(dpl::isgreater((float)0, (double)0)), bool>::value), "");
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::isgreater((float)0, (long double)0)), bool>::value), ""))
    static_assert((std::is_same<decltype(dpl::isgreater((double)0, (float)0)), bool>::value), "");
    static_assert((std::is_same<decltype(dpl::isgreater((double)0, (double)0)), bool>::value), "");
    static_assert((std::is_same<decltype(dpl::isgreater(0, (double)0)), bool>::value), "");
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::isgreater((double)0, (long double)0)), bool>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::isgreater((long double)0, (float)0)), bool>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::isgreater((long double)0, (double)0)), bool>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::isgreater((long double)0, (long double)0)), bool>::value), ""))
    static_assert((std::is_same<decltype(isgreater(Ambiguous(), Ambiguous())), Ambiguous>::value), "");
    assert(dpl::isgreater(-1.0, 0.F) == false);
}

ONEDPL_TEST_NUM_TEMPLATE
void test_isgreaterequal()
{
#ifdef isgreaterequal
#error isgreaterequal defined
#endif
    static_assert((std::is_same<decltype(dpl::isgreaterequal((float)0, (float)0)), bool>::value), "");
    static_assert((std::is_same<decltype(dpl::isgreaterequal((float)0, (double)0)), bool>::value), "");
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::isgreaterequal((float)0, (long double)0)), bool>::value), ""))
    static_assert((std::is_same<decltype(dpl::isgreaterequal((double)0, (float)0)), bool>::value), "");
    static_assert((std::is_same<decltype(dpl::isgreaterequal((double)0, (double)0)), bool>::value), "");
    static_assert((std::is_same<decltype(dpl::isgreaterequal(0, (double)0)), bool>::value), "");
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::isgreaterequal((double)0, (long double)0)), bool>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::isgreaterequal((long double)0, (float)0)), bool>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::isgreaterequal((long double)0, (double)0)), bool>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::isgreaterequal((long double)0, (long double)0)), bool>::value), ""))
    static_assert((std::is_same<decltype(isgreaterequal(Ambiguous(), Ambiguous())), Ambiguous>::value), "");
    assert(dpl::isgreaterequal(-1.0, 0.F) == false);
}

ONEDPL_TEST_NUM_TEMPLATE
void test_isinf()
{
#ifdef isinf
#error isinf defined
#endif
    static_assert((std::is_same<decltype(dpl::isinf((float)0)), bool>::value), "");

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

    static_assert((std::is_same<decltype(dpl::isinf(0)), bool>::value), "");
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::isinf((long double)0)), bool>::value), ""))
    assert(dpl::isinf(-1.0) == false);
    assert(dpl::isinf(0) == false);
    assert(dpl::isinf(1) == false);
    assert(dpl::isinf(-1) == false);
    assert(dpl::isinf(std::numeric_limits<int>::max()) == false);
    assert(dpl::isinf(std::numeric_limits<int>::min()) == false);
}

ONEDPL_TEST_NUM_TEMPLATE
void test_isless()
{
#ifdef isless
#error isless defined
#endif
    static_assert((std::is_same<decltype(dpl::isless((float)0, (float)0)), bool>::value), "");
    static_assert((std::is_same<decltype(dpl::isless((float)0, (double)0)), bool>::value), "");
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::isless((float)0, (long double)0)), bool>::value), ""))
    static_assert((std::is_same<decltype(dpl::isless((double)0, (float)0)), bool>::value), "");
    static_assert((std::is_same<decltype(dpl::isless((double)0, (double)0)), bool>::value), "");
    static_assert((std::is_same<decltype(dpl::isless(0, (double)0)), bool>::value), "");
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::isless((double)0, (long double)0)), bool>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::isless((long double)0, (float)0)), bool>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::isless((long double)0, (double)0)), bool>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::isless((long double)0, (long double)0)), bool>::value), ""))
    static_assert((std::is_same<decltype(isless(Ambiguous(), Ambiguous())), Ambiguous>::value), "");
    assert(dpl::isless(-1.0, 0.F) == true);
}

ONEDPL_TEST_NUM_TEMPLATE
void test_islessequal()
{
#ifdef islessequal
#error islessequal defined
#endif
    static_assert((std::is_same<decltype(dpl::islessequal((float)0, (float)0)), bool>::value), "");
    static_assert((std::is_same<decltype(dpl::islessequal((float)0, (double)0)), bool>::value), "");
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::islessequal((float)0, (long double)0)), bool>::value), ""))
    static_assert((std::is_same<decltype(dpl::islessequal((double)0, (float)0)), bool>::value), "");
    static_assert((std::is_same<decltype(dpl::islessequal((double)0, (double)0)), bool>::value), "");
    static_assert((std::is_same<decltype(dpl::islessequal(0, (double)0)), bool>::value), "");
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::islessequal((double)0, (long double)0)), bool>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::islessequal((long double)0, (float)0)), bool>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::islessequal((long double)0, (double)0)), bool>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::islessequal((long double)0, (long double)0)), bool>::value), ""))
    static_assert((std::is_same<decltype(islessequal(Ambiguous(), Ambiguous())), Ambiguous>::value), "");
    assert(dpl::islessequal(-1.0, 0.F) == true);
}

ONEDPL_TEST_NUM_TEMPLATE
void test_islessgreater()
{
#ifdef islessgreater
#error islessgreater defined
#endif
    static_assert((std::is_same<decltype(dpl::islessgreater((float)0, (float)0)), bool>::value), "");
    static_assert((std::is_same<decltype(dpl::islessgreater((float)0, (double)0)), bool>::value), "");
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::islessgreater((float)0, (long double)0)), bool>::value), ""))
    static_assert((std::is_same<decltype(dpl::islessgreater((double)0, (float)0)), bool>::value), "");
    static_assert((std::is_same<decltype(dpl::islessgreater((double)0, (double)0)), bool>::value), "");
    static_assert((std::is_same<decltype(dpl::islessgreater(0, (double)0)), bool>::value), "");
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::islessgreater((double)0, (long double)0)), bool>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::islessgreater((long double)0, (float)0)), bool>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::islessgreater((long double)0, (double)0)), bool>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::islessgreater((long double)0, (long double)0)), bool>::value), ""))
    static_assert((std::is_same<decltype(islessgreater(Ambiguous(), Ambiguous())), Ambiguous>::value), "");
    assert(dpl::islessgreater(-1.0, 0.F) == true);
}

ONEDPL_TEST_NUM_TEMPLATE
void test_isnan()
{
#ifdef isnan
#error isnan defined
#endif
    static_assert((std::is_same<decltype(dpl::isnan((float)0)), bool>::value), "");

    typedef decltype(dpl::isnan((double)0)) DoubleRetType;
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

    static_assert((std::is_same<decltype(dpl::isnan(0)), bool>::value), "");
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::isnan((long double)0)), bool>::value), ""))
    assert(dpl::isnan(-1.0) == false);
    assert(dpl::isnan(0) == false);
    assert(dpl::isnan(1) == false);
    assert(dpl::isnan(-1) == false);
    assert(dpl::isnan(std::numeric_limits<int>::max()) == false);
    assert(dpl::isnan(std::numeric_limits<int>::min()) == false);
}

ONEDPL_TEST_NUM_TEMPLATE
void test_isunordered()
{
#ifdef isunordered
#error isunordered defined
#endif
    static_assert((std::is_same<decltype(dpl::isunordered((float)0, (float)0)), bool>::value), "");
    static_assert((std::is_same<decltype(dpl::isunordered((float)0, (double)0)), bool>::value), "");
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::isunordered((float)0, (long double)0)), bool>::value), ""))
    static_assert((std::is_same<decltype(dpl::isunordered((double)0, (float)0)), bool>::value), "");
    static_assert((std::is_same<decltype(dpl::isunordered((double)0, (double)0)), bool>::value), "");
    static_assert((std::is_same<decltype(dpl::isunordered(0, (double)0)), bool>::value), "");
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::isunordered((double)0, (long double)0)), bool>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::isunordered((long double)0, (float)0)), bool>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::isunordered((long double)0, (double)0)), bool>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::isunordered((long double)0, (long double)0)), bool>::value), ""))
    static_assert((std::is_same<decltype(isunordered(Ambiguous(), Ambiguous())), Ambiguous>::value), "");
    assert(dpl::isunordered(-1.0, 0.F) == false);
}

ONEDPL_TEST_NUM_TEMPLATE
void test_acosh()
{
    static_assert((std::is_same<decltype(dpl::acosh((float)0)), float>::value), "");
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::acosh((bool)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::acosh((unsigned short)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::acosh((int)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::acosh((unsigned int)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::acosh((long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::acosh((unsigned long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::acosh((long long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::acosh((unsigned long long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::acosh((double)0)), double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::acosh((long double)0)), long double>::value), ""))
    static_assert((std::is_same<decltype(dpl::acoshf(0)), float>::value), "");
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::acoshl(0)), long double>::value), ""))
    static_assert((std::is_same<decltype(acosh(Ambiguous())), Ambiguous>::value), "");
    assert(dpl::acosh(1) == 0);
}

ONEDPL_TEST_NUM_TEMPLATE
void test_asinh()
{
    static_assert((std::is_same<decltype(dpl::asinh((float)0)), float>::value), "");
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::asinh((bool)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::asinh((unsigned short)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::asinh((int)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::asinh((unsigned int)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::asinh((long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::asinh((unsigned long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::asinh((long long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::asinh((unsigned long long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::asinh((double)0)), double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::asinh((long double)0)), long double>::value), ""))
    static_assert((std::is_same<decltype(dpl::asinhf(0)), float>::value), "");
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::asinhl(0)), long double>::value), ""))
    static_assert((std::is_same<decltype(asinh(Ambiguous())), Ambiguous>::value), "");
    assert(dpl::asinh(0) == 0);
}

ONEDPL_TEST_NUM_TEMPLATE
void test_atanh()
{
    static_assert((std::is_same<decltype(dpl::atanh((float)0)), float>::value), "");
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::atanh((bool)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::atanh((unsigned short)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::atanh((int)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::atanh((unsigned int)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::atanh((long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::atanh((unsigned long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::atanh((long long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::atanh((unsigned long long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::atanh((double)0)), double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::atanh((long double)0)), long double>::value), ""))
    static_assert((std::is_same<decltype(dpl::atanhf(0)), float>::value), "");
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::atanhl(0)), long double>::value), ""))
    static_assert((std::is_same<decltype(atanh(Ambiguous())), Ambiguous>::value), "");
    assert(dpl::atanh(0) == 0);
}

ONEDPL_TEST_NUM_TEMPLATE
void test_cbrt()
{
    static_assert((std::is_same<decltype(dpl::cbrt((float)0)), float>::value), "");
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::cbrt((bool)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::cbrt((unsigned short)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::cbrt((int)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::cbrt((unsigned int)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::cbrt((long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::cbrt((unsigned long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::cbrt((long long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::cbrt((unsigned long long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::cbrt((double)0)), double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::cbrt((long double)0)), long double>::value), ""))
    static_assert((std::is_same<decltype(dpl::cbrtf(0)), float>::value), "");
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::cbrtl(0)), long double>::value), ""))
    static_assert((std::is_same<decltype(cbrt(Ambiguous())), Ambiguous>::value), "");
    assert(truncate_fp(dpl::cbrt(1)) == 1);
}

ONEDPL_TEST_NUM_TEMPLATE
void test_copysign()
{
    static_assert((std::is_same<decltype(dpl::copysign((float)0, (float)0)), float>::value), "");
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::copysign((bool)0, (float)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::copysign((unsigned short)0, (double)0)), double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::copysign((int)0, (long double)0)), long double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::copysign((float)0, (unsigned int)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::copysign((double)0, (long)0)), double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::copysign((long double)0, (unsigned long)0)), long double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::copysign((int)0, (long long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::copysign((int)0, (unsigned long long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::copysign((double)0, (double)0)), double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::copysign((long double)0, (long double)0)), long double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::copysign((float)0, (double)0)), double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::copysign((float)0, (long double)0)), long double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::copysign((double)0, (long double)0)), long double>::value), ""))
    static_assert((std::is_same<decltype(dpl::copysignf(0,0)), float>::value), "");
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::copysignl(0,0)), long double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::copysign((int)0, (int)0)), double>::value), ""))
    static_assert((std::is_same<decltype(copysign(Ambiguous(), Ambiguous())), Ambiguous>::value), "");
    assert(dpl::copysign(1,1) == 1);
}

ONEDPL_TEST_NUM_TEMPLATE
void test_erf()
{
    static_assert((std::is_same<decltype(dpl::erf((float)0)), float>::value), "");
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::erf((bool)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::erf((unsigned short)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::erf((int)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::erf((unsigned int)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::erf((long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::erf((unsigned long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::erf((long long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::erf((unsigned long long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::erf((double)0)), double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::erf((long double)0)), long double>::value), ""))
    static_assert((std::is_same<decltype(dpl::erff(0)), float>::value), "");
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::erfl(0)), long double>::value), ""))
    static_assert((std::is_same<decltype(erf(Ambiguous())), Ambiguous>::value), "");
    assert(dpl::erf(0) == 0);
}

ONEDPL_TEST_NUM_TEMPLATE
void test_erfc()
{
    static_assert((std::is_same<decltype(dpl::erfc((float)0)), float>::value), "");
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::erfc((bool)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::erfc((unsigned short)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::erfc((int)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::erfc((unsigned int)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::erfc((long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::erfc((unsigned long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::erfc((long long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::erfc((unsigned long long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::erfc((double)0)), double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::erfc((long double)0)), long double>::value), ""))
    static_assert((std::is_same<decltype(dpl::erfcf(0)), float>::value), "");
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::erfcl(0)), long double>::value), ""))
    static_assert((std::is_same<decltype(erfc(Ambiguous())), Ambiguous>::value), "");
    assert(dpl::erfc(0) == 1);
}

ONEDPL_TEST_NUM_TEMPLATE
void test_exp2()
{
    static_assert((std::is_same<decltype(dpl::exp2((float)0)), float>::value), "");
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::exp2((bool)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::exp2((unsigned short)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::exp2((int)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::exp2((unsigned int)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::exp2((long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::exp2((unsigned long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::exp2((long long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::exp2((unsigned long long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::exp2((double)0)), double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::exp2((long double)0)), long double>::value), ""))
    static_assert((std::is_same<decltype(dpl::exp2f(0)), float>::value), "");
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::exp2l(0)), long double>::value), ""))
    static_assert((std::is_same<decltype(exp2(Ambiguous())), Ambiguous>::value), "");
    assert(dpl::exp2(1) == 2);
}

ONEDPL_TEST_NUM_TEMPLATE
void test_expm1()
{
    static_assert((std::is_same<decltype(dpl::expm1((float)0)), float>::value), "");
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::expm1((bool)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::expm1((unsigned short)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::expm1((int)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::expm1((unsigned int)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::expm1((long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::expm1((unsigned long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::expm1((long long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::expm1((unsigned long long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::expm1((double)0)), double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::expm1((long double)0)), long double>::value), ""))
    static_assert((std::is_same<decltype(dpl::expm1f(0)), float>::value), "");
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::expm1l(0)), long double>::value), ""))
    static_assert((std::is_same<decltype(expm1(Ambiguous())), Ambiguous>::value), "");
    assert(dpl::expm1(0) == 0);
}

ONEDPL_TEST_NUM_TEMPLATE
void test_fdim()
{
    static_assert((std::is_same<decltype(dpl::fdim((float)0, (float)0)), float>::value), "");
    static_assert((std::is_same<decltype(dpl::fdim((bool)0, (float)0)), double>::value), "");
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::fdim((unsigned short)0, (double)0)), double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::fdim((int)0, (long double)0)), long double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::fdim((float)0, (unsigned int)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::fdim((double)0, (long)0)), double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::fdim((long double)0, (unsigned long)0)), long double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::fdim((int)0, (long long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::fdim((int)0, (unsigned long long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::fdim((double)0, (double)0)), double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::fdim((long double)0, (long double)0)), long double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::fdim((float)0, (double)0)), double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::fdim((float)0, (long double)0)), long double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::fdim((double)0, (long double)0)), long double>::value), ""))
    static_assert((std::is_same<decltype(dpl::fdimf(0,0)), float>::value), "");
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::fdiml(0,0)), long double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::fdim((int)0, (int)0)), double>::value), ""))
    static_assert((std::is_same<decltype(fdim(Ambiguous(), Ambiguous())), Ambiguous>::value), "");
    assert(dpl::fdim(1,0) == 1);
}

ONEDPL_TEST_NUM_TEMPLATE
void test_fma()
{
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::fma((bool)0, (float)0, (float)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::fma((char)0, (float)0, (float)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::fma((unsigned)0, (float)0, (float)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::fma((float)0, (int)0, (float)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::fma((float)0, (long)0, (float)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::fma((float)0, (float)0, (unsigned long long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::fma((float)0, (float)0, (double)0)), double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::fma((float)0, (float)0, (long double)0)), long double>::value), ""))
    static_assert((std::is_same<decltype(dpl::fma((float)0, (float)0, (float)0)), float>::value), "");

    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::fma((bool)0, (double)0, (double)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::fma((char)0, (double)0, (double)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::fma((unsigned)0, (double)0, (double)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::fma((double)0, (int)0, (double)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::fma((double)0, (long)0, (double)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::fma((double)0, (double)0, (unsigned long long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::fma((double)0, (double)0, (float)0)), double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::fma((double)0, (double)0, (long double)0)), long double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::fma((double)0, (double)0,  (double)0)), double>::value), ""))

    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::fma((bool)0, (long double)0, (long double)0)), long double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::fma((char)0, (long double)0, (long double)0)), long double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::fma((unsigned)0, (long double)0, (long double)0)), long double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::fma((long double)0, (int)0, (long double)0)), long double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::fma((long double)0, (long)0, (long double)0)), long double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::fma((long double)0, (long double)0, (unsigned long long)0)), long double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::fma((long double)0, (long double)0, (float)0)), long double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::fma((double)0, (long double)0, (long double)0)), long double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::fma((long double)0, (long double)0, (long double)0)), long double>::value), ""))

    static_assert((std::is_same<decltype(dpl::fmaf(0,0,0)), float>::value), "");
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::fmal(0,0,0)), long double>::value), ""))
    static_assert((std::is_same<decltype(fma(Ambiguous(), Ambiguous(), Ambiguous())), Ambiguous>::value), "");
    assert(dpl::fma(1,1,1) == 2);
}

ONEDPL_TEST_NUM_TEMPLATE
void test_fmax()
{
    static_assert((std::is_same<decltype(dpl::fmax((float)0, (float)0)), float>::value), "");
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::fmax((bool)0, (float)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::fmax((unsigned short)0, (double)0)), double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::fmax((int)0, (long double)0)), long double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::fmax((float)0, (unsigned int)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::fmax((double)0, (long)0)), double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::fmax((long double)0, (unsigned long)0)), long double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::fmax((int)0, (long long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::fmax((int)0, (unsigned long long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::fmax((double)0, (double)0)), double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::fmax((long double)0, (long double)0)), long double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::fmax((float)0, (double)0)), double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::fmax((float)0, (long double)0)), long double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::fmax((double)0, (long double)0)), long double>::value), ""))
    static_assert((std::is_same<decltype(dpl::fmaxf(0,0)), float>::value), "");
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::fmaxl(0,0)), long double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::fmax((int)0, (int)0)), double>::value), ""))
    static_assert((std::is_same<decltype(fmax(Ambiguous(), Ambiguous())), Ambiguous>::value), "");
    assert(dpl::fmax(1,0) == 1);
}

ONEDPL_TEST_NUM_TEMPLATE
void test_fmin()
{
    static_assert((std::is_same<decltype(dpl::fmin((float)0, (float)0)), float>::value), "");
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::fmin((bool)0, (float)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::fmin((unsigned short)0, (double)0)), double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::fmin((int)0, (long double)0)), long double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::fmin((float)0, (unsigned int)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::fmin((double)0, (long)0)), double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::fmin((long double)0, (unsigned long)0)), long double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::fmin((int)0, (long long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::fmin((int)0, (unsigned long long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::fmin((double)0, (double)0)), double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::fmin((long double)0, (long double)0)), long double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::fmin((float)0, (double)0)), double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::fmin((float)0, (long double)0)), long double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::fmin((double)0, (long double)0)), long double>::value), ""))
    static_assert((std::is_same<decltype(dpl::fminf(0,0)), float>::value), "");
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::fminl(0,0)), long double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::fmin((int)0, (int)0)), double>::value), ""))
    static_assert((std::is_same<decltype(fmin(Ambiguous(), Ambiguous())), Ambiguous>::value), "");
    assert(dpl::fmin(1,0) == 0);
}

ONEDPL_TEST_NUM_TEMPLATE
void test_hypot()
{
    static_assert((std::is_same<decltype(dpl::hypot((float)0, (float)0)), float>::value), "");
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::hypot((bool)0, (float)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::hypot((unsigned short)0, (double)0)), double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::hypot((int)0, (long double)0)), long double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::hypot((float)0, (unsigned int)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::hypot((double)0, (long)0)), double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::hypot((long double)0, (unsigned long)0)), long double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::hypot((int)0, (long long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::hypot((int)0, (unsigned long long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::hypot((double)0, (double)0)), double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::hypot((long double)0, (long double)0)), long double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::hypot((float)0, (double)0)), double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::hypot((float)0, (long double)0)), long double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::hypot((double)0, (long double)0)), long double>::value), ""))
    static_assert((std::is_same<decltype(dpl::hypotf(0,0)), float>::value), "");
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::hypotl(0,0)), long double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::hypot((int)0, (int)0)), double>::value), ""))
    static_assert((std::is_same<decltype(hypot(Ambiguous(), Ambiguous())), Ambiguous>::value), "");
    assert(dpl::hypot(3,4) == 5);

#if TEST_STD_VER > 14
    static_assert((std::is_same<decltype(dpl::hypot((float)0, (float)0, (float)0)), float>::value), "");
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::hypot((float)0, (bool)0, (float)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::hypot((float)0, (unsigned short)0, (double)0)), double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::hypot((float)0, (int)0, (long double)0)), long double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::hypot((float)0, (double)0, (long)0)), double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::hypot((float)0, (long double)0, (unsigned long)0)), long double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::hypot((float)0, (int)0, (long long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::hypot((float)0, (int)0, (unsigned long long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::hypot((float)0, (double)0, (double)0)), double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::hypot((float)0, (long double)0, (long double)0)), long double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::hypot((float)0, (float)0, (double)0)), double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::hypot((float)0, (float)0, (long double)0)), long double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::hypot((float)0, (double)0, (long double)0)), long double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::hypot((int)0, (int)0, (int)0)), double>::value), ""))
    static_assert((std::is_same<decltype(hypot(Ambiguous(), Ambiguous(), Ambiguous())), Ambiguous>::value), "");

    assert(dpl::hypot(2,3,6) == 7);
    assert(dpl::hypot(1,4,8) == 9);
#endif
}

ONEDPL_TEST_NUM_TEMPLATE
void test_ilogb()
{
    static_assert((std::is_same<decltype(dpl::ilogb((float)0)), int>::value), "");
    static_assert((std::is_same<decltype(dpl::ilogb((bool)0)), int>::value), "");
    static_assert((std::is_same<decltype(dpl::ilogb((unsigned short)0)), int>::value), "");
    static_assert((std::is_same<decltype(dpl::ilogb((int)0)), int>::value), "");
    static_assert((std::is_same<decltype(dpl::ilogb((unsigned int)0)), int>::value), "");
    static_assert((std::is_same<decltype(dpl::ilogb((long)0)), int>::value), "");
    static_assert((std::is_same<decltype(dpl::ilogb((unsigned long)0)), int>::value), "");
    static_assert((std::is_same<decltype(dpl::ilogb((long long)0)), int>::value), "");
    static_assert((std::is_same<decltype(dpl::ilogb((unsigned long long)0)), int>::value), "");
    static_assert((std::is_same<decltype(dpl::ilogb((double)0)), int>::value), "");
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::ilogb((long double)0)), int>::value), ""))
    static_assert((std::is_same<decltype(dpl::ilogbf(0)), int>::value), "");
    static_assert((std::is_same<decltype(dpl::ilogbl(0)), int>::value), "");
    static_assert((std::is_same<decltype(ilogb(Ambiguous())), Ambiguous>::value), "");
    assert(dpl::ilogb(1) == 0);
}

ONEDPL_TEST_NUM_TEMPLATE
void test_lgamma()
{
    static_assert((std::is_same<decltype(dpl::lgamma((float)0)), float>::value), "");
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::lgamma((bool)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::lgamma((unsigned short)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::lgamma((int)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::lgamma((unsigned int)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::lgamma((long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::lgamma((unsigned long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::lgamma((long long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::lgamma((unsigned long long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::lgamma((double)0)), double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::lgamma((long double)0)), long double>::value), ""))
    static_assert((std::is_same<decltype(dpl::lgammaf(0)), float>::value), "");
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::lgammal(0)), long double>::value), ""))
    static_assert((std::is_same<decltype(lgamma(Ambiguous())), Ambiguous>::value), "");
    assert(dpl::lgamma(1) == 0);
}

ONEDPL_TEST_NUM_TEMPLATE
void test_llrint()
{
    static_assert((std::is_same<decltype(dpl::llrint((float)0)), long long>::value), "");
    static_assert((std::is_same<decltype(dpl::llrint((bool)0)), long long>::value), "");
    static_assert((std::is_same<decltype(dpl::llrint((unsigned short)0)), long long>::value), "");
    static_assert((std::is_same<decltype(dpl::llrint((int)0)), long long>::value), "");
    static_assert((std::is_same<decltype(dpl::llrint((unsigned int)0)), long long>::value), "");
    static_assert((std::is_same<decltype(dpl::llrint((long)0)), long long>::value), "");
    static_assert((std::is_same<decltype(dpl::llrint((unsigned long)0)), long long>::value), "");
    static_assert((std::is_same<decltype(dpl::llrint((long long)0)), long long>::value), "");
    static_assert((std::is_same<decltype(dpl::llrint((unsigned long long)0)), long long>::value), "");
    static_assert((std::is_same<decltype(dpl::llrint((double)0)), long long>::value), "");
    static_assert((std::is_same<decltype(dpl::llrint((long double)0)), long long>::value), "");
    static_assert((std::is_same<decltype(dpl::llrintf(0)), long long>::value), "");
    static_assert((std::is_same<decltype(dpl::llrintl(0)), long long>::value), "");
    static_assert((std::is_same<decltype(llrint(Ambiguous())), Ambiguous>::value), "");
    assert(dpl::llrint(1) == 1LL);
}

ONEDPL_TEST_NUM_TEMPLATE
void test_llround()
{
    static_assert((std::is_same<decltype(dpl::llround((float)0)), long long>::value), "");
    static_assert((std::is_same<decltype(dpl::llround((bool)0)), long long>::value), "");
    static_assert((std::is_same<decltype(dpl::llround((unsigned short)0)), long long>::value), "");
    static_assert((std::is_same<decltype(dpl::llround((int)0)), long long>::value), "");
    static_assert((std::is_same<decltype(dpl::llround((unsigned int)0)), long long>::value), "");
    static_assert((std::is_same<decltype(dpl::llround((long)0)), long long>::value), "");
    static_assert((std::is_same<decltype(dpl::llround((unsigned long)0)), long long>::value), "");
    static_assert((std::is_same<decltype(dpl::llround((long long)0)), long long>::value), "");
    static_assert((std::is_same<decltype(dpl::llround((unsigned long long)0)), long long>::value), "");
    static_assert((std::is_same<decltype(dpl::llround((double)0)), long long>::value), "");
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::llround((long double)0)), long long>::value), ""))
    static_assert((std::is_same<decltype(dpl::llroundf(0)), long long>::value), "");
    static_assert((std::is_same<decltype(dpl::llroundl(0)), long long>::value), "");
    static_assert((std::is_same<decltype(llround(Ambiguous())), Ambiguous>::value), "");
    assert(dpl::llround(1) == 1LL);
}

ONEDPL_TEST_NUM_TEMPLATE
void test_log1p()
{
    static_assert((std::is_same<decltype(dpl::log1p((float)0)), float>::value), "");
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::log1p((bool)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::log1p((unsigned short)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::log1p((int)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::log1p((unsigned int)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::log1p((long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::log1p((unsigned long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::log1p((long long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::log1p((unsigned long long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::log1p((double)0)), double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::log1p((long double)0)), long double>::value), ""))
    static_assert((std::is_same<decltype(dpl::log1pf(0)), float>::value), "");
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::log1pl(0)), long double>::value), ""))
    static_assert((std::is_same<decltype(log1p(Ambiguous())), Ambiguous>::value), "");
    assert(dpl::log1p(0) == 0);
}

ONEDPL_TEST_NUM_TEMPLATE
void test_log2()
{
    static_assert((std::is_same<decltype(dpl::log2((float)0)), float>::value), "");
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::log2((bool)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::log2((unsigned short)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::log2((int)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::log2((unsigned int)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::log2((long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::log2((unsigned long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::log2((long long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::log2((unsigned long long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::log2((double)0)), double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::log2((long double)0)), long double>::value), ""))
    static_assert((std::is_same<decltype(dpl::log2f(0)), float>::value), "");
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::log2l(0)), long double>::value), ""))
    static_assert((std::is_same<decltype(log2(Ambiguous())), Ambiguous>::value), "");
    assert(dpl::log2(1) == 0);
}

ONEDPL_TEST_NUM_TEMPLATE
void test_logb()
{
    static_assert((std::is_same<decltype(dpl::logb((float)0)), float>::value), "");
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::logb((bool)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::logb((unsigned short)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::logb((int)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::logb((unsigned int)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::logb((long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::logb((unsigned long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::logb((long long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::logb((unsigned long long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::logb((double)0)), double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::logb((long double)0)), long double>::value), ""))
    static_assert((std::is_same<decltype(dpl::logbf(0)), float>::value), "");
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::logbl(0)), long double>::value), ""))
    static_assert((std::is_same<decltype(logb(Ambiguous())), Ambiguous>::value), "");
    assert(dpl::logb(1) == 0);
}

ONEDPL_TEST_NUM_TEMPLATE
void test_lrint()
{
    static_assert((std::is_same<decltype(dpl::lrint((float)0)), long>::value), "");
    static_assert((std::is_same<decltype(dpl::lrint((bool)0)), long>::value), "");
    static_assert((std::is_same<decltype(dpl::lrint((unsigned short)0)), long>::value), "");
    static_assert((std::is_same<decltype(dpl::lrint((int)0)), long>::value), "");
    static_assert((std::is_same<decltype(dpl::lrint((unsigned int)0)), long>::value), "");
    static_assert((std::is_same<decltype(dpl::lrint((long)0)), long>::value), "");
    static_assert((std::is_same<decltype(dpl::lrint((unsigned long)0)), long>::value), "");
    static_assert((std::is_same<decltype(dpl::lrint((long long)0)), long>::value), "");
    static_assert((std::is_same<decltype(dpl::lrint((unsigned long long)0)), long>::value), "");
    static_assert((std::is_same<decltype(dpl::lrint((double)0)), long>::value), "");
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::lrint((long double)0)), long>::value), ""))
    static_assert((std::is_same<decltype(dpl::lrintf(0)), long>::value), "");
    static_assert((std::is_same<decltype(dpl::lrintl(0)), long>::value), "");
    static_assert((std::is_same<decltype(lrint(Ambiguous())), Ambiguous>::value), "");
    assert(dpl::lrint(1) == 1L);
}

ONEDPL_TEST_NUM_TEMPLATE
void test_lround()
{
    static_assert((std::is_same<decltype(dpl::lround((float)0)), long>::value), "");
    static_assert((std::is_same<decltype(dpl::lround((bool)0)), long>::value), "");
    static_assert((std::is_same<decltype(dpl::lround((unsigned short)0)), long>::value), "");
    static_assert((std::is_same<decltype(dpl::lround((int)0)), long>::value), "");
    static_assert((std::is_same<decltype(dpl::lround((unsigned int)0)), long>::value), "");
    static_assert((std::is_same<decltype(dpl::lround((long)0)), long>::value), "");
    static_assert((std::is_same<decltype(dpl::lround((unsigned long)0)), long>::value), "");
    static_assert((std::is_same<decltype(dpl::lround((long long)0)), long>::value), "");
    static_assert((std::is_same<decltype(dpl::lround((unsigned long long)0)), long>::value), "");
    static_assert((std::is_same<decltype(dpl::lround((double)0)), long>::value), "");
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::lround((long double)0)), long>::value), ""))
    static_assert((std::is_same<decltype(dpl::lroundf(0)), long>::value), "");
    static_assert((std::is_same<decltype(dpl::lroundl(0)), long>::value), "");
    static_assert((std::is_same<decltype(lround(Ambiguous())), Ambiguous>::value), "");
    assert(dpl::lround(1) == 1L);
}

ONEDPL_TEST_NUM_TEMPLATE
void test_nan()
{
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::nan("")), double>::value), ""))
    static_assert((std::is_same<decltype(dpl::nanf("")), float>::value), "");
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::nanl("")), long double>::value), ""))
}

ONEDPL_TEST_NUM_TEMPLATE
void test_nearbyint()
{
    static_assert((std::is_same<decltype(dpl::nearbyint((float)0)), float>::value), "");
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::nearbyint((bool)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::nearbyint((unsigned short)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::nearbyint((int)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::nearbyint((unsigned int)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::nearbyint((long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::nearbyint((unsigned long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::nearbyint((long long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::nearbyint((unsigned long long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::nearbyint((double)0)), double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::nearbyint((long double)0)), long double>::value), ""))
    static_assert((std::is_same<decltype(dpl::nearbyintf(0)), float>::value), "");
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::nearbyintl(0)), long double>::value), ""))
    static_assert((std::is_same<decltype(nearbyint(Ambiguous())), Ambiguous>::value), "");
    assert(dpl::nearbyint(1) == 1);
}

ONEDPL_TEST_NUM_TEMPLATE
void test_nextafter()
{
    static_assert((std::is_same<decltype(dpl::nextafter((float)0, (float)0)), float>::value), "");
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::nextafter((bool)0, (float)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::nextafter((unsigned short)0, (double)0)), double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::nextafter((int)0, (long double)0)), long double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::nextafter((float)0, (unsigned int)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::nextafter((double)0, (long)0)), double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::nextafter((long double)0, (unsigned long)0)), long double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::nextafter((int)0, (long long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::nextafter((int)0, (unsigned long long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::nextafter((double)0, (double)0)), double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::nextafter((long double)0, (long double)0)), long double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::nextafter((float)0, (double)0)), double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::nextafter((float)0, (long double)0)), long double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::nextafter((double)0, (long double)0)), long double>::value), ""))
    static_assert((std::is_same<decltype(dpl::nextafterf(0,0)), float>::value), "");
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::nextafterl(0,0)), long double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::nextafter((int)0, (int)0)), double>::value), ""))
    static_assert((std::is_same<decltype(nextafter(Ambiguous(), Ambiguous())), Ambiguous>::value), "");
    assert(dpl::nextafter(0,1) == hexfloat<double>(0x1, 0, -1074));
}

ONEDPL_TEST_NUM_TEMPLATE
void test_nexttoward()
{
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::nexttoward((float)0, (long double)0)), float>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::nexttoward((bool)0, (long double)0)), double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::nexttoward((unsigned short)0, (long double)0)), double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::nexttoward((int)0, (long double)0)), double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::nexttoward((unsigned int)0, (long double)0)), double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::nexttoward((long)0, (long double)0)), double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::nexttoward((unsigned long)0, (long double)0)), double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::nexttoward((long long)0, (long double)0)), double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::nexttoward((unsigned long long)0, (long double)0)), double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::nexttoward((double)0, (long double)0)), double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::nexttoward((long double)0, (long double)0)), long double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::nexttowardf(0, (long double)0)), float>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::nexttowardl(0, (long double)0)), long double>::value), ""))
    static_assert((std::is_same<decltype(nexttoward(Ambiguous(), Ambiguous())), Ambiguous>::value), "");
    assert(dpl::nexttoward(0, 1) == hexfloat<double>(0x1, 0, -1074));
}

ONEDPL_TEST_NUM_TEMPLATE
void test_remainder()
{
    static_assert((std::is_same<decltype(dpl::remainder((float)0, (float)0)), float>::value), "");
    static_assert((std::is_same<decltype(dpl::remainder((bool)0, (float)0)), double>::value), "");
    static_assert((std::is_same<decltype(dpl::remainder((unsigned short)0, (double)0)), double>::value), "");
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::remainder((int)0, (long double)0)), long double>::value), ""))
    static_assert((std::is_same<decltype(dpl::remainder((float)0, (unsigned int)0)), double>::value), "");
    static_assert((std::is_same<decltype(dpl::remainder((double)0, (long)0)), double>::value), "");
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::remainder((long double)0, (unsigned long)0)), long double>::value), ""))
    static_assert((std::is_same<decltype(dpl::remainder((int)0, (long long)0)), double>::value), "");
    static_assert((std::is_same<decltype(dpl::remainder((int)0, (unsigned long long)0)), double>::value), "");
    static_assert((std::is_same<decltype(dpl::remainder((double)0, (double)0)), double>::value), "");
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::remainder((long double)0, (long double)0)), long double>::value), ""))
    static_assert((std::is_same<decltype(dpl::remainder((float)0, (double)0)), double>::value), "");
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::remainder((float)0, (long double)0)), long double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::remainder((double)0, (long double)0)), long double>::value), ""))
    static_assert((std::is_same<decltype(dpl::remainderf(0,0)), float>::value), "");
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::remainderl(0,0)), long double>::value), ""))
    static_assert((std::is_same<decltype(dpl::remainder((int)0, (int)0)), double>::value), "");
    static_assert((std::is_same<decltype(remainder(Ambiguous(), Ambiguous())), Ambiguous>::value), "");
    assert(dpl::remainder(0.5,1) == 0.5);
}

ONEDPL_TEST_NUM_TEMPLATE
void test_remquo()
{
    int ip;
    static_assert((std::is_same<decltype(dpl::remquo((float)0, (float)0, &ip)), float>::value), "");
    static_assert((std::is_same<decltype(dpl::remquo((bool)0, (float)0, &ip)), double>::value), "");
    static_assert((std::is_same<decltype(dpl::remquo((unsigned short)0, (double)0, &ip)), double>::value), "");
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::remquo((int)0, (long double)0, &ip)), long double>::value), ""))
    static_assert((std::is_same<decltype(dpl::remquo((float)0, (unsigned int)0, &ip)), double>::value), "");
    static_assert((std::is_same<decltype(dpl::remquo((double)0, (long)0, &ip)), double>::value), "");
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::remquo((long double)0, (unsigned long)0, &ip)), long double>::value), ""))
    static_assert((std::is_same<decltype(dpl::remquo((int)0, (long long)0, &ip)), double>::value), "");
    static_assert((std::is_same<decltype(dpl::remquo((int)0, (unsigned long long)0, &ip)), double>::value), "");
    static_assert((std::is_same<decltype(dpl::remquo((double)0, (double)0, &ip)), double>::value), "");
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::remquo((long double)0, (long double)0, &ip)), long double>::value), ""))
    static_assert((std::is_same<decltype(dpl::remquo((float)0, (double)0, &ip)), double>::value), "");
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::remquo((float)0, (long double)0, &ip)), long double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::remquo((double)0, (long double)0, &ip)), long double>::value), ""))
    static_assert((std::is_same<decltype(dpl::remquof(0,0, &ip)), float>::value), "");
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::remquol(0,0, &ip)), long double>::value), ""))
    static_assert((std::is_same<decltype(dpl::remquo((int)0, (int)0, &ip)), double>::value), "");
    static_assert((std::is_same<decltype(remquo(Ambiguous(), Ambiguous(), &ip)), Ambiguous>::value), "");
    assert(dpl::remquo(0.5,1, &ip) == 0.5);
}

ONEDPL_TEST_NUM_TEMPLATE
void test_rint()
{
    static_assert((std::is_same<decltype(dpl::rint((float)0)), float>::value), "");
    static_assert((std::is_same<decltype(dpl::rint((bool)0)), double>::value), "");
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::rint((unsigned short)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::rint((int)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::rint((unsigned int)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::rint((long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::rint((unsigned long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::rint((long long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::rint((unsigned long long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::rint((double)0)), double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::rint((long double)0)), long double>::value), ""))
    static_assert((std::is_same<decltype(dpl::rintf(0)), float>::value), "");
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::rintl(0)), long double>::value), ""))
    static_assert((std::is_same<decltype(rint(Ambiguous())), Ambiguous>::value), "");
    assert(dpl::rint(1) == 1);
}

ONEDPL_TEST_NUM_TEMPLATE
void test_round()
{
    static_assert((std::is_same<decltype(dpl::round((float)0)), float>::value), "");
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::round((bool)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::round((unsigned short)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::round((int)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::round((unsigned int)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::round((long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::round((unsigned long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::round((long long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::round((unsigned long long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::round((double)0)), double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::round((long double)0)), long double>::value), ""))
    static_assert((std::is_same<decltype(dpl::roundf(0)), float>::value), "");
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::roundl(0)), long double>::value), ""))
    static_assert((std::is_same<decltype(round(Ambiguous())), Ambiguous>::value), "");
    assert(dpl::round(1) == 1);
}

ONEDPL_TEST_NUM_TEMPLATE
void test_scalbln()
{
    static_assert((std::is_same<decltype(dpl::scalbln((float)0, (long)0)), float>::value), "");
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::scalbln((bool)0, (long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::scalbln((unsigned short)0, (long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::scalbln((int)0, (long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::scalbln((unsigned int)0, (long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::scalbln((long)0, (long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::scalbln((unsigned long)0, (long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::scalbln((long long)0, (long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::scalbln((unsigned long long)0, (long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::scalbln((double)0, (long)0)), double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::scalbln((long double)0, (long)0)), long double>::value), ""))
    static_assert((std::is_same<decltype(dpl::scalblnf(0, (long)0)), float>::value), "");
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::scalblnl(0, (long)0)), long double>::value), ""))
    static_assert((std::is_same<decltype(scalbln(Ambiguous(), Ambiguous())), Ambiguous>::value), "");
    assert(dpl::scalbln(1, 1) == 2);
}

ONEDPL_TEST_NUM_TEMPLATE
void test_scalbn()
{
    static_assert((std::is_same<decltype(dpl::scalbn((float)0, (int)0)), float>::value), "");
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::scalbn((bool)0, (int)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::scalbn((unsigned short)0, (int)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::scalbn((int)0, (int)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::scalbn((unsigned int)0, (int)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::scalbn((long)0, (int)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::scalbn((unsigned long)0, (int)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::scalbn((long long)0, (int)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::scalbn((unsigned long long)0, (int)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::scalbn((double)0, (int)0)), double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::scalbn((long double)0, (int)0)), long double>::value), ""))
    static_assert((std::is_same<decltype(dpl::scalbnf(0, (int)0)), float>::value), "");
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::scalbnl(0, (int)0)), long double>::value), ""))
    static_assert((std::is_same<decltype(scalbn(Ambiguous(), Ambiguous())), Ambiguous>::value), "");
    assert(dpl::scalbn(1, 1) == 2);
}

ONEDPL_TEST_NUM_TEMPLATE
void test_tgamma()
{
    static_assert((std::is_same<decltype(dpl::tgamma((float)0)), float>::value), "");
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::tgamma((bool)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::tgamma((unsigned short)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::tgamma((int)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::tgamma((unsigned int)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::tgamma((long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::tgamma((unsigned long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::tgamma((long long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::tgamma((unsigned long long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::tgamma((double)0)), double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::tgamma((long double)0)), long double>::value), ""))
    static_assert((std::is_same<decltype(dpl::tgammaf(0)), float>::value), "");
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::tgammal(0)), long double>::value), ""))
    static_assert((std::is_same<decltype(tgamma(Ambiguous())), Ambiguous>::value), "");
    assert(dpl::tgamma(1) == 1);
}

ONEDPL_TEST_NUM_TEMPLATE
void test_trunc()
{
    static_assert((std::is_same<decltype(dpl::trunc((float)0)), float>::value), "");
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::trunc((bool)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::trunc((unsigned short)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::trunc((int)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::trunc((unsigned int)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::trunc((long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::trunc((unsigned long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::trunc((long long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::trunc((unsigned long long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::trunc((double)0)), double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::trunc((long double)0)), long double>::value), ""))
    static_assert((std::is_same<decltype(dpl::truncf(0)), float>::value), "");
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::truncl(0)), long double>::value), ""))
    static_assert((std::is_same<decltype(trunc(Ambiguous())), Ambiguous>::value), "");
    assert(dpl::trunc(1) == 1);
}

ONEDPL_TEST_NUM_MAIN
{
    ONEDPL_TEST_NUM_CALL(test_abs)
    ONEDPL_TEST_NUM_CALL(test_acos)
    ONEDPL_TEST_NUM_CALL(test_asin)
    ONEDPL_TEST_NUM_CALL(test_atan)
    ONEDPL_TEST_NUM_CALL(test_atan2)
    ONEDPL_TEST_NUM_CALL(test_ceil)
    ONEDPL_TEST_NUM_CALL(test_cos)
    ONEDPL_TEST_NUM_CALL(test_cosh)
    ONEDPL_TEST_NUM_CALL(test_exp)
    ONEDPL_TEST_NUM_CALL(test_fabs)
    ONEDPL_TEST_NUM_CALL(test_floor)
    ONEDPL_TEST_NUM_CALL(test_fmod)
    ONEDPL_TEST_NUM_CALL(test_frexp)
    ONEDPL_TEST_NUM_CALL(test_ldexp)
    ONEDPL_TEST_NUM_CALL(test_log)
    ONEDPL_TEST_NUM_CALL(test_log10)
    ONEDPL_TEST_NUM_CALL(test_modf)
    ONEDPL_TEST_NUM_CALL(test_pow)
    ONEDPL_TEST_NUM_CALL(test_sin)
    ONEDPL_TEST_NUM_CALL(test_sinh)
    ONEDPL_TEST_NUM_CALL(test_sqrt)
    ONEDPL_TEST_NUM_CALL(test_tan)
    ONEDPL_TEST_NUM_CALL(test_tanh)
    ONEDPL_TEST_NUM_CALL(test_signbit)
    ONEDPL_TEST_NUM_CALL(test_fpclassify)
    ONEDPL_TEST_NUM_CALL(test_isfinite)
    ONEDPL_TEST_NUM_CALL(test_isnormal)
    ONEDPL_TEST_NUM_CALL(test_isgreater)
    ONEDPL_TEST_NUM_CALL(test_isgreaterequal)
    ONEDPL_TEST_NUM_CALL(test_isinf)
    ONEDPL_TEST_NUM_CALL(test_isless)
    ONEDPL_TEST_NUM_CALL(test_islessequal)
    ONEDPL_TEST_NUM_CALL(test_islessgreater)
    ONEDPL_TEST_NUM_CALL(test_isnan)
    ONEDPL_TEST_NUM_CALL(test_isunordered)
    ONEDPL_TEST_NUM_CALL(test_acosh)
    ONEDPL_TEST_NUM_CALL(test_asinh)
    ONEDPL_TEST_NUM_CALL(test_atanh)
    ONEDPL_TEST_NUM_CALL(test_cbrt)
    ONEDPL_TEST_NUM_CALL(test_copysign)
    ONEDPL_TEST_NUM_CALL(test_erf)
    ONEDPL_TEST_NUM_CALL(test_erfc)
    ONEDPL_TEST_NUM_CALL(test_exp2)
    ONEDPL_TEST_NUM_CALL(test_expm1)
    ONEDPL_TEST_NUM_CALL(test_fdim)
    ONEDPL_TEST_NUM_CALL(test_fma)
    ONEDPL_TEST_NUM_CALL(test_fmax)
    ONEDPL_TEST_NUM_CALL(test_fmin)
    ONEDPL_TEST_NUM_CALL(test_hypot)
    ONEDPL_TEST_NUM_CALL(test_ilogb)
    ONEDPL_TEST_NUM_CALL(test_lgamma)
    ONEDPL_TEST_NUM_CALL(test_llrint)
    ONEDPL_TEST_NUM_CALL(test_llround)
    ONEDPL_TEST_NUM_CALL(test_log1p)
    ONEDPL_TEST_NUM_CALL(test_log2)
    ONEDPL_TEST_NUM_CALL(test_logb)
    ONEDPL_TEST_NUM_CALL(test_lrint)
    ONEDPL_TEST_NUM_CALL(test_lround)
    ONEDPL_TEST_NUM_CALL(test_nan)
    ONEDPL_TEST_NUM_CALL(test_nearbyint)
    ONEDPL_TEST_NUM_CALL(test_nextafter)
    ONEDPL_TEST_NUM_CALL(test_nexttoward)
    ONEDPL_TEST_NUM_CALL(test_remainder)
    ONEDPL_TEST_NUM_CALL(test_remquo)
    ONEDPL_TEST_NUM_CALL(test_rint)
    ONEDPL_TEST_NUM_CALL(test_round)
    ONEDPL_TEST_NUM_CALL(test_scalbln)
    ONEDPL_TEST_NUM_CALL(test_scalbn)
    ONEDPL_TEST_NUM_CALL(test_tgamma)
    ONEDPL_TEST_NUM_CALL(test_trunc)
}
