//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <complex>

// test cases

#ifndef CASES_H
#define CASES_H

#include <oneapi/dpl/complex>
#include <oneapi/dpl/cmath>
#include <cassert>

const dpl::complex<double> testcases[] =
{
    dpl::complex<double>( 1.e-6,  1.e-6),
    dpl::complex<double>(-1.e-6,  1.e-6),
    dpl::complex<double>(-1.e-6, -1.e-6),
    dpl::complex<double>( 1.e-6, -1.e-6),

    dpl::complex<double>( 1.e+6,  1.e-6),
    dpl::complex<double>(-1.e+6,  1.e-6),
    dpl::complex<double>(-1.e+6, -1.e-6),
    dpl::complex<double>( 1.e+6, -1.e-6),

    dpl::complex<double>( 1.e-6,  1.e+6),
    dpl::complex<double>(-1.e-6,  1.e+6),
    dpl::complex<double>(-1.e-6, -1.e+6),
    dpl::complex<double>( 1.e-6, -1.e+6),

    dpl::complex<double>( 1.e+6,  1.e+6),
    dpl::complex<double>(-1.e+6,  1.e+6),
    dpl::complex<double>(-1.e+6, -1.e+6),
    dpl::complex<double>( 1.e+6, -1.e+6),

    dpl::complex<double>(-0, -1.e-6),
    dpl::complex<double>(-0,  1.e-6),
    dpl::complex<double>(-0,  1.e+6),
    dpl::complex<double>(-0, -1.e+6),
    dpl::complex<double>( 0, -1.e-6),
    dpl::complex<double>( 0,  1.e-6),
    dpl::complex<double>( 0,  1.e+6),
    dpl::complex<double>( 0, -1.e+6),

    dpl::complex<double>(-1.e-6, -0),
    dpl::complex<double>( 1.e-6, -0),
    dpl::complex<double>( 1.e+6, -0),
    dpl::complex<double>(-1.e+6, -0),
    dpl::complex<double>(-1.e-6,  0),
    dpl::complex<double>( 1.e-6,  0),
    dpl::complex<double>( 1.e+6,  0),
    dpl::complex<double>(-1.e+6,  0),

    dpl::complex<double>(NAN, NAN),
    dpl::complex<double>(-INFINITY, NAN),
    dpl::complex<double>(-2, NAN),
    dpl::complex<double>(-1, NAN),
    dpl::complex<double>(-0.5, NAN),
    dpl::complex<double>(-0., NAN),
    dpl::complex<double>(+0., NAN),
    dpl::complex<double>(0.5, NAN),
    dpl::complex<double>(1, NAN),
    dpl::complex<double>(2, NAN),
    dpl::complex<double>(INFINITY, NAN),

    dpl::complex<double>(NAN, -INFINITY),
    dpl::complex<double>(-INFINITY, -INFINITY),
    dpl::complex<double>(-2, -INFINITY),
    dpl::complex<double>(-1, -INFINITY),
    dpl::complex<double>(-0.5, -INFINITY),
    dpl::complex<double>(-0., -INFINITY),
    dpl::complex<double>(+0., -INFINITY),
    dpl::complex<double>(0.5, -INFINITY),
    dpl::complex<double>(1, -INFINITY),
    dpl::complex<double>(2, -INFINITY),
    dpl::complex<double>(INFINITY, -INFINITY),

    dpl::complex<double>(NAN, -2),
    dpl::complex<double>(-INFINITY, -2),
    dpl::complex<double>(-2, -2),
    dpl::complex<double>(-1, -2),
    dpl::complex<double>(-0.5, -2),
    dpl::complex<double>(-0., -2),
    dpl::complex<double>(+0., -2),
    dpl::complex<double>(0.5, -2),
    dpl::complex<double>(1, -2),
    dpl::complex<double>(2, -2),
    dpl::complex<double>(INFINITY, -2),

    dpl::complex<double>(NAN, -1),
    dpl::complex<double>(-INFINITY, -1),
    dpl::complex<double>(-2, -1),
    dpl::complex<double>(-1, -1),
    dpl::complex<double>(-0.5, -1),
    dpl::complex<double>(-0., -1),
    dpl::complex<double>(+0., -1),
    dpl::complex<double>(0.5, -1),
    dpl::complex<double>(1, -1),
    dpl::complex<double>(2, -1),
    dpl::complex<double>(INFINITY, -1),

    dpl::complex<double>(NAN, -0.5),
    dpl::complex<double>(-INFINITY, -0.5),
    dpl::complex<double>(-2, -0.5),
    dpl::complex<double>(-1, -0.5),
    dpl::complex<double>(-0.5, -0.5),
    dpl::complex<double>(-0., -0.5),
    dpl::complex<double>(+0., -0.5),
    dpl::complex<double>(0.5, -0.5),
    dpl::complex<double>(1, -0.5),
    dpl::complex<double>(2, -0.5),
    dpl::complex<double>(INFINITY, -0.5),

    dpl::complex<double>(NAN, -0.),
    dpl::complex<double>(-INFINITY, -0.),
    dpl::complex<double>(-2, -0.),
    dpl::complex<double>(-1, -0.),
    dpl::complex<double>(-0.5, -0.),
    dpl::complex<double>(-0., -0.),
    dpl::complex<double>(+0., -0.),
    dpl::complex<double>(0.5, -0.),
    dpl::complex<double>(1, -0.),
    dpl::complex<double>(2, -0.),
    dpl::complex<double>(INFINITY, -0.),

    dpl::complex<double>(NAN, +0.),
    dpl::complex<double>(-INFINITY, +0.),
    dpl::complex<double>(-2, +0.),
    dpl::complex<double>(-1, +0.),
    dpl::complex<double>(-0.5, +0.),
    dpl::complex<double>(-0., +0.),
    dpl::complex<double>(+0., +0.),
    dpl::complex<double>(0.5, +0.),
    dpl::complex<double>(1, +0.),
    dpl::complex<double>(2, +0.),
    dpl::complex<double>(INFINITY, +0.),

    dpl::complex<double>(NAN, 0.5),
    dpl::complex<double>(-INFINITY, 0.5),
    dpl::complex<double>(-2, 0.5),
    dpl::complex<double>(-1, 0.5),
    dpl::complex<double>(-0.5, 0.5),
    dpl::complex<double>(-0., 0.5),
    dpl::complex<double>(+0., 0.5),
    dpl::complex<double>(0.5, 0.5),
    dpl::complex<double>(1, 0.5),
    dpl::complex<double>(2, 0.5),
    dpl::complex<double>(INFINITY, 0.5),

    dpl::complex<double>(NAN, 1),
    dpl::complex<double>(-INFINITY, 1),
    dpl::complex<double>(-2, 1),
    dpl::complex<double>(-1, 1),
    dpl::complex<double>(-0.5, 1),
    dpl::complex<double>(-0., 1),
    dpl::complex<double>(+0., 1),
    dpl::complex<double>(0.5, 1),
    dpl::complex<double>(1, 1),
    dpl::complex<double>(2, 1),
    dpl::complex<double>(INFINITY, 1),

    dpl::complex<double>(NAN, 2),
    dpl::complex<double>(-INFINITY, 2),
    dpl::complex<double>(-2, 2),
    dpl::complex<double>(-1, 2),
    dpl::complex<double>(-0.5, 2),
    dpl::complex<double>(-0., 2),
    dpl::complex<double>(+0., 2),
    dpl::complex<double>(0.5, 2),
    dpl::complex<double>(1, 2),
    dpl::complex<double>(2, 2),
    dpl::complex<double>(INFINITY, 2),

    dpl::complex<double>(NAN, INFINITY),
    dpl::complex<double>(-INFINITY, INFINITY),
    dpl::complex<double>(-2, INFINITY),
    dpl::complex<double>(-1, INFINITY),
    dpl::complex<double>(-0.5, INFINITY),
    dpl::complex<double>(-0., INFINITY),
    dpl::complex<double>(+0., INFINITY),
    dpl::complex<double>(0.5, INFINITY),
    dpl::complex<double>(1, INFINITY),
    dpl::complex<double>(2, INFINITY),
    dpl::complex<double>(INFINITY, INFINITY)
};

enum {zero, non_zero, inf, NaN, non_zero_nan};

template <class T>
int
classify(const dpl::complex<T>& x)
{
// Suppress clang warning: comparison with infinity always evaluates to false in fast floating point modes [-Wtautological-constant-compare]
CLANG_DIAGNOSTIC_PUSH
CLANG_DIAGNOSTIC_IGNORED_AUTOLOGICAL_CONSTANT_COMPARE

    if (x == dpl::complex<T>())
        return zero;
    if (dpl::isinf(x.real()) || dpl::isinf(x.imag()))
        return inf;
    if (dpl::isnan(x.real()) && dpl::isnan(x.imag()))
        return NaN;
    if (dpl::isnan(x.real()))
    {
        if (x.imag() == T(0))
            return NaN;
        return non_zero_nan;
    }
    if (dpl::isnan(x.imag()))
    {
        if (x.real() == T(0))
            return NaN;
        return non_zero_nan;
    }
    return non_zero;

CLANG_DIAGNOSTIC_POP
}

inline
int
classify(double x)
{
// Suppress clang warning: comparison with infinity always evaluates to false in fast floating point modes [-Wtautological-constant-compare]
CLANG_DIAGNOSTIC_PUSH
CLANG_DIAGNOSTIC_IGNORED_AUTOLOGICAL_CONSTANT_COMPARE

    if (x == 0)
        return zero;
    if (dpl::isinf(x))
        return inf;
    if (dpl::isnan(x))
        return NaN;
    return non_zero;

CLANG_DIAGNOSTIC_POP
}

void is_about(float x, float y)
{
// Suppress clang warning: floating-point comparison is always true; constant cannot be represented exactly in type 'float' [-Wliteral-range]
CLANG_DIAGNOSTIC_PUSH
CLANG_DIAGNOSTIC_IGNORED_LITERAL_RANGE

    assert(dpl::abs((x-y)/(x+y)) < 1.e-6);

CLANG_DIAGNOSTIC_POP
}

void is_about(double x, double y)
{
    assert(dpl::abs((x-y)/(x+y)) < 1.e-14);
}

void is_about(long double x, long double y)
{
    assert(dpl::abs((x-y)/(x+y)) < 1.e-14);
}

#endif // CASES_H
