//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <complex>

// test cases

#ifndef _CASES_H
#define _CASES_H

#include <oneapi/dpl/complex>
#include <cassert>
#include <type_traits>

const dpl::complex<double> testcases[] =
{
    dpl::complex<double>( 1.e-6,  1.e-6),       // 0
    dpl::complex<double>(-1.e-6,  1.e-6),       // 1
    dpl::complex<double>(-1.e-6, -1.e-6),       // 2
    dpl::complex<double>( 1.e-6, -1.e-6),       // 3

    dpl::complex<double>( 1.e+6,  1.e-6),       // 4
    dpl::complex<double>(-1.e+6,  1.e-6),       // 5
    dpl::complex<double>(-1.e+6, -1.e-6),       // 6
    dpl::complex<double>( 1.e+6, -1.e-6),       // 7

    dpl::complex<double>( 1.e-6,  1.e+6),       // 8
    dpl::complex<double>(-1.e-6,  1.e+6),       // 9
    dpl::complex<double>(-1.e-6, -1.e+6),       // 10
    dpl::complex<double>( 1.e-6, -1.e+6),       // 11

    dpl::complex<double>( 1.e+6,  1.e+6),       // 12
    dpl::complex<double>(-1.e+6,  1.e+6),       // 13
    dpl::complex<double>(-1.e+6, -1.e+6),       // 14
    dpl::complex<double>( 1.e+6, -1.e+6),       // 15

    dpl::complex<double>(-0, -1.e-6),           // 16
    dpl::complex<double>(-0,  1.e-6),           // 17
    dpl::complex<double>(-0,  1.e+6),           // 18
    dpl::complex<double>(-0, -1.e+6),           // 19
    dpl::complex<double>( 0, -1.e-6),           // 20
    dpl::complex<double>( 0,  1.e-6),           // 21
    dpl::complex<double>( 0,  1.e+6),           // 22
    dpl::complex<double>( 0, -1.e+6),           // 23

    dpl::complex<double>(-1.e-6, -0),           // 24
    dpl::complex<double>( 1.e-6, -0),           // 25
    dpl::complex<double>( 1.e+6, -0),           // 26
    dpl::complex<double>(-1.e+6, -0),           // 27
    dpl::complex<double>(-1.e-6,  0),           // 28
    dpl::complex<double>( 1.e-6,  0),           // 29
    dpl::complex<double>( 1.e+6,  0),           // 30
    dpl::complex<double>(-1.e+6,  0),           // 31

    dpl::complex<double>(NAN, NAN),             // 32
    dpl::complex<double>(-INFINITY, NAN),       // 33
    dpl::complex<double>(-2, NAN),              // 34
    dpl::complex<double>(-1, NAN),              // 35
    dpl::complex<double>(-0.5, NAN),            // 36
    dpl::complex<double>(-0., NAN),             // 37
    dpl::complex<double>(+0., NAN),             // 38
    dpl::complex<double>(0.5, NAN),             // 39
    dpl::complex<double>(1, NAN),               // 40
    dpl::complex<double>(2, NAN),               // 41
    dpl::complex<double>(INFINITY, NAN),        // 42

    dpl::complex<double>(NAN, -INFINITY),       // 43
    dpl::complex<double>(-INFINITY, -INFINITY), // 44
    dpl::complex<double>(-2, -INFINITY),        // 45
    dpl::complex<double>(-1, -INFINITY),        // 46
    dpl::complex<double>(-0.5, -INFINITY),      // 47
    dpl::complex<double>(-0., -INFINITY),       // 48
    dpl::complex<double>(+0., -INFINITY),       // 49
    dpl::complex<double>(0.5, -INFINITY),       // 50
    dpl::complex<double>(1, -INFINITY),         // 51
    dpl::complex<double>(2, -INFINITY),         // 52
    dpl::complex<double>(INFINITY, -INFINITY),  // 53

    dpl::complex<double>(NAN, -2),              // 54
    dpl::complex<double>(-INFINITY, -2),        // 55
    dpl::complex<double>(-2, -2),               // 56
    dpl::complex<double>(-1, -2),               // 57
    dpl::complex<double>(-0.5, -2),             // 58
    dpl::complex<double>(-0., -2),              // 59
    dpl::complex<double>(+0., -2),              // 60
    dpl::complex<double>(0.5, -2),              // 61
    dpl::complex<double>(1, -2),                // 62
    dpl::complex<double>(2, -2),                // 63
    dpl::complex<double>(INFINITY, -2),         // 64

    dpl::complex<double>(NAN, -1),              // 65
    dpl::complex<double>(-INFINITY, -1),        // 66
    dpl::complex<double>(-2, -1),               // 67
    dpl::complex<double>(-1, -1),               // 68
    dpl::complex<double>(-0.5, -1),             // 69
    dpl::complex<double>(-0., -1),              // 70
    dpl::complex<double>(+0., -1),              // 71
    dpl::complex<double>(0.5, -1),              // 72
    dpl::complex<double>(1, -1),                // 73
    dpl::complex<double>(2, -1),                // 74
    dpl::complex<double>(INFINITY, -1),         // 75

    dpl::complex<double>(NAN, -0.5),            // 76
    dpl::complex<double>(-INFINITY, -0.5),      // 77
    dpl::complex<double>(-2, -0.5),             // 78
    dpl::complex<double>(-1, -0.5),             // 79
    dpl::complex<double>(-0.5, -0.5),           // 80
    dpl::complex<double>(-0., -0.5),            // 81
    dpl::complex<double>(+0., -0.5),            // 82
    dpl::complex<double>(0.5, -0.5),            // 83
    dpl::complex<double>(1, -0.5),              // 84
    dpl::complex<double>(2, -0.5),              // 85
    dpl::complex<double>(INFINITY, -0.5),       // 86

    dpl::complex<double>(NAN, -0.),             // 87
    dpl::complex<double>(-INFINITY, -0.),       // 88
    dpl::complex<double>(-2, -0.),              // 89
    dpl::complex<double>(-1, -0.),              // 90
    dpl::complex<double>(-0.5, -0.),            // 91
    dpl::complex<double>(-0., -0.),             // 92
    dpl::complex<double>(+0., -0.),             // 93
    dpl::complex<double>(0.5, -0.),             // 94
    dpl::complex<double>(1, -0.),               // 95
    dpl::complex<double>(2, -0.),               // 96
    dpl::complex<double>(INFINITY, -0.),        // 97

    dpl::complex<double>(NAN, +0.),             // 98
    dpl::complex<double>(-INFINITY, +0.),       // 99
    dpl::complex<double>(-2, +0.),              // 100
    dpl::complex<double>(-1, +0.),              // 101
    dpl::complex<double>(-0.5, +0.),            // 102
    dpl::complex<double>(-0., +0.),             // 103
    dpl::complex<double>(+0., +0.),             // 104
    dpl::complex<double>(0.5, +0.),             // 105
    dpl::complex<double>(1, +0.),               // 106
    dpl::complex<double>(2, +0.),               // 107
    dpl::complex<double>(INFINITY, +0.),        // 108

    dpl::complex<double>(NAN, 0.5),             // 109
    dpl::complex<double>(-INFINITY, 0.5),       // 110
    dpl::complex<double>(-2, 0.5),              // 111
    dpl::complex<double>(-1, 0.5),              // 112
    dpl::complex<double>(-0.5, 0.5),            // 113
    dpl::complex<double>(-0., 0.5),             // 114
    dpl::complex<double>(+0., 0.5),             // 115
    dpl::complex<double>(0.5, 0.5),             // 116
    dpl::complex<double>(1, 0.5),               // 117
    dpl::complex<double>(2, 0.5),               // 118
    dpl::complex<double>(INFINITY, 0.5),        // 119

    dpl::complex<double>(NAN, 1),               // 120
    dpl::complex<double>(-INFINITY, 1),         // 121
    dpl::complex<double>(-2, 1),                // 122
    dpl::complex<double>(-1, 1),                // 123
    dpl::complex<double>(-0.5, 1),              // 124
    dpl::complex<double>(-0., 1),               // 125
    dpl::complex<double>(+0., 1),               // 126
    dpl::complex<double>(0.5, 1),               // 127
    dpl::complex<double>(1, 1),                 // 128
    dpl::complex<double>(2, 1),                 // 129
    dpl::complex<double>(INFINITY, 1),          // 130

    dpl::complex<double>(NAN, 2),               // 131
    dpl::complex<double>(-INFINITY, 2),         // 132
    dpl::complex<double>(-2, 2),                // 133
    dpl::complex<double>(-1, 2),                // 134
    dpl::complex<double>(-0.5, 2),              // 135
    dpl::complex<double>(-0., 2),               // 136
    dpl::complex<double>(+0., 2),               // 137
    dpl::complex<double>(0.5, 2),               // 138
    dpl::complex<double>(1, 2),                 // 139
    dpl::complex<double>(2, 2),                 // 140
    dpl::complex<double>(INFINITY, 2),          // 141

    dpl::complex<double>(NAN, INFINITY),        // 142
    dpl::complex<double>(-INFINITY, INFINITY),  // 143
    dpl::complex<double>(-2, INFINITY),         // 144
    dpl::complex<double>(-1, INFINITY),         // 145
    dpl::complex<double>(-0.5, INFINITY),       // 146
    dpl::complex<double>(-0., INFINITY),        // 147
    dpl::complex<double>(+0., INFINITY),        // 148
    dpl::complex<double>(0.5, INFINITY),        // 149
    dpl::complex<double>(1, INFINITY),          // 150
    dpl::complex<double>(2, INFINITY),          // 151
    dpl::complex<double>(INFINITY, INFINITY)    // 152
};

enum {zero, non_zero, inf, NaN, non_zero_nan};

template <class T>
int
classify(const dpl::complex<T>& x)
{
    if (x == dpl::complex<T>())
        return zero;
    if (std::isinf(x.real()) || std::isinf(x.imag()))
        return inf;
    if (std::isnan(x.real()) && std::isnan(x.imag()))
        return NaN;
    if (std::isnan(x.real()))
    {
        if (x.imag() == T(0))
            return NaN;
        return non_zero_nan;
    }
    if (std::isnan(x.imag()))
    {
        if (x.real() == T(0))
            return NaN;
        return non_zero_nan;
    }
    return non_zero;
}

inline
int
classify(double x)
{
    if (x == 0)
        return zero;
    if (std::isinf(x))
        return inf;
    if (std::isnan(x))
        return NaN;
    return non_zero;
}

template <typename T>
constexpr auto __tol = std::numeric_limits<T>::epsilon() * 1e5;

template <typename X, typename Y>
::std::enable_if_t<!std::numeric_limits<X>::is_integer || !std::numeric_limits<Y>::is_integer>
is_about(X x, Y y, const X eps = __tol<X>)
{
    assert(std::fabs(x - y) <= eps);
}

template <typename T>
::std::enable_if_t<!std::numeric_limits<T>::is_integer>
is_about(const dpl::complex<T>& x, const dpl::complex<T>& y, const T eps = __tol<T>)
{
    return is_about(::std::abs(y - x), T(0.), eps);
}

#endif // _CASES_H
