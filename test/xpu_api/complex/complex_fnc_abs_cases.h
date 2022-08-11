//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <complex>

// test cases

#ifndef __COMPLEX_ABS_CASES_H
#define __COMPLEX_ABS_CASES_H

#include <oneapi/dpl/complex>

const dpl::complex<double> testcases[] =
{
    dpl::complex<double>( 1.e-6,  1.e-6),
    dpl::complex<double>(-2, -2)
};

#endif  // __COMPLEX_ABS_CASES_H
