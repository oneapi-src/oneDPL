//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <complex>

// complex& operator-=(const complex& rhs);

#include <complex>
#include <cassert>

#include "test_macros.h"

template <class T>
void
test()
{
    dpl::complex<T> c;
    const dpl::complex<T> c2(1.5, 2.5);
    assert(c.real() == 0);
    assert(c.imag() == 0);
    c -= c2;
    assert(c.real() == -1.5);
    assert(c.imag() == -2.5);
    c -= c2;
    assert(c.real() == -3);
    assert(c.imag() == -5);

    dpl::complex<T> c3;

    c3 = c;
    dpl::complex<int> ic (1,1);
    c3 -= ic;
    assert(c3.real() == -4);
    assert(c3.imag() == -6);

    c3 = c;
    dpl::complex<float> fc (1,1);
    c3 -= fc;
    assert(c3.real() == -4);
    assert(c3.imag() == -6);
}

void run_test()
{
    test<float>();
    test<double>();
    test<long double>();
}

int main(int, char**)
{
    run_test();

  return 0;
}
