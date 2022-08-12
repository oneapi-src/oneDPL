//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <complex>

// template<class T>
//   complex<T>
//   operator+(const complex<T>&);

#include <complex>
#include <cassert>

#include "test_macros.h"

template <class T>
void
test()
{
    dpl::complex<T> z(1.5, 2.5);
    assert(z.real() == 1.5);
    assert(z.imag() == 2.5);
    dpl::complex<T> c = +z;
    assert(c.real() == 1.5);
    assert(c.imag() == 2.5);
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
