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
//   conj(const complex<T>& x);

#include <complex>
#include <cassert>

#include "test_macros.h"

template <class T>
void
test(const dpl::complex<T>& z, dpl::complex<T> x)
{
    assert(conj(z) == x);
}

template <class T>
void
test()
{
    test(dpl::complex<T>(1, 2), dpl::complex<T>(1, -2));
    test(dpl::complex<T>(-1, 2), dpl::complex<T>(-1, -2));
    test(dpl::complex<T>(1, -2), dpl::complex<T>(1, 2));
    test(dpl::complex<T>(-1, -2), dpl::complex<T>(-1, 2));
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
