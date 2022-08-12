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
//   operator+(const complex<T>& lhs, const T& rhs);

#include <complex>
#include <cassert>

#include "test_macros.h"

template <class T>
void
test(const dpl::complex<T>& lhs, const T& rhs, dpl::complex<T> x)
{
    assert(lhs + rhs == x);
}

template <class T>
void
test()
{
    {
    dpl::complex<T> lhs(1.5, 2.5);
    T rhs(3.5);
    dpl::complex<T>   x(5.0, 2.5);
    test(lhs, rhs, x);
    }
    {
    dpl::complex<T> lhs(1.5, -2.5);
    T rhs(-3.5);
    dpl::complex<T>   x(-2.0, -2.5);
    test(lhs, rhs, x);
    }
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
