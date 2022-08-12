//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// XFAIL: LIBCXX-AIX-FIXME

// <complex>

// template<Arithmetic T>
//   T
//   arg(T x);

#include <complex>
#include <type_traits>
#include <cassert>

#include "test_macros.h"
#include "../cases.h"

template <class T>
void
test(T x, typename std::enable_if<std::is_integral<T>::value>::type* = 0)
{
    static_assert((std::is_same<decltype(std::arg(x)), double>::value), "");
    assert(std::arg(x) == arg(dpl::complex<double>(static_cast<double>(x), 0)));
}

template <class T>
void
test(T x, typename std::enable_if<!std::is_integral<T>::value>::type* = 0)
{
    static_assert((std::is_same<decltype(std::arg(x)), T>::value), "");
    assert(std::arg(x) == arg(dpl::complex<T>(x, 0)));
}

template <class T>
void
test()
{
    test<T>(0);
    test<T>(1);
    test<T>(10);
}

void run_test()
{
    test<float>();
    test<double>();
    test<long double>();
    test<int>();
    test<unsigned>();
    test<long long>();
}

int main(int, char**)
{
    run_test();

  return 0;
}
