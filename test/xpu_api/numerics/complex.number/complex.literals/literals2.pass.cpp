//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11
// <complex>

#include <complex>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

void run_test()
{
    using namespace std;

    {
    dpl::complex<long double> c1 = 3.0il;
    assert ( c1 == dpl::complex<long double>(0, 3.0));
    auto c2 = 3il;
    assert ( c1 == c2 );
    }

    {
    dpl::complex<double> c1 = 3.0i;
    assert ( c1 == dpl::complex<double>(0, 3.0));
    auto c2 = 3i;
    assert ( c1 == c2 );
    }

    {
    dpl::complex<float> c1 = 3.0if;
    assert ( c1 == dpl::complex<float>(0, 3.0));
    auto c2 = 3if;
    assert ( c1 == c2 );
    }
}

int main(int, char**)
{
    run_test();

  return 0;
}
