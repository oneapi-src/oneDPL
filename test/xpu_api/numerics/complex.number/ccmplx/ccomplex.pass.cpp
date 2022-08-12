//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <ccomplex>

#include <ccomplex>

#include "test_macros.h"

void run_test()
{
    dpl::complex<double> d;
    (void)d;
}

int main(int, char**)
{
    run_test();

  return 0;
}
