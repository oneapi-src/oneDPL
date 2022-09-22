//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "support/test_support.h"
#include "support/test_complex.h"

#include <ccomplex>

ONEDPL_TEST_NUM_MAIN
{
    TestUtils::invoke_test_if(HasDoubleSupportInRuntime(),
                              []()
                              {
                                  dpl::complex<double> d;
                                  (void)d;
                              });

  return 0;
}
