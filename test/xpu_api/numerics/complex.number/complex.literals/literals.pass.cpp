//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11
// <chrono>

#include "support/test_complex.h"

ONEDPL_TEST_NUM_MAIN
{
    using namespace std::literals::complex_literals;

    //  Make sure the types are right
    TestUtils::invoke_test_if(HasLongDoubleSupportInCompiletime(), []()
        {
            static_assert(std::is_same<decltype(3.0il), dpl::complex<long double>>::value, "");
            static_assert(std::is_same<decltype(3il), dpl::complex<long double>>::value, "");
        });
    TestUtils::invoke_test_if(HasDoubleSupportInRuntime(),
                              []()
                              {
                                  static_assert(std::is_same<decltype(3.0i), dpl::complex<double>>::value, "");
                                  static_assert(std::is_same<decltype(3i), dpl::complex<double>>::value, "");
                              });
    static_assert ( std::is_same<decltype( 3.0if ), dpl::complex<float>>::value, "" );
    static_assert ( std::is_same<decltype( 3if   ), dpl::complex<float>>::value, "" );

    TestUtils::invoke_test_if(HasLongDoubleSupportInCompiletime(), []()
                              {
                                  dpl::complex<long double> c1 = 3.0il;
                                  assert(c1 == dpl::complex<long double>(0, 3.0));
                                  auto c2 = 3il; assert(c1 == c2);

                                  dpl::complex<long double> c1 = 3.0il;
                                  assert(c1 == dpl::complex<long double>(0, 3.0));
                                  auto c2 = 3il; assert(c1 == c2);
                              });

    TestUtils::invoke_test_if(HasDoubleSupportInRuntime(),
                              []()
                              {
                                  dpl::complex<double> c1 = 3.0i;
                                  assert(c1 == dpl::complex<double>(0, 3.0));
                                  auto c2 = 3i;
                                  assert(c1 == c2);
                              });

    {
    dpl::complex<float> c1 = 3.0if;
    assert ( c1 == dpl::complex<float>(0, 3.0));
    auto c2 = 3if;
    assert ( c1 == c2 );
    }

  return 0;
}
