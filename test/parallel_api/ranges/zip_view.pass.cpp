// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This file incorporates work covered by the following copyright and permission
// notice:
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#include "support/pstl_test_config.h"

#include _PSTL_TEST_HEADER(execution)
#if _ENABLE_RANGES_TESTING
#include _PSTL_TEST_HEADER(ranges)
#endif

#include "support/utils.h"

#include <iostream>

int32_t
main()
{
#if _ENABLE_RANGES_TESTING
    constexpr int max_n = 10;
    char data[max_n] = {'b', 'e', 'g', 'f', 'c', 'd', 'a', 'j', 'i', 'h'};
    int key[max_n] = {1, 4, 6, 5, 2, 3, 0, 9, 8, 7};

    using namespace oneapi::dpl::experimental::ranges;

    auto view = nano::views::all(data);
    auto z = zip_view(nano::views::all(data), nano::views::all(key));

    //check access
    EXPECT_TRUE(::std::get<0>(z[2]) == 'g', "wrong effect with zip_view");
#endif //_ENABLE_RANGES_TESTING
    ::std::cout << TestUtils::done() << ::std::endl;
    return 0;
}
