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

// UNSUPPORTED: c++03

// <algorithm>

// template <class T>
//   T
//   max(initializer_list<T> t);

// In Windows, as a temporary workaround, disable vector algorithm calls to avoid calls within sycl kernels
#if defined(_MSC_VER)
#    define _USE_STD_VECTOR_ALGORITHMS 0
#endif

#include "support/test_complex.h"
#include "support/test_macros.h"

#include <oneapi/dpl/algorithm>
#include <cassert>

ONEDPL_TEST_NUM_MAIN
{
    int i = dpl::max({2, 3, 1});
    assert(i == 3);
    i = dpl::max({2, 1, 3});
    assert(i == 3);
    i = dpl::max({3, 1, 2});
    assert(i == 3);
    i = dpl::max({3, 2, 1});
    assert(i == 3);
    i = dpl::max({1, 2, 3});
    assert(i == 3);
    i = dpl::max({1, 3, 2});
    assert(i == 3);
#if TEST_STD_VER >= 14
    {
    static_assert(dpl::max({1, 3, 2}) == 3, "");
    static_assert(dpl::max({2, 1, 3}) == 3, "");
    static_assert(dpl::max({3, 2, 1}) == 3, "");
    }
#endif

  return 0;
}
