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

#include "support/test_config.h"

#if TEST_DPCPP_BACKEND_PRESENT
#    include _PSTL_TEST_HEADER(async)
#endif // TEST_DPCPP_BACKEND_PRESENT
#include _PSTL_TEST_HEADER(execution)

#include "support/utils.h"

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT

    constexpr std::size_t n = 100;

    auto my_policy = TestUtils::make_device_policy<class Kernel1>(oneapi::dpl::execution::dpcpp_default);

    auto q = sycl::queue{TestUtils::default_selector};

    using T = float;
    T* v = sycl::malloc_device<T>(n, q);

    q.fill<T>(v, 1, n).wait();

    auto f = oneapi::dpl::experimental::reduce_async(my_policy, v, v + n, T(0), std::plus());
    f.get();

    sycl::free(v, q);

#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done();
}
