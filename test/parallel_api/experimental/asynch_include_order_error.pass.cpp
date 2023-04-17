// -*- C++ -*-
//===-- async.pass.cpp ----------------------------------------------------===//
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

// Including `async` after `execution` and `numeric` compiles successfully.
#include <oneapi/dpl/async>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/numeric>

#include <iostream>

#include "support/utils.h"

#if TEST_DPCPP_BACKEND_PRESENT
#include <sycl/sycl.hpp>
#endif // TEST_DPCPP_BACKEND_PRESENT

int
main(int argc, char** argv)
{
#if TEST_DPCPP_BACKEND_PRESENT
    sycl::queue q(sycl::default_selector_v);

    std::size_t n = 100;

    using T = float;

    T* v = sycl::malloc_device<T>(n, q);

    q.fill<T>(v, 1, n).wait();

    oneapi::dpl::execution::device_policy policy(q);

    auto f = oneapi::dpl::experimental::reduce_async(policy, v, v + n, T(0), std::plus());

    T value = f.get();

    std::cout << value << std::endl;
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}