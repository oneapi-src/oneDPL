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

#include <sycl/sycl.hpp>
#include <iostream>

// Including `async` after `execution` and `numeric` compiles successfully.
#include <oneapi/dpl/async>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/numeric>

int
main(int argc, char** argv)
{
    sycl::queue q(sycl::default_selector_v);

    std::size_t n = 100;

    using T = float;

    T* v = sycl::malloc_device<T>(n, q);

    q.fill<T>(v, 1, n).wait();

    oneapi::dpl::execution::device_policy policy(q);

    auto f = oneapi::dpl::experimental::reduce_async(policy, v, v + n, T(0), std::plus());

    T value = f.get();

    std::cout << value << std::endl;

    return 0;
}