// -*- C++ -*-
//===-- merge.pass.cpp ----------------------------------------------------===//
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

#include _PSTL_TEST_HEADER(execution)
#include _PSTL_TEST_HEADER(algorithm)

#include "support/utils.h"

#include <functional>

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT

    using _Type = float;
    //const int n = 1 * 1024 * 1024;
    const int n = 100 * 1024;
    sycl::buffer<_Type> buf{ sycl::range<1>(n) };
    auto buf_begin = oneapi::dpl::begin(buf);
    auto buf_end = buf_begin + n;

    const auto policy = TestUtils::default_dpcpp_policy;
    auto buf_begin_discard_write = oneapi::dpl::begin(buf, sycl::write_only, sycl::property::no_init{});
    std::fill(policy, buf_begin_discard_write, buf_begin_discard_write + n, 1);

    dpl::sort(policy, buf_begin, buf_end, [](_Type v1, _Type v2){ return v1 < v2; });

    assert(dpl::is_sorted(policy, buf_begin, buf_end, std::less<_Type>{}));
#endif

    return TestUtils::done();
}
