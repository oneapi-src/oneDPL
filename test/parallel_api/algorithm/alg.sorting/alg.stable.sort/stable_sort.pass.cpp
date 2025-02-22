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
#include _PSTL_TEST_HEADER(execution)
#include _PSTL_TEST_HEADER(algorithm)

#include <vector>
#include <cstdint>

#include "support/utils.h"
#include "support/utils_sort.h"

int main()
{
    constexpr bool Stable = true;
    SortTestConfig cfg{Stable};
    cfg.test_usm_shared = true;
    std::vector<std::size_t> sizes = test_sizes(TestUtils::max_n);

#if !TEST_ONLY_HETERO_POLICIES
    test_sort<TestUtils::float32_t>(cfg.msg("float, host"), sizes, Host{}, Converter<TestUtils::float32_t>{});
    test_sort<std::int64_t>(cfg.msg("int64_t, host"), sizes, Host{}, Converter<std::int64_t>{});
    // TODO: add a test for stability
#endif

#if TEST_DPCPP_BACKEND_PRESENT
    test_sort<TestUtils::float32_t>(cfg.msg("float, device"), sizes, Device<0>{}, Converter<TestUtils::float32_t>{});
    test_sort<std::int64_t>(cfg.msg("int64_t, device"), sizes, Device<1>{}, Converter<std::int64_t>{});
    // TODO: add a test for stability
#endif

    return TestUtils::done();
}
