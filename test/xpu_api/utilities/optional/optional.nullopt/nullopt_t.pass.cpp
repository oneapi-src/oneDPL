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

// <optional>

// struct nullopt_t{see below};
// inline constexpr nullopt_t nullopt(unspecified);

// [optional.nullopt]/2:
//   Type nullopt_t shall not have a default constructor or an initializer-list
//   constructor, and shall not be an aggregate.

#include "support/test_config.h"

#include <oneapi/dpl/optional>
#include <oneapi/dpl/type_traits>

#include "support/test_macros.h"
#include "support/utils.h"

#if TEST_DPCPP_BACKEND_PRESENT
using dpl::nullopt;
using dpl::nullopt_t;

constexpr bool
test()
{
    nullopt_t foo{nullopt};
    (void)foo;
    return true;
}

void
kernel_test()
{
    sycl::queue q;
    q.submit([&](sycl::handler& cgh) {
        cgh.single_task<class KernelTest>([=]() {
            static_assert(dpl::is_empty_v<nullopt_t>);
            static_assert(!dpl::is_default_constructible_v<nullopt_t>);

            static_assert(dpl::is_same_v<const nullopt_t, decltype(nullopt)>);
            static_assert(test());
        });
    });
}
#endif // TEST_DPCPP_BACKEND_PRESENT

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    kernel_test();
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
