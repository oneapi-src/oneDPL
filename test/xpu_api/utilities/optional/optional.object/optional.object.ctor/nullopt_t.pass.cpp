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

#include <oneapi/dpl/optional>
#include <oneapi/dpl/type_traits>

#include "support/test_macros.h"
#include "support/utils.h"

#if TEST_DPCPP_BACKEND_PRESENT
using dpl::nullopt;
using dpl::nullopt_t;
using dpl::optional;

template <class Opt>
void
test_constexpr()
{
    sycl::queue q;
    sycl::range<1> numOfItems1{1};
    {
        q.submit([&](sycl::handler& cgh) {
            cgh.single_task<Opt>([=]() {
                static_assert(dpl::is_nothrow_constructible<Opt, nullopt_t&>::value);
                static_assert(dpl::is_trivially_destructible<Opt>::value);
                static_assert(dpl::is_trivially_destructible<typename Opt::value_type>::value);

                constexpr Opt opt(nullopt);
                static_assert(static_cast<bool>(opt) == false);

                struct test_constexpr_ctor : public Opt
                {
                    constexpr test_constexpr_ctor() {}
                };
            });
        });
    }
}
#endif // TEST_DPCPP_BACKEND_PRESENT

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    test_constexpr<optional<int>>();
    test_constexpr<optional<int*>>();
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
