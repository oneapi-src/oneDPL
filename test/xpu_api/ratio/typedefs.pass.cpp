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

#include <oneapi/dpl/ratio>

#include "support/test_macros.h"
#include "support/utils.h"

#if TEST_DPCPP_BACKEND_PRESENT
bool
test()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    bool ret = false;
    sycl::range<1> item1{1};
    {
        sycl::buffer<bool, 1> buffer1(&ret, item1);
        deviceQueue.submit([&](sycl::handler& cgh) {
            auto ret_acc = buffer1.get_access<sycl::access::mode::write>(cgh);
            cgh.single_task<class KernelTest>([=]() {
                static_assert(std::atto::num == 1 && std::atto::den == 1000000000000000000ULL);
                static_assert(std::femto::num == 1 && std::femto::den == 1000000000000000ULL);
                static_assert(std::pico::num == 1 && std::pico::den == 1000000000000ULL);
                static_assert(std::nano::num == 1 && std::nano::den == 1000000000ULL);
                static_assert(std::micro::num == 1 && std::micro::den == 1000000ULL);
                static_assert(std::milli::num == 1 && std::milli::den == 1000ULL);
                static_assert(std::centi::num == 1 && std::centi::den == 100ULL);
                static_assert(std::deci::num == 1 && std::deci::den == 10ULL);
                static_assert(std::deca::num == 10ULL && std::deca::den == 1);
                static_assert(std::hecto::num == 100ULL && std::hecto::den == 1);
                static_assert(std::kilo::num == 1000ULL && std::kilo::den == 1);
                static_assert(std::mega::num == 1000000ULL && std::mega::den == 1);
                static_assert(std::giga::num == 1000000000ULL && std::giga::den == 1);
                static_assert(std::tera::num == 1000000000000ULL && std::tera::den == 1);
                static_assert(std::peta::num == 1000000000000000ULL && std::peta::den == 1);
                static_assert(std::exa::num == 1000000000000000000ULL && std::exa::den == 1);
                ret_acc[0] = true;
            });
        });
    }
    return ret;
}
#endif // TEST_DPCPP_BACKEND_PRESENT

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    auto ret = test();
    EXPECT_TRUE(ret, "Wrong result of ratio typedefs check");
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
