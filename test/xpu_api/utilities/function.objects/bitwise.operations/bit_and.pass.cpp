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

#include <oneapi/dpl/functional>
#include <oneapi/dpl/type_traits>

#include "support/utils.h"
#include "support/test_macros.h"

class KernelBitAndTest;

void
kernel_test()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    bool ret = false;
    sycl::range<1> numOfItems{1};
    sycl::buffer<bool, 1> buffer1(&ret, numOfItems);
    deviceQueue.submit([&](sycl::handler& cgh) {
        auto ret_access = buffer1.get_access<sycl::access::mode::write>(cgh);
        cgh.single_task<class KernelBitAndTest>([=]() {
            typedef dpl::bit_and<int> F;
            const F f = F();
#if TEST_STD_VER < 20
            static_assert(dpl::is_same<int, F::first_argument_type>::value);
            static_assert(dpl::is_same<int, F::second_argument_type>::value);
            static_assert(dpl::is_same<int, F::result_type>::value);
#endif // TEST_STD_VER < 20
            ret_access[0] = (f(0xEA95, 0xEA95) == 0xEA95);
            ret_access[0] &= (f(0xEA95, 0x58D3) == 0x4891);
            ret_access[0] &= (f(0x58D3, 0xEA95) == 0x4891);
            ret_access[0] &= (f(0x58D3, 0) == 0);
            ret_access[0] &= (f(0xFFFF, 0x58D3) == 0x58D3);

            const dpl::bit_and<long> f2;
            ret_access[0] &= (f2(0xEA95L, 0xEA95) == 0xEA95);
            ret_access[0] &= (f2(0xEA95, 0xEA95L) == 0xEA95);

            ret_access[0] &= (f2(0xEA95L, 0x58D3) == 0x4891);
            ret_access[0] &= (f2(0xEA95, 0x58D3L) == 0x4891);

            ret_access[0] &= (f2(0x58D3L, 0xEA95) == 0x4891);
            ret_access[0] &= (f2(0x58D3, 0xEA95L) == 0x4891);

            ret_access[0] &= (f2(0x58D3L, 0) == 0);
            ret_access[0] &= (f2(0x58D3, 0L) == 0);

            ret_access[0] &= (f2(0xFFFFL, 0x58D3) == 0x58D3);
            ret_access[0] &= (f2(0xFFFF, 0x58D3L) == 0x58D3);
        });
    });

    auto ret_access_host = buffer1.get_host_access(sycl::read_only);
    EXPECT_TRUE(ret_access_host[0], "Error in work with dpl::bit_and");
}

int
main()
{
    kernel_test();

    return TestUtils::done();
}
