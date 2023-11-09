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
template <class Rat1, class Rat2, bool result, class KernelName>
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
            cgh.single_task<KernelName>([=]() {
                static_assert(result == dpl::ratio_less<Rat1, Rat2>::value);
                ret_acc[0] = true;
            });
        });
    }
    return ret;
}

class T1;
class T2;
class T3;
class T4;
class T5;
class T6;
class T7;
class T8;
class T9;
class T10;
class T11;
class T12;
class T13;
class T14;

bool
kernel_test()
{
    auto ret = true;
    {
        typedef dpl::ratio<1, 1> R1;
        typedef dpl::ratio<1, 1> R2;
        ret &= test<R1, R2, false, T1>();
    }
    {
        typedef dpl::ratio<0x7FFFFFFFFFFFFFFFLL, 1> R1;
        typedef dpl::ratio<0x7FFFFFFFFFFFFFFFLL, 1> R2;
        ret &= test<R1, R2, false, T2>();
    }
    {
        typedef dpl::ratio<-0x7FFFFFFFFFFFFFFFLL, 1> R1;
        typedef dpl::ratio<-0x7FFFFFFFFFFFFFFFLL, 1> R2;
        ret &= test<R1, R2, false, T3>();
    }
    {
        typedef dpl::ratio<1, 0x7FFFFFFFFFFFFFFFLL> R1;
        typedef dpl::ratio<1, 0x7FFFFFFFFFFFFFFFLL> R2;
        ret &= test<R1, R2, false, T4>();
    }
    {
        typedef dpl::ratio<1, 1> R1;
        typedef dpl::ratio<1, -1> R2;
        ret &= test<R1, R2, false, T5>();
    }
    {
        typedef dpl::ratio<0x7FFFFFFFFFFFFFFFLL, 1> R1;
        typedef dpl::ratio<-0x7FFFFFFFFFFFFFFFLL, 1> R2;
        ret &= test<R1, R2, false, T6>();
    }
    {
        typedef dpl::ratio<-0x7FFFFFFFFFFFFFFFLL, 1> R1;
        typedef dpl::ratio<0x7FFFFFFFFFFFFFFFLL, 1> R2;
        ret &= test<R1, R2, true, T7>();
    }
    {
        typedef dpl::ratio<1, 0x7FFFFFFFFFFFFFFFLL> R1;
        typedef dpl::ratio<1, -0x7FFFFFFFFFFFFFFFLL> R2;
        ret &= test<R1, R2, false, T8>();
    }
    {
        typedef dpl::ratio<0x7FFFFFFFFFFFFFFFLL, 0x7FFFFFFFFFFFFFFELL> R1;
        typedef dpl::ratio<0x7FFFFFFFFFFFFFFDLL, 0x7FFFFFFFFFFFFFFCLL> R2;
        ret &= test<R1, R2, true, T9>();
    }
    {
        typedef dpl::ratio<0x7FFFFFFFFFFFFFFDLL, 0x7FFFFFFFFFFFFFFCLL> R1;
        typedef dpl::ratio<0x7FFFFFFFFFFFFFFFLL, 0x7FFFFFFFFFFFFFFELL> R2;
        ret &= test<R1, R2, false, T10>();
    }
    {
        typedef dpl::ratio<-0x7FFFFFFFFFFFFFFDLL, 0x7FFFFFFFFFFFFFFCLL> R1;
        typedef dpl::ratio<-0x7FFFFFFFFFFFFFFFLL, 0x7FFFFFFFFFFFFFFELL> R2;
        ret &= test<R1, R2, true, T11>();
    }
    {
        typedef dpl::ratio<0x7FFFFFFFFFFFFFFFLL, 0x7FFFFFFFFFFFFFFELL> R1;
        typedef dpl::ratio<0x7FFFFFFFFFFFFFFELL, 0x7FFFFFFFFFFFFFFDLL> R2;
        ret &= test<R1, R2, true, T12>();
    }
    {
        typedef dpl::ratio<641981, 1339063> R1;
        typedef dpl::ratio<1291640, 2694141LL> R2;
        ret &= test<R1, R2, false, T13>();
    }
    {
        typedef dpl::ratio<1291640, 2694141LL> R1;
        typedef dpl::ratio<641981, 1339063> R2;
        ret &= test<R1, R2, true, T14>();
    }

    return ret;
}
#endif // TEST_DPCPP_BACKEND_PRESENT

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    auto ret = kernel_test();
    EXPECT_TRUE(ret, "Wrong result of dpl::ratio_less check");
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
