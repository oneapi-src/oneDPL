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

#include <oneapi/dpl/utility>

#include "support/test_macros.h"
#include "support/utils.h"

#if TEST_DPCPP_BACKEND_PRESENT
class test_obj
{
    int i;

  public:
    test_obj(int arg = 0) : i(arg) {}
    bool
    operator==(const test_obj& rhs) const
    {
        return i == rhs.i;
    }
    bool
    operator<(const test_obj& rhs) const
    {
        return i < rhs.i;
    }
};

template <typename T>
struct test_t
{
    bool b;

  public:
    test_t(bool arg = 0) : b(arg) {}
    bool
    operator==(const test_t& rhs) const
    {
        return b == rhs.b;
    }
    bool
    operator<(const test_t& rhs) const
    {
        return int(b) < int(rhs.b);
    }
};

// const&
bool
kernel_test()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    bool ret = false;
    bool check = false;
    sycl::range<1> numOfItem{1};

    const test_obj& obj1 = test_obj(5);
    const test_t<long>& tmpl1 = test_t<long>(false);
    const dpl::pair<test_t<long>, test_obj> p_st_1(tmpl1, obj1);
    const dpl::pair<test_t<long>, test_obj> p_st_2 = dpl::make_pair(tmpl1, obj1);
    {
        sycl::buffer<bool, 1> buffer1(&ret, numOfItem);
        sycl::buffer<bool, 1> buffer2(&check, numOfItem);
        sycl::buffer<decltype(p_st_1), 1> buffer3(&p_st_1, numOfItem);
        sycl::buffer<decltype(p_st_1), 1> buffer4(&p_st_2, numOfItem);
        deviceQueue.submit([&](sycl::handler& cgh) {
            auto ret_acc = buffer1.get_access<sycl::access::mode::write>(cgh);
            auto check_acc = buffer2.get_access<sycl::access::mode::write>(cgh);
            auto acc1 = buffer3.get_access<sycl::access::mode::read>(cgh);
            auto acc2 = buffer4.get_access<sycl::access::mode::read>(cgh);
            cgh.single_task<class KernelTest>([=]() {
                // check if there is change from input after data transfer
                check_acc[0] = (acc1[0].first == tmpl1);
                check_acc[0] &= (acc1[0].second == obj1);
                if (check_acc[0])
                {
                    ret_acc[0] = (acc1[0] == acc2[0]);
                    ret_acc[0] &= !(acc1[0] < acc2[0]);
                }
            });
        });
    }
    // check data after executing kernel function
    check &= (p_st_1.first == tmpl1);
    check &= (p_st_1.second == obj1);
    check &= (p_st_1 == p_st_2);
    if (!check)
        return false;
    return ret;
}
#endif // TEST_DPCPP_BACKEND_PRESENT

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    auto ret = kernel_test();
    EXPECT_TRUE(ret, "Wrong result of global dpl::make_pair check in kernel_test (#4)");
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
