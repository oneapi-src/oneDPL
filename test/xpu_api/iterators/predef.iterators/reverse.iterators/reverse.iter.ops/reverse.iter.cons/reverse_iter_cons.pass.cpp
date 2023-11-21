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

// <iterator>

// reverse_iterator

// template <class U>
//   requires HasConstructor<Iter, const U&>
//   constexpr reverse_iterator(const reverse_iterator<U> &u);
//
// constexpr in C++17

#include "support/test_config.h"

#include <oneapi/dpl/iterator>
#include <oneapi/dpl/type_traits>

#include "support/test_macros.h"
#include "support/test_iterators.h"
#include "support/utils.h"

#if TEST_DPCPP_BACKEND_PRESENT
template <class It, class U>
bool
test(U u)
{
    const dpl::reverse_iterator<U> r2(u);
    dpl::reverse_iterator<It> r1 = r2;
    return (r1.base() == u);
}

struct Base
{
};
struct Derived : Base
{
};

bool
kernel_test()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    bool ret = true;
    {
        sycl::range<1> numOfItems{1};
        sycl::buffer<bool, 1> buffer1(&ret, numOfItems);
        deviceQueue.submit([&](sycl::handler& cgh) {
            auto ret_access = buffer1.get_access<sycl::access::mode::write>(cgh);
            cgh.single_task<class KernelTest>([=]() {
                Derived d;

                ret_access[0] &= test<bidirectional_iterator<Base*>>(bidirectional_iterator<Derived*>(&d));
                ret_access[0] &= test<random_access_iterator<const Base*>>(random_access_iterator<Derived*>(&d));
                ret_access[0] &= test<Base*>(&d);

                {
                    constexpr const Derived* p = nullptr;
                    constexpr dpl::reverse_iterator<const Derived*> it1 = dpl::make_reverse_iterator(p);
                    constexpr dpl::reverse_iterator<const Base*> it2(it1);
                    static_assert(it2.base() == p);
                }
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
    auto ret = kernel_test();
    EXPECT_TRUE(ret, "Wrong result of work with const reverse iterator in kernel_test()");
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
