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

// requires RandomAccessIterator<Iter>
//   constexpr reverse_iterator operator-(difference_type n) const;
//
// constexpr in C++17

#include "support/test_config.h"

#include <oneapi/dpl/iterator>
#include <oneapi/dpl/type_traits>

#include "support/test_macros.h"
#include "support/test_iterators.h"
#include "support/utils.h"

#ifdef TEST_DPCPP_BACKEND_PRESENT
template <class It>
bool
test(It i, typename dpl::iterator_traits<It>::difference_type n, It x)
{
    const dpl::reverse_iterator<It> r(i);
    dpl::reverse_iterator<It> rr = r - n;
    return (rr.base() == x);
}

bool
kernel_test()
{
    sycl::queue deviceQueue;
    bool ret = true;
    {
        sycl::range<1> numOfItems{1};
        sycl::buffer<bool, 1> buffer1(&ret, numOfItems);
        deviceQueue.submit([&](sycl::handler& cgh) {
            auto ret_access = buffer1.get_access<sycl::access::mode::write>(cgh);
            cgh.single_task<class KernelTest>([=]() {
                const char* s = "1234567890";
                ret_access[0] &=
                    test(random_access_iterator<const char*>(s + 5), 5, random_access_iterator<const char*>(s + 10));
                ret_access[0] &= test(s + 5, 5, s + 10);

                {
                    constexpr const char* p = "123456789";
                    typedef dpl::reverse_iterator<const char*> RI;
                    constexpr RI it1 = dpl::make_reverse_iterator(p);
                    constexpr RI it2 = dpl::make_reverse_iterator(p + 5);
                    constexpr RI it3 = it1 - 5;
                    static_assert(it1 != it2);
                    static_assert(it1 != it3);
                    static_assert(it2 == it3);
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
#ifdef TEST_DPCPP_BACKEND_PRESENT
    auto ret = kernel_test();
    EXPECT_TRUE(ret, "Wrong result of work with reverse iterator and minus in kernel_test()");
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
