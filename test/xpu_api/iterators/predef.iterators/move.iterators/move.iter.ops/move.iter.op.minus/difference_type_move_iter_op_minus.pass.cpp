//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iterator>

// move_iterator

// requires RandomAccessIterator<Iter>
//   move_iterator operator-(difference_type n) const;
//
//  constexpr in C++17

#include "support/test_config.h"

#include <oneapi/dpl/iterator>
#include <oneapi/dpl/type_traits>

#include "support/test_iterators.h"
#include "support/utils.h"

#if TEST_DPCPP_BACKEND_PRESENT

template <class It>
bool
test(It i, typename dpl::iterator_traits<It>::difference_type n, It x)
{
    const dpl::move_iterator<It> r(i);
    dpl::move_iterator<It> rr = r - n;
    return (rr.base() == x);
}
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
                const char* s = "1234567890";
                ret_access[0] &=
                    test(random_access_iterator<const char*>(s + 5), 5, random_access_iterator<const char*>(s));
                ret_access[0] &= test(s + 5, 5, s);

                {
                    constexpr const char* p = "123456789";
                    typedef dpl::move_iterator<const char*> MI;
                    constexpr MI it1 = dpl::make_move_iterator(p);
                    constexpr MI it2 = dpl::make_move_iterator(p + 5);
                    constexpr MI it3 = it2 - 5;
                    static_assert(it1 != it2);
                    static_assert(it1 == it3);
                    static_assert(it2 != it3);
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
    EXPECT_TRUE(ret, "Wrong result of move_iterator check with minus operator in kernel_test()");
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
