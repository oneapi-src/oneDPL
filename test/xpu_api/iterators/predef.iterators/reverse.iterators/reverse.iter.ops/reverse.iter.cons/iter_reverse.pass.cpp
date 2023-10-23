//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iterator>

// reverse_iterator

// explicit constexpr reverse_iterator(Iter x);
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
test(It i)
{
    dpl::reverse_iterator<It> r(i);
    return (r.base() == i);
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
                const char s[] = "123";
                ret_access[0] &= test(bidirectional_iterator<const char*>(s));
                ret_access[0] &= test(random_access_iterator<const char*>(s));
                ret_access[0] &= test(s);

                {
                    constexpr const char* p = "123456789";
                    constexpr dpl::reverse_iterator<const char*> it(p);
                    static_assert(it.base() == p);
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
    EXPECT_TRUE(ret, "Wrong result of work with reverse_iterator in kernel_test()");
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
