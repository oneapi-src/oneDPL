//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iterator>

// reverse_iterator

// constexpr reverse_iterator();
//
// constexpr in C++17

#include "support/test_config.h"

#include <oneapi/dpl/iterator>
#include <oneapi/dpl/type_traits>

#include "support/test_iterators.h"
#include "support/utils.h"

#ifdef TEST_DPCPP_BACKEND_PRESENT
template <class It>
void
test()
{
    dpl::reverse_iterator<It> r;
    (void)r;
}

void
kernel_test()
{
    sycl::queue deviceQueue;
    {
        deviceQueue.submit([&](sycl::handler& cgh) {
            cgh.single_task<class KernelTest>([=]() {
                test<bidirectional_iterator<const char*>>();
                test<random_access_iterator<char*>>();
                test<char*>();
                test<const char*>();

                {
                    constexpr dpl::reverse_iterator<const char*> it;
                    (void)it;
                }
            });
        });
    }
}
#endif // TEST_DPCPP_BACKEND_PRESENT

int
main()
{
#ifdef TEST_DPCPP_BACKEND_PRESENT
    kernel_test();
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
