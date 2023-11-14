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

#include <oneapi/dpl/tuple>
#include <oneapi/dpl/type_traits>
#include <oneapi/dpl/cstddef>

#include "support/test_macros.h"
#include "support/utils.h"

#if TEST_DPCPP_BACKEND_PRESENT
class KernelTupleSizeTest;

template <class T, dpl::size_t N>
void
kernel_test()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    sycl::range<1> numOfItems{1};
    sycl::buffer<bool, 1> buffer1(&ret, numOfItems);
    deviceQueue.submit([&](sycl::handler& cgh) {
        cgh.single_task<class KernelTupleSizeTest>([=]() {
            static_assert(dpl::is_base_of_v<dpl::integral_constant<dpl::size_t, N>, dpl::tuple_size<T>>);
            static_assert(dpl::is_base_of_v<dpl::integral_constant<dpl::size_t, N>, dpl::tuple_size<const T>>);
            static_assert(dpl::is_base_of_v<dpl::integral_constant<dpl::size_t, N>, dpl::tuple_size<volatile T>>);
            static_assert(dpl::is_base_of_v<dpl::integral_constant<dpl::size_t, N>, dpl::tuple_size<const volatile T>>);
        });
    });
}
#endif // TEST_DPCPP_BACKEND_PRESENT

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    kernel_test<std::tuple<>, 0>();
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
