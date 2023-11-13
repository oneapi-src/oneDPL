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
template <class T, dpl::size_t N, class U>
void __attribute__((always_inline)) test_compile()
{
    static_assert(dpl::is_same<typename dpl::tuple_element<N, T>::type, U>::value);
    static_assert(dpl::is_same<typename dpl::tuple_element<N, const T>::type, const U>::value);
    static_assert(dpl::is_same<typename dpl::tuple_element<N, volatile T>::type, volatile U>::value);
    static_assert(dpl::is_same<typename dpl::tuple_element<N, const volatile T>::type, const volatile U>::value);
}

template <class T, dpl::size_t N, class U>
bool __attribute__((always_inline)) test_runtime()
{
    bool ret = (dpl::is_same<typename dpl::tuple_element<N, T>::type, U>::value);
    ret &= (dpl::is_same<typename dpl::tuple_element<N, const T>::type, const U>::value);
    ret &= (dpl::is_same<typename dpl::tuple_element<N, volatile T>::type, volatile U>::value);
    ret &= (dpl::is_same<typename dpl::tuple_element<N, const volatile T>::type, const volatile U>::value);

    return ret;
}

class KernelTupleElementTest;

void
kernel_test()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    bool ret = true;
    sycl::range<1> numOfItems{1};
    sycl::buffer<bool, 1> buffer1(&ret, numOfItems);
    deviceQueue.submit([&](sycl::handler& cgh) {
        auto ret_access = buffer1.get_access<sycl::access::mode::write>(cgh);
        cgh.single_task<class KernelTupleElementTest>([=]() {
            test_compile<dpl::tuple<int>, 0, int>();
            test_compile<dpl::tuple<char, int>, 0, char>();
            test_compile<dpl::tuple<char, int>, 1, int>();
            test_compile<dpl::tuple<int*, char, int>, 0, int*>();
            test_compile<dpl::tuple<int*, char, int>, 1, char>();
            test_compile<dpl::tuple<int*, char, int>, 2, int>();

            // Runtime test

            ret_access[0] = test_runtime<dpl::tuple<int>, 0, int>();
            ret_access[0] &= test_runtime<dpl::tuple<char, int>, 0, char>();
            ret_access[0] &= test_runtime<dpl::tuple<char, int>, 1, int>();
            ret_access[0] &= test_runtime<dpl::tuple<int*, char, int>, 0, int*>();
            ret_access[0] &= test_runtime<dpl::tuple<int*, char, int>, 1, char>();
            ret_access[0] &= test_runtime<dpl::tuple<int*, char, int>, 2, int>();
        });
    });

    auto ret_access_host = buffer1.get_host_access(sycl::read_only);
    EXPECT_TRUE(ret_access_host[0], "Wrong result of dpl::tuple_element check");
}
#endif // TEST_DPCPP_BACKEND_PRESENT

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    kernel_test();
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
