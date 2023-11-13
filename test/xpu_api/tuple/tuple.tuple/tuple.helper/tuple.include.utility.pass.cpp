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
#include <oneapi/dpl/utility>
#include <oneapi/dpl/cstddef>
#include <oneapi/dpl/type_traits>

#include "support/test_macros.h"
#include "support/utils.h"

#if TEST_DPCPP_BACKEND_PRESENT
class KernelTupleIncludeUtilityTest;

template <class T, dpl::size_t N, class U, size_t idx>
void test()
{
    static_assert(dpl::is_base_of_v<dpl::integral_constant<dpl::size_t, N>, dpl::tuple_size<T>>);
    static_assert(dpl::is_base_of_v<dpl::integral_constant<dpl::size_t, N>, dpl::tuple_size<const T>>);
    static_assert(dpl::is_base_of_v<dpl::integral_constant<dpl::size_t, N>, dpl::tuple_size<volatile T>>);
    static_assert(dpl::is_base_of_v<dpl::integral_constant<dpl::size_t, N>, dpl::tuple_size<const volatile T>>);
    static_assert(dpl::is_same_v<typename dpl::tuple_element<idx, T>::type, U>);
    static_assert(dpl::is_same_v<typename dpl::tuple_element<idx, const T>::type, const U>);
    static_assert(dpl::is_same_v<typename dpl::tuple_element<idx, volatile T>::type, volatile U>);
    static_assert(dpl::is_same_v<typename dpl::tuple_element<idx, const volatile T>::type, const volatile U>);
}

template <class T, dpl::size_t N, class U, size_t idx>
bool test_runtime()
{
    bool ret = (dpl::is_base_of_v<dpl::integral_constant<dpl::size_t, N>, dpl::tuple_size<T>>);
    ret &= (dpl::is_base_of_v<dpl::integral_constant<dpl::size_t, N>, dpl::tuple_size<const T>>);
    ret &= (dpl::is_base_of_v<dpl::integral_constant<dpl::size_t, N>, dpl::tuple_size<volatile T>>);
    ret &= (dpl::is_base_of_v<dpl::integral_constant<dpl::size_t, N>, dpl::tuple_size<const volatile T>>);
    ret &= (dpl::is_same_v<typename dpl::tuple_element<idx, T>::type, U>);
    ret &= (dpl::is_same_v<typename dpl::tuple_element<idx, const T>::type, const U>);
    ret &= (dpl::is_same_v<typename dpl::tuple_element<idx, volatile T>::type, volatile U>);
    ret &= (dpl::is_same_v<typename dpl::tuple_element<idx, const volatile T>::type, const volatile U>);

    return ret;
}

void
kernel_test()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    bool ret = false;
    sycl::range<1> numOfItems{1};
    sycl::buffer<bool, 1> buffer1(&ret, numOfItems);
    deviceQueue.submit([&](sycl::handler& cgh) {
        auto ret_access = buffer1.get_access<sycl::access::mode::write>(cgh);
        cgh.single_task<class KernelTupleIncludeUtilityTest>([=]() {
            // Compile time check

            test<dpl::pair<int, int>, 2, int, 0>();
            test<dpl::pair<int, int>, 2, int, 1>();
            test<dpl::pair<const int, int>, 2, int, 1>();
            test<dpl::pair<int, volatile int>, 2, volatile int, 1>();
            test<dpl::pair<char*, int>, 2, char*, 0>();
            test<dpl::pair<char*, int>, 2, int, 1>();

            ret_access[0] = test_runtime<dpl::pair<int, int>, 2, int, 0>();
            ret_access[0] &= test_runtime<dpl::pair<int, int>, 2, int, 1>();
            ret_access[0] &= test_runtime<dpl::pair<const int, int>, 2, int, 1>();
            ret_access[0] &= test_runtime<dpl::pair<int, volatile int>, 2, volatile int, 1>();
            ret_access[0] &= test_runtime<dpl::pair<char*, int>, 2, char*, 0>();
            ret_access[0] &= test_runtime<dpl::pair<char*, int>, 2, int, 1>();
        });
    });

    auto ret_access_host = buffer1.get_host_access(sycl::read_only);
    EXPECT_TRUE(ret_access_host[0], "Wrong result of ");
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
