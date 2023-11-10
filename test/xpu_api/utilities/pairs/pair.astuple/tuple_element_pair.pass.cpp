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

#include <oneapi/dpl/type_traits>
#include <oneapi/dpl/utility>

#include "support/test_macros.h"
#include "support/utils.h"

#if TEST_DPCPP_BACKEND_PRESENT
template <typename T1, typename T2>
bool
test()
{

    bool result = false;
    {
        typedef dpl::pair<T1, T2> P;
        static_assert(dpl::is_same<typename dpl::tuple_element<0, P>::type, T1>::value);
        static_assert(dpl::is_same<typename dpl::tuple_element<1, P>::type, T2>::value);

        static_assert(dpl::is_same<dpl::tuple_element_t<0, P>, T1>::value);
        static_assert(dpl::is_same<dpl::tuple_element_t<1, P>, T2>::value);

        result = (dpl::is_same<typename dpl::tuple_element<0, P>::type, T1>::value);
        result &= (dpl::is_same<typename dpl::tuple_element<1, P>::type, T2>::value);
    }
    {
        typedef T1 const Exp1;
        typedef T2 const Exp2;
        typedef dpl::pair<T1, T2> const P;
        static_assert(dpl::is_same<typename dpl::tuple_element<0, P>::type, Exp1>::value);
        static_assert(dpl::is_same<typename dpl::tuple_element<1, P>::type, Exp2>::value);

        static_assert(dpl::is_same<dpl::tuple_element_t<0, P>, Exp1>::value);
        static_assert(dpl::is_same<dpl::tuple_element_t<1, P>, Exp2>::value);

        result &= (dpl::is_same<typename dpl::tuple_element<0, P>::type, Exp1>::value);
        result &= (dpl::is_same<typename dpl::tuple_element<1, P>::type, Exp2>::value);
    }
    {
        typedef T1 volatile Exp1;
        typedef T2 volatile Exp2;
        typedef dpl::pair<T1, T2> volatile P;
        static_assert(dpl::is_same<typename dpl::tuple_element<0, P>::type, Exp1>::value);
        static_assert(dpl::is_same<typename dpl::tuple_element<1, P>::type, Exp2>::value);

        static_assert(dpl::is_same<dpl::tuple_element_t<0, P>, Exp1>::value);
        static_assert(dpl::is_same<dpl::tuple_element_t<1, P>, Exp2>::value);

        result &= (dpl::is_same<typename dpl::tuple_element<0, P>::type, Exp1>::value);
        result &= (dpl::is_same<typename dpl::tuple_element<1, P>::type, Exp2>::value);
    }
    {
        typedef T1 const volatile Exp1;
        typedef T2 const volatile Exp2;
        typedef dpl::pair<T1, T2> const volatile P;
        static_assert(dpl::is_same<typename dpl::tuple_element<0, P>::type, Exp1>::value);
        static_assert(dpl::is_same<typename dpl::tuple_element<1, P>::type, Exp2>::value);

        static_assert(dpl::is_same<dpl::tuple_element_t<0, P>, Exp1>::value);
        static_assert(dpl::is_same<dpl::tuple_element_t<1, P>, Exp2>::value);

        result &= (dpl::is_same<typename dpl::tuple_element<0, P>::type, Exp1>::value);
        result &= (dpl::is_same<typename dpl::tuple_element<1, P>::type, Exp2>::value);
    }

    return result;
}

class KernelPairTest1;
class KernelPairTest2;

template <typename T1, typename T2, typename KC>
void
kernel_test()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    bool ret = false;
    sycl::range<1> numOfItems{1};
    sycl::buffer<bool, 1> buffer1(&ret, numOfItems);
    deviceQueue.submit([&](sycl::handler& cgh) {
        auto ret_access = buffer1.get_access<sycl::access::mode::write>(cgh);
        cgh.single_task<KC>([=]() { ret_access[0] = test<T1, T2>(); });
    });

    auto ret_access_host = buffer1.get_host_access(sycl::read_only);
    EXPECT_TRUE(ret_access_host[0], "Wrong result of dpl::pair elements check: should be dpl::tuple_element");
}
#endif // TEST_DPCPP_BACKEND_PRESENT

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    kernel_test<int, short, KernelPairTest1>();
    kernel_test<int*, char, KernelPairTest2>();
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
