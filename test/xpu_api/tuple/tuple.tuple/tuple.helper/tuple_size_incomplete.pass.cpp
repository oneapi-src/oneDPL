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
#include <oneapi/dpl/utility>
#include <oneapi/dpl/array>
#include <oneapi/dpl/tuple>

#include "support/test_macros.h"
#include "support/utils.h"

#if TEST_DPCPP_BACKEND_PRESENT
class KernelTupleSizeIncompleteTest;

template <class T, std::size_t Size = sizeof(dpl::tuple_size<T>)>
constexpr bool
is_complete(int)
{
    static_assert(Size > 0);
    return true;
}
template <class>
constexpr bool
is_complete(long)
{
    return false;
}
template <class T>
constexpr bool
is_complete()
{
    return is_complete<T>(0);
}

template <class T, std::size_t Size = sizeof(dpl::tuple_size<T>)>
constexpr bool
is_complete_runtime(int)
{
    return (Size > 0);
}
template <class>
constexpr bool
is_complete_runtime(long)
{
    return false;
}
template <class T>
constexpr bool
is_complete_runtime()
{
    return is_complete<T>(0);
}

struct Dummy1
{
};
struct Dummy2
{
};

namespace std
{
template <>
struct tuple_size<Dummy1> : public integral_constant<std::size_t, 0>
{
};
} // namespace std

template <class T>
void test_complete_compile()
{
    static_assert(is_complete<T>());
    static_assert(is_complete<const T>());
    static_assert(is_complete<volatile T>());
    static_assert(is_complete<const volatile T>());
}

template <class T>
bool test_complete_runtime()
{
    bool ret;
    ret = is_complete<T>();
    ret &= is_complete<const T>();
    ret &= is_complete<volatile T>();
    ret &= is_complete<const volatile T>();
    return ret;
}

template <class T>
bool test_incomplete()
{
    bool ret;
    ret = !is_complete_runtime<T>();
    ret &= !is_complete_runtime<const T>();
    ret &= !is_complete_runtime<volatile T>();
    ret &= !is_complete_runtime<const volatile T>();
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
        cgh.single_task<class KernelTupleSizeIncompleteTest>([=]() {
            // Compile time check

            test_complete_compile<dpl::tuple<>>();
            test_complete_compile<dpl::tuple<int&>>();
            test_complete_compile<dpl::tuple<int&&, int&, void*>>();
            test_complete_compile<dpl::pair<int, long>>();
            test_complete_compile<dpl::array<int, 5>>();
            test_complete_compile<Dummy1>();

            ret_access[0] = test_complete_runtime<dpl::tuple<>>();
            ret_access[0] &= test_complete_runtime<dpl::tuple<int&>>();
            ret_access[0] &= test_complete_runtime<dpl::tuple<int&&, int&, void*>>();
            ret_access[0] &= test_complete_runtime<dpl::pair<int, long>>();
            ret_access[0] &= test_complete_runtime<dpl::array<int, 5>>();
            ret_access[0] &= test_complete_runtime<Dummy1>();
        });
    });

    auto ret_access_host = buffer1.get_host_access(sycl::read_only);
    EXPECT_TRUE(ret_access_host[0], "Wrong result of dpl::tuple_size check in SFINAE");
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
