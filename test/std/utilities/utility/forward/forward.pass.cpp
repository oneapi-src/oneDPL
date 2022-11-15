//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03

// test forward

#include "oneapi_std_test_config.h"
#include "test_macros.h"

#include <iostream>

#ifdef USE_ONEAPI_STD
#    include _ONEAPI_STD_TEST_HEADER(type_traits)
#    include _ONEAPI_STD_TEST_HEADER(utility)
namespace s = oneapi_cpp_ns;
#else
#    include <utility>
#    include <type_traits>
namespace s = std;
#endif

#if TEST_DPCPP_BACKEND_PRESENT
constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;

struct A
{
};

A
source() TEST_NOEXCEPT
{
    return A();
}
const A
csource() TEST_NOEXCEPT
{
    return A();
}

#if TEST_STD_VER > 11
constexpr bool
test_constexpr_forward()
{
    int x = 42;
    const int cx = 101;
    return s::forward<int&>(x) == 42 && s::forward<int>(x) == 42 && s::forward<const int&>(x) == 42 &&
           s::forward<const int>(x) == 42 && s::forward<int&&>(x) == 42 && s::forward<const int&&>(x) == 42 &&
           s::forward<const int&>(cx) == 101 && s::forward<const int>(cx) == 101;
}
#endif

cl::sycl::cl_bool
kernel_test()
{
    A a;
    const A ca = A();

    ((void)a);  // Prevent unused warning
    ((void)ca); // Prevent unused warning

    static_assert(s::is_same<decltype(s::forward<A&>(a)), A&>::value, "");
    static_assert(s::is_same<decltype(s::forward<A>(a)), A&&>::value, "");
    static_assert(s::is_same<decltype(s::forward<A>(source())), A&&>::value, "");
    ASSERT_NOEXCEPT(s::forward<A&>(a));
    ASSERT_NOEXCEPT(s::forward<A>(a));
    ASSERT_NOEXCEPT(s::forward<A>(source()));

    static_assert(s::is_same<decltype(s::forward<const A&>(a)), const A&>::value, "");
    static_assert(s::is_same<decltype(s::forward<const A>(a)), const A&&>::value, "");
    static_assert(s::is_same<decltype(s::forward<const A>(source())), const A&&>::value, "");
    ASSERT_NOEXCEPT(s::forward<const A&>(a));
    ASSERT_NOEXCEPT(s::forward<const A>(a));
    ASSERT_NOEXCEPT(s::forward<const A>(source()));

    static_assert(s::is_same<decltype(s::forward<const A&>(ca)), const A&>::value, "");
    static_assert(s::is_same<decltype(s::forward<const A>(ca)), const A&&>::value, "");
    static_assert(s::is_same<decltype(s::forward<const A>(csource())), const A&&>::value, "");
    ASSERT_NOEXCEPT(s::forward<const A&>(ca));
    ASSERT_NOEXCEPT(s::forward<const A>(ca));
    ASSERT_NOEXCEPT(s::forward<const A>(csource()));

#if TEST_STD_VER > 11
    {
        constexpr int i2 = s::forward<int>(42);
        static_assert(s::forward<int>(42) == 42, "");
        static_assert(s::forward<const int&>(i2) == 42, "");
        static_assert(test_constexpr_forward(), "");
    }
#endif
#if TEST_STD_VER == 11 && defined(_LIBCPP_VERSION)
    // Test that s::forward is constexpr in C++11. This is an extension
    // provided by both libc++ and libstdc++.
    {
        constexpr int i2 = s::forward<int>(42);
        static_assert(s::forward<int>(42) == 42, "");
        static_assert(s::forward<const int&>(i2) == 42, "");
    }
#endif

    return true;
}

class KernelTest;
#endif // TEST_DPCPP_BACKEND_PRESENT

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    cl::sycl::queue deviceQueue;
    cl::sycl::cl_bool ret = false;
    cl::sycl::range<1> numOfItems{1};
    {
        cl::sycl::buffer<cl::sycl::cl_bool, 1> buffer1(&ret, numOfItems);
        deviceQueue.submit([&](cl::sycl::handler& cgh) {
            auto ret_access = buffer1.get_access<sycl_write>(cgh);
            cgh.single_task<class KernelTest>([=]() { ret_access[0] = kernel_test(); });
        });
    }

    TestUtils::exitOnError(ret);
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
