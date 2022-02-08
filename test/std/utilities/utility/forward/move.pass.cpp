//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test move

#include "oneapi_std_test_config.h"
#include "test_macros.h"
#include <CL/sycl.hpp>
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

constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;
class move_only
{
    move_only(const move_only&);
    move_only&
    operator=(const move_only&);

  public:
    move_only(move_only&&) {}
    move_only&
    operator=(move_only&&)
    {
        return *this;
    }

    move_only() {}
};

move_only
source()
{
    return move_only();
}
const move_only
csource()
{
    return move_only();
}

void test(move_only) {}

struct A
{
    A() {}
    A(const A&) { ++copy_ctor; }
    A(A&&) { ++move_ctor; }
    A&
    operator=(const A&) = delete;
    int copy_ctor = 0;
    int move_ctor = 0;
};

#if TEST_STD_VER > 11
constexpr bool
test_constexpr_move()
{
    int y = 42;
    const int cy = y;
    return s::move(y) == 42 && s::move(cy) == 42 && s::move(static_cast<int&&>(y)) == 42 &&
           s::move(static_cast<int const&&>(y)) == 42;
}
#endif
cl::sycl::cl_bool
kernel_test()
{

    int x = 42;
    const int& cx = x;
    cl::sycl::cl_bool ret = false;
    { // Test return type and noexcept.
        static_assert(s::is_same<decltype(s::move(x)), int&&>::value, "");
        ASSERT_NOEXCEPT(s::move(x));
        static_assert(s::is_same<decltype(s::move(cx)), const int&&>::value, "");
        ASSERT_NOEXCEPT(s::move(cx));
        static_assert(s::is_same<decltype(s::move(42)), int&&>::value, "");
        ASSERT_NOEXCEPT(s::move(42));
    }
    { // test copy and move semantics
        A a;
        const A ca = A();

        ret = (a.copy_ctor == 0);
        ret &= (a.move_ctor == 0);

        A a2 = a;
        ret &= (a2.copy_ctor == 1);
        ret &= (a2.move_ctor == 0);

        A a3 = s::move(a);
        ret &= (a3.copy_ctor == 0);
        ret &= (a3.move_ctor == 1);

        A a4 = ca;
        ret &= (a4.copy_ctor == 1);
        ret &= (a4.move_ctor == 0);

        A a5 = s::move(ca);
        ret &= (a5.copy_ctor == 1);
        ret &= (a5.move_ctor == 0);
    }
    { // test on a move only type
        move_only mo;
        test(s::move(mo));
        test(source());
    }
#if TEST_STD_VER > 11
    {
        constexpr int y = 42;
        static_assert(s::move(y) == 42, "");
        static_assert(test_constexpr_move(), "");
    }
#endif
#if TEST_STD_VER == 11 && defined(_LIBCPP_VERSION)
    // Test that s::forward is constexpr in C++11. This is an extension
    // provided by both libc++ and libstdc++.
    {
        constexpr int y = 42;
        static_assert(s::move(y) == 42, "");
    }
#endif

    return ret;
}

class KernelTest;

int
main()
{
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

    if (ret)
    {
        std::cout << "Pass" << std::endl;
    }
    else
    {
        std::cout << "Fail" << std::endl;
    }

    return 0;
}
