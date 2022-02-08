//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <utility>

// template <class T>
//     typename conditional
//     <
//         !is_nothrow_move_constructible<T>::value && is_copy_constructible<T>::value,
//         const T&,
//         T&&
//     >::type
//     move_if_noexcept(T& x);

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

class A
{
    A(const A&);
    A&
    operator=(const A&);

  public:
    A() {}
    A(A&&) {}
};

struct legacy
{
    legacy() {}
    legacy(const legacy&);
};

cl::sycl::cl_bool
kernel_test()
{
    int i = 0;
    const int ci = 0;

    legacy l;
    A a;
    const A ca;

#if TEST_STD_VER >= 11
    static_assert((s::is_same<decltype(s::move_if_noexcept(i)), int&&>::value), "");
    static_assert((s::is_same<decltype(s::move_if_noexcept(ci)), const int&&>::value), "");
    static_assert((s::is_same<decltype(s::move_if_noexcept(a)), A&&>::value), "");
    static_assert((s::is_same<decltype(s::move_if_noexcept(ca)), const A&&>::value), "");
    static_assert((s::is_same<decltype(s::move_if_noexcept(l)), const legacy&>::value), "");
#else // C++ < 11
    // In C++03 we don't have noexcept so we can never move :-(
    static_assert((s::is_same<decltype(s::move_if_noexcept(i)), const int&>::value), "");
    static_assert((s::is_same<decltype(s::move_if_noexcept(ci)), const int&>::value), "");
    static_assert((s::is_same<decltype(s::move_if_noexcept(a)), const A&>::value), "");
    static_assert((s::is_same<decltype(s::move_if_noexcept(ca)), const A&>::value), "");
    static_assert((s::is_same<decltype(s::move_if_noexcept(l)), const legacy&>::value), "");
#endif

#if TEST_STD_VER > 11
    constexpr int i1 = 23;
    constexpr int i2 = s::move_if_noexcept(i1);
    static_assert(i2 == 23, "");
#endif

    return true;
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
