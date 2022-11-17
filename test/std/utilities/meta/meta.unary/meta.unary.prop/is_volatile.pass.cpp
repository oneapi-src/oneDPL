//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// is_volatile

#include "oneapi_std_test_config.h"
#include "test_macros.h"

#include <iostream>

#ifdef USE_ONEAPI_STD
#    include _ONEAPI_STD_TEST_HEADER(type_traits)
namespace s = oneapi_cpp_ns;
#else
#    include <type_traits>
namespace s = std;
#endif

constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;

template <class T>
void
test_is_volatile()
{
    static_assert(!s::is_volatile<T>::value, "");
    static_assert(!s::is_volatile<const T>::value, "");
    static_assert(s::is_volatile<volatile T>::value, "");
    static_assert(s::is_volatile<const volatile T>::value, "");
#if TEST_STD_VER > 14
    static_assert(!s::is_volatile_v<T>, "");
    static_assert(!s::is_volatile_v<const T>, "");
    static_assert(s::is_volatile_v<volatile T>, "");
    static_assert(s::is_volatile_v<const volatile T>, "");
#endif
}

struct A; // incomplete

cl::sycl::cl_bool
kernel_test()
{
    test_is_volatile<void>();
    test_is_volatile<int>();
    test_is_volatile<float>();
    test_is_volatile<int*>();
    test_is_volatile<const int*>();
    test_is_volatile<char[3]>();
    test_is_volatile<char[]>();

    test_is_volatile<A>();

    static_assert(!s::is_volatile<int&>::value, "");
    static_assert(!s::is_volatile<volatile int&>::value, "");
    return true;
}

int
main(int, char**)
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
