//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// is_nothrow_default_constructible

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
test_is_nothrow_default_constructible()
{
    static_assert(s::is_nothrow_default_constructible<T>::value, "");
    static_assert(s::is_nothrow_default_constructible<const T>::value, "");
    static_assert(s::is_nothrow_default_constructible<volatile T>::value, "");
    static_assert(s::is_nothrow_default_constructible<const volatile T>::value, "");
#if TEST_STD_VER > 14
    static_assert(s::is_nothrow_default_constructible_v<T>, "");
    static_assert(s::is_nothrow_default_constructible_v<const T>, "");
    static_assert(s::is_nothrow_default_constructible_v<volatile T>, "");
    static_assert(s::is_nothrow_default_constructible_v<const volatile T>, "");
#endif
}

template <class T>
void
test_has_not_nothrow_default_constructor()
{
    static_assert(!s::is_nothrow_default_constructible<T>::value, "");
    static_assert(!s::is_nothrow_default_constructible<const T>::value, "");
    static_assert(!s::is_nothrow_default_constructible<volatile T>::value, "");
    static_assert(!s::is_nothrow_default_constructible<const volatile T>::value, "");
#if TEST_STD_VER > 14
    static_assert(!s::is_nothrow_default_constructible_v<T>, "");
    static_assert(!s::is_nothrow_default_constructible_v<const T>, "");
    static_assert(!s::is_nothrow_default_constructible_v<volatile T>, "");
    static_assert(!s::is_nothrow_default_constructible_v<const volatile T>, "");
#endif
}

class Empty
{
};

union Union {
};

struct bit_zero
{
    int : 0;
};

struct A
{
    A();
};

cl::sycl::cl_bool
kernel_test()
{
    test_has_not_nothrow_default_constructor<void>();
    test_has_not_nothrow_default_constructor<int&>();
    test_has_not_nothrow_default_constructor<A>();

    test_is_nothrow_default_constructible<Union>();
    test_is_nothrow_default_constructible<Empty>();
    test_is_nothrow_default_constructible<int>();
    test_is_nothrow_default_constructible<float>();
    test_is_nothrow_default_constructible<int*>();
    test_is_nothrow_default_constructible<const int*>();
    test_is_nothrow_default_constructible<char[3]>();
    test_is_nothrow_default_constructible<bit_zero>();
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
