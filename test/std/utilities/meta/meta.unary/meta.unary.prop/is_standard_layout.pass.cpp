//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// is_standard_layout

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

#if TEST_DPCPP_BACKEND_PRESENT
constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;

template <class T>
void
test_is_standard_layout()
{
    static_assert(s::is_standard_layout<T>::value, "");
    static_assert(s::is_standard_layout<const T>::value, "");
    static_assert(s::is_standard_layout<volatile T>::value, "");
    static_assert(s::is_standard_layout<const volatile T>::value, "");
#if TEST_STD_VER > 14
    static_assert(s::is_standard_layout_v<T>, "");
    static_assert(s::is_standard_layout_v<const T>, "");
    static_assert(s::is_standard_layout_v<volatile T>, "");
    static_assert(s::is_standard_layout_v<const volatile T>, "");
#endif
}

template <class T>
void
test_is_not_standard_layout()
{
    static_assert(!s::is_standard_layout<T>::value, "");
    static_assert(!s::is_standard_layout<const T>::value, "");
    static_assert(!s::is_standard_layout<volatile T>::value, "");
    static_assert(!s::is_standard_layout<const volatile T>::value, "");
#if TEST_STD_VER > 14
    static_assert(!s::is_standard_layout_v<T>, "");
    static_assert(!s::is_standard_layout_v<const T>, "");
    static_assert(!s::is_standard_layout_v<volatile T>, "");
    static_assert(!s::is_standard_layout_v<const volatile T>, "");
#endif
}

template <class T1, class T2>
struct pair
{
    T1 first;
    T2 second;
};

cl::sycl::cl_bool
kernel_test()
{
    test_is_standard_layout<int>();
    test_is_standard_layout<int[3]>();
    test_is_standard_layout<pair<int, float>>();

    test_is_not_standard_layout<int&>();
    return true;
}
#endif // TEST_DPCPP_BACKEND_PRESENT

int
main(int, char**)
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

    if (ret)
    {
        std::cout << "Pass" << std::endl;
    }
    else
    {
        std::cout << "Fail" << std::endl;
    }
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
