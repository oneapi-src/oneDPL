//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// is_trivial

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
test_is_trivial()
{
    static_assert(s::is_trivial<T>::value, "");
    static_assert(s::is_trivial<const T>::value, "");
    static_assert(s::is_trivial<volatile T>::value, "");
    static_assert(s::is_trivial<const volatile T>::value, "");
#if TEST_STD_VER > 14
    static_assert(s::is_trivial_v<T>, "");
    static_assert(s::is_trivial_v<const T>, "");
    static_assert(s::is_trivial_v<volatile T>, "");
    static_assert(s::is_trivial_v<const volatile T>, "");
#endif
}

template <class T>
void
test_is_not_trivial()
{
    static_assert(!s::is_trivial<T>::value, "");
    static_assert(!s::is_trivial<const T>::value, "");
    static_assert(!s::is_trivial<volatile T>::value, "");
    static_assert(!s::is_trivial<const volatile T>::value, "");
#if TEST_STD_VER > 14
    static_assert(!s::is_trivial_v<T>, "");
    static_assert(!s::is_trivial_v<const T>, "");
    static_assert(!s::is_trivial_v<volatile T>, "");
    static_assert(!s::is_trivial_v<const volatile T>, "");
#endif
}

struct A
{
};

class B
{
  public:
    B();
};

cl::sycl::cl_bool
kernel_test()
{
    test_is_trivial<int>();
    test_is_trivial<A>();

    test_is_not_trivial<int&>();
    test_is_not_trivial<volatile int&>();
    test_is_not_trivial<B>();
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

    TestUtils::exitOnError(ret);
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
