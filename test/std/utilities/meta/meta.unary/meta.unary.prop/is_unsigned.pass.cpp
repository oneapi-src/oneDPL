//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// is_unsigned

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
constexpr sycl::access::mode sycl_read = sycl::access::mode::read;
constexpr sycl::access::mode sycl_write = sycl::access::mode::write;

template <class T>
void
test_is_unsigned()
{
    static_assert(s::is_unsigned<T>::value, "");
    static_assert(s::is_unsigned<const T>::value, "");
    static_assert(s::is_unsigned<volatile T>::value, "");
    static_assert(s::is_unsigned<const volatile T>::value, "");
#if TEST_STD_VER > 14
    static_assert(s::is_unsigned_v<T>, "");
    static_assert(s::is_unsigned_v<const T>, "");
    static_assert(s::is_unsigned_v<volatile T>, "");
    static_assert(s::is_unsigned_v<const volatile T>, "");
#endif
}

template <class T>
void
test_is_not_unsigned()
{
    static_assert(!s::is_unsigned<T>::value, "");
    static_assert(!s::is_unsigned<const T>::value, "");
    static_assert(!s::is_unsigned<volatile T>::value, "");
    static_assert(!s::is_unsigned<const volatile T>::value, "");
#if TEST_STD_VER > 14
    static_assert(!s::is_unsigned_v<T>, "");
    static_assert(!s::is_unsigned_v<const T>, "");
    static_assert(!s::is_unsigned_v<volatile T>, "");
    static_assert(!s::is_unsigned_v<const volatile T>, "");
#endif
}

class Class
{
  public:
    ~Class();
};

struct A; // incomplete

sycl::cl_bool
kernel_test()
{
    test_is_not_unsigned<void>();
    test_is_not_unsigned<int&>();
    test_is_not_unsigned<Class>();
    test_is_not_unsigned<int*>();
    test_is_not_unsigned<const int*>();
    test_is_not_unsigned<char[3]>();
    test_is_not_unsigned<char[]>();
    test_is_not_unsigned<int>();
    test_is_not_unsigned<float>();
    test_is_not_unsigned<A>();

    test_is_unsigned<bool>();
    test_is_unsigned<unsigned>();

    return true;
}
#endif // TEST_DPCPP_BACKEND_PRESENT

int
main(int, char**)
{
#if TEST_DPCPP_BACKEND_PRESENT
    sycl::queue deviceQueue;
    sycl::cl_bool ret = false;
    sycl::range<1> numOfItems{1};
    {
        sycl::buffer<sycl::cl_bool, 1> buffer1(&ret, numOfItems);
        deviceQueue.submit([&](sycl::handler& cgh) {
            auto ret_access = buffer1.get_access<sycl_write>(cgh);
            cgh.single_task<class KernelTest>([=]() { ret_access[0] = kernel_test(); });
        });
    }

    TestUtils::exitOnError(ret);
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
