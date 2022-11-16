//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// is_nothrow_copy_constructible

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

template <class KernelTest, class T>
void
test_is_nothrow_copy_constructible(cl::sycl::queue& deviceQueue)
{
    deviceQueue.submit([&](cl::sycl::handler& cgh) {
        cgh.single_task<KernelTest>([=]() {
            static_assert(s::is_nothrow_copy_constructible<T>::value, "");
            static_assert(s::is_nothrow_copy_constructible<const T>::value, "");
#if TEST_STD_VER > 14
            static_assert(s::is_nothrow_copy_constructible_v<T>, "");
            static_assert(s::is_nothrow_copy_constructible_v<const T>, "");
#endif
        });
    });
}

template <class KernelTest, class T>
void
test_has_not_nothrow_copy_constructor(cl::sycl::queue& deviceQueue)
{
    deviceQueue.submit([&](cl::sycl::handler& cgh) {
        cgh.single_task<KernelTest>([=]() {
            static_assert(!s::is_nothrow_copy_constructible<T>::value, "");
            static_assert(!s::is_nothrow_copy_constructible<const T>::value, "");
            static_assert(!s::is_nothrow_copy_constructible<volatile T>::value, "");
            static_assert(!s::is_nothrow_copy_constructible<const volatile T>::value, "");
#if TEST_STD_VER > 14
            static_assert(!s::is_nothrow_copy_constructible_v<T>, "");
            static_assert(!s::is_nothrow_copy_constructible_v<const T>, "");
            static_assert(!s::is_nothrow_copy_constructible_v<volatile T>, "");
            static_assert(!s::is_nothrow_copy_constructible_v<const volatile T>, "");
#endif
        });
    });
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
    A(const A&);
};

class KernelTest1;
class KernelTest2;
class KernelTest3;
class KernelTest4;
class KernelTest5;
class KernelTest6;
class KernelTest7;
class KernelTest8;
class KernelTest9;
class KernelTest10;

void
kernel_test()
{
    cl::sycl::queue deviceQueue;
    test_has_not_nothrow_copy_constructor<KernelTest1, void>(deviceQueue);
    test_has_not_nothrow_copy_constructor<KernelTest2, A>(deviceQueue);

    test_is_nothrow_copy_constructible<KernelTest3, int&>(deviceQueue);
    test_is_nothrow_copy_constructible<KernelTest4, Union>(deviceQueue);
    test_is_nothrow_copy_constructible<KernelTest5, Empty>(deviceQueue);
    test_is_nothrow_copy_constructible<KernelTest6, int>(deviceQueue);
    test_is_nothrow_copy_constructible<KernelTest7, int*>(deviceQueue);
    test_is_nothrow_copy_constructible<KernelTest8, const int*>(deviceQueue);
    test_is_nothrow_copy_constructible<KernelTest9, bit_zero>(deviceQueue);
    if (deviceQueue.get_device().has_extension("cl_khr_fp64"))
    {
        test_is_nothrow_copy_constructible<KernelTest10, double>(deviceQueue);
    }
}
#endif // TEST_DPCPP_BACKEND_PRESENT

int
main(int, char**)
{
#if TEST_DPCPP_BACKEND_PRESENT
    kernel_test();
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
