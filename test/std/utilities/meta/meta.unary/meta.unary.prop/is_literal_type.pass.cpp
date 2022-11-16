//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// is_literal_type

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
test_is_literal_type(cl::sycl::queue& deviceQueue)
{
    deviceQueue.submit([&](cl::sycl::handler& cgh) {
        cgh.single_task<KernelTest>([=]() {
            static_assert(s::is_literal_type<T>::value, "");
            static_assert(s::is_literal_type<const T>::value, "");
            static_assert(s::is_literal_type<volatile T>::value, "");
            static_assert(s::is_literal_type<const volatile T>::value, "");
#if TEST_STD_VER > 14
            static_assert(s::is_literal_type_v<T>, "");
            static_assert(s::is_literal_type_v<const T>, "");
            static_assert(s::is_literal_type_v<volatile T>, "");
            static_assert(s::is_literal_type_v<const volatile T>, "");
#endif
        });
    });
}

template <class KernelTest, class T>
void
test_is_not_literal_type(cl::sycl::queue& deviceQueue)
{
    deviceQueue.submit([&](cl::sycl::handler& cgh) {
        cgh.single_task<KernelTest>([=]() {
            static_assert(!s::is_literal_type<T>::value, "");
            static_assert(!s::is_literal_type<const T>::value, "");
            static_assert(!s::is_literal_type<volatile T>::value, "");
            static_assert(!s::is_literal_type<const volatile T>::value, "");
#if TEST_STD_VER > 14
            static_assert(!s::is_literal_type_v<T>, "");
            static_assert(!s::is_literal_type_v<const T>, "");
            static_assert(!s::is_literal_type_v<volatile T>, "");
            static_assert(!s::is_literal_type_v<const volatile T>, "");
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

enum Enum
{
    zero,
    one
};

typedef void (*FunctionPtr)();

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
class KernelTest11;
class KernelTest12;
class KernelTest13;
class KernelTest14;
class KernelTest15;
class KernelTest16;

void
kernel_test()
{
    cl::sycl::queue deviceQueue;
    test_is_literal_type<KernelTest1, std::nullptr_t>(deviceQueue);

// Before C++14, void was not a literal type
// In C++14, cv-void is a literal type
#if TEST_STD_VER < 14
    test_is_not_literal_type<KernelTest2, void>(deviceQueue);
#else
    test_is_literal_type<KernelTest3, void>(deviceQueue);
#endif

    test_is_literal_type<KernelTest4, int>(deviceQueue);
    test_is_literal_type<KernelTest5, int*>(deviceQueue);
    test_is_literal_type<KernelTest6, const int*>(deviceQueue);
    test_is_literal_type<KernelTest7, int&>(deviceQueue);
    test_is_literal_type<KernelTest8, int&&>(deviceQueue);
    test_is_literal_type<KernelTest9, char[3]>(deviceQueue);
    test_is_literal_type<KernelTest10, char[]>(deviceQueue);
    test_is_literal_type<KernelTest11, Empty>(deviceQueue);
    test_is_literal_type<KernelTest12, bit_zero>(deviceQueue);
    test_is_literal_type<KernelTest13, Union>(deviceQueue);
    test_is_literal_type<KernelTest14, Enum>(deviceQueue);
    test_is_literal_type<KernelTest15, FunctionPtr>(deviceQueue);
    if (deviceQueue.get_device().has_extension("cl_khr_fp64"))
    {
        test_is_literal_type<KernelTest16, double>(deviceQueue);
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
