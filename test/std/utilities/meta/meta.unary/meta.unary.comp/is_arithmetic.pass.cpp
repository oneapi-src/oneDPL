//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// is_arithmetic

#include "oneapi_std_test_config.h"
#include "test_macros.h"

#include <iostream>

#ifdef USE_ONEAPI_STD
#    include _ONEAPI_STD_TEST_HEADER(type_traits)
#    include _ONEAPI_STD_TEST_HEADER(cstddef)
namespace s = oneapi_cpp_ns;
#else
#    include <cstddef>
#    include <type_traits>
namespace s = std;
#endif

#if TEST_DPCPP_BACKEND_PRESENT
constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;

template <class KernelTest, class T>
void
test_is_arithmetic(cl::sycl::queue& deviceQueue)
{
    deviceQueue.submit([&](cl::sycl::handler& cgh) {
        cgh.single_task<KernelTest>([=]() {
            static_assert(s::is_arithmetic<T>::value, "");
            static_assert(s::is_arithmetic<const T>::value, "");
            static_assert(s::is_arithmetic<volatile T>::value, "");
            static_assert(s::is_arithmetic<const volatile T>::value, "");
#if TEST_STD_VER > 14
            static_assert(s::is_arithmetic_v<T>, "");
            static_assert(s::is_arithmetic_v<const T>, "");
            static_assert(s::is_arithmetic_v<volatile T>, "");
            static_assert(s::is_arithmetic_v<const volatile T>, "");
#endif
        });
    });
}

template <class KernelTest, class T>
void
test_is_not_arithmetic(cl::sycl::queue& deviceQueue)
{
    deviceQueue.submit([&](cl::sycl::handler& cgh) {
        cgh.single_task<KernelTest>([=]() {
            static_assert(!s::is_arithmetic<T>::value, "");
            static_assert(!s::is_arithmetic<const T>::value, "");
            static_assert(!s::is_arithmetic<volatile T>::value, "");
            static_assert(!s::is_arithmetic<const volatile T>::value, "");
#if TEST_STD_VER > 14
            static_assert(!s::is_arithmetic_v<T>, "");
            static_assert(!s::is_arithmetic_v<const T>, "");
            static_assert(!s::is_arithmetic_v<volatile T>, "");
            static_assert(!s::is_arithmetic_v<const volatile T>, "");
#endif
        });
    });
}

class incomplete_type;

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
class KernelTest17;
class KernelTest18;
class KernelTest19;
class KernelTest20;
class KernelTest21;
class KernelTest22;
class KernelTest23;
class KernelTest24;
class KernelTest25;
class KernelTest26;

void
kernel_test()
{
    cl::sycl::queue deviceQueue;
    test_is_arithmetic<KernelTest1, short>(deviceQueue);
    test_is_arithmetic<KernelTest2, unsigned short>(deviceQueue);
    test_is_arithmetic<KernelTest3, int>(deviceQueue);
    test_is_arithmetic<KernelTest4, unsigned int>(deviceQueue);
    test_is_arithmetic<KernelTest5, long>(deviceQueue);
    test_is_arithmetic<KernelTest6, unsigned long>(deviceQueue);
    test_is_arithmetic<KernelTest7, bool>(deviceQueue);
    test_is_arithmetic<KernelTest8, char>(deviceQueue);
    test_is_arithmetic<KernelTest9, signed char>(deviceQueue);
    test_is_arithmetic<KernelTest10, unsigned char>(deviceQueue);
    test_is_arithmetic<KernelTest11, wchar_t>(deviceQueue);
    if (deviceQueue.get_device().has_extension("cl_khr_fp64"))
    {
        test_is_arithmetic<KernelTest12, double>(deviceQueue);
    }

    test_is_not_arithmetic<KernelTest13, s::nullptr_t>(deviceQueue);
    test_is_not_arithmetic<KernelTest14, void>(deviceQueue);
    test_is_not_arithmetic<KernelTest15, int&>(deviceQueue);
    test_is_not_arithmetic<KernelTest16, int&&>(deviceQueue);
    test_is_not_arithmetic<KernelTest17, int*>(deviceQueue);
    test_is_not_arithmetic<KernelTest18, const int*>(deviceQueue);
    test_is_not_arithmetic<KernelTest19, char[3]>(deviceQueue);
    test_is_not_arithmetic<KernelTest20, char[]>(deviceQueue);
    test_is_not_arithmetic<KernelTest21, Union>(deviceQueue);
    test_is_not_arithmetic<KernelTest22, Enum>(deviceQueue);
    test_is_not_arithmetic<KernelTest23, FunctionPtr>(deviceQueue);
    test_is_not_arithmetic<KernelTest24, Empty>(deviceQueue);
    test_is_not_arithmetic<KernelTest25, incomplete_type>(deviceQueue);
    test_is_not_arithmetic<KernelTest26, bit_zero>(deviceQueue);
}
#endif // TEST_DPCPP_BACKEND_PRESENT

int
main(int, char**)
{
#if TEST_DPCPP_BACKEND_PRESENT
    kernel_test();
    TestUtils::exitOnError(true);
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
