//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// is_reference

#include "oneapi_std_test_config.h"
#include "test_macros.h"

#include <iostream>

#ifdef USE_ONEAPI_STD
#    include _ONEAPI_STD_TEST_HEADER(type_traits)
#    include _ONEAPI_STD_TEST_HEADER(cstddef)
namespace s = oneapi_cpp_ns;
#else
#    include <type_traits>
#    include <cstddef> // for std::nullptr_t
namespace s = std;
#endif

constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;

template <class KernelTest, class T>
void
test_is_reference(cl::sycl::queue& deviceQueue)
{
    deviceQueue.submit([&](cl::sycl::handler& cgh) {
        cgh.single_task<KernelTest>([=]() {
            static_assert(s::is_reference<T>::value, "");
            static_assert(s::is_reference<const T>::value, "");
            static_assert(s::is_reference<volatile T>::value, "");
            static_assert(s::is_reference<const volatile T>::value, "");
#if TEST_STD_VER > 14
            static_assert(s::is_reference_v<T>, "");
            static_assert(s::is_reference_v<const T>, "");
            static_assert(s::is_reference_v<volatile T>, "");
            static_assert(s::is_reference_v<const volatile T>, "");
#endif
        });
    });
}

template <class KernelTest, class T>
void
test_is_not_reference(cl::sycl::queue& deviceQueue)
{
    deviceQueue.submit([&](cl::sycl::handler& cgh) {
        cgh.single_task<KernelTest>([=]() {
            static_assert(!s::is_reference<T>::value, "");
            static_assert(!s::is_reference<const T>::value, "");
            static_assert(!s::is_reference<volatile T>::value, "");
            static_assert(!s::is_reference<const volatile T>::value, "");
#if TEST_STD_VER > 14
            static_assert(!s::is_reference_v<T>, "");
            static_assert(!s::is_reference_v<const T>, "");
            static_assert(!s::is_reference_v<volatile T>, "");
            static_assert(!s::is_reference_v<const volatile T>, "");
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

void
kernel_test()
{
    cl::sycl::queue deviceQueue;
    test_is_reference<KernelTest1, int&>(deviceQueue);
    test_is_reference<KernelTest2, int&&>(deviceQueue);

    test_is_not_reference<KernelTest3, std::nullptr_t>(deviceQueue);
    test_is_not_reference<KernelTest4, void>(deviceQueue);
    test_is_not_reference<KernelTest5, int>(deviceQueue);
    test_is_not_reference<KernelTest6, char[3]>(deviceQueue);
    test_is_not_reference<KernelTest7, char[]>(deviceQueue);
    test_is_not_reference<KernelTest8, void*>(deviceQueue);
    test_is_not_reference<KernelTest9, FunctionPtr>(deviceQueue);
    test_is_not_reference<KernelTest10, Union>(deviceQueue);
    test_is_not_reference<KernelTest11, incomplete_type>(deviceQueue);
    test_is_not_reference<KernelTest12, Empty>(deviceQueue);
    test_is_not_reference<KernelTest13, bit_zero>(deviceQueue);
    test_is_not_reference<KernelTest14, int*>(deviceQueue);
    test_is_not_reference<KernelTest15, const int*>(deviceQueue);
    test_is_not_reference<KernelTest16, Enum>(deviceQueue);
    test_is_not_reference<KernelTest17, int(int)>(deviceQueue);
    test_is_not_reference<KernelTest18, int Empty::*>(deviceQueue);
    test_is_not_reference<KernelTest19, void (Empty::*)(int)>(deviceQueue);
    if (deviceQueue.get_device().has_extension("cl_khr_fp64"))
    {
        test_is_not_reference<KernelTest20, double>(deviceQueue);
    }
}

int
main(int, char**)
{
    kernel_test();
    std::cout << "Pass" << std::endl;
    return 0;
}
