//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// is_compound

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

#if TEST_DPCPP_BACKEND_PRESENT
constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;

template <class KernelTest, class T>
void
test_is_compound(cl::sycl::queue& deviceQueue)
{
    deviceQueue.submit([&](cl::sycl::handler& cgh) {
        cgh.single_task<KernelTest>([=]() {
            static_assert(s::is_compound<T>::value, "");
            static_assert(s::is_compound<const T>::value, "");
            static_assert(s::is_compound<volatile T>::value, "");
            static_assert(s::is_compound<const volatile T>::value, "");
#if TEST_STD_VER > 14
            static_assert(s::is_compound_v<T>, "");
            static_assert(s::is_compound_v<const T>, "");
            static_assert(s::is_compound_v<volatile T>, "");
            static_assert(s::is_compound_v<const volatile T>, "");
#endif
        });
    });
}

template <class KernelTest, class T>
void
test_is_not_compound(cl::sycl::queue& deviceQueue)
{
    deviceQueue.submit([&](cl::sycl::handler& cgh) {
        cgh.single_task<KernelTest>([=]() {
            static_assert(!s::is_compound<T>::value, "");
            static_assert(!s::is_compound<const T>::value, "");
            static_assert(!s::is_compound<volatile T>::value, "");
            static_assert(!s::is_compound<const volatile T>::value, "");
#if TEST_STD_VER > 14
            static_assert(!s::is_compound_v<T>, "");
            static_assert(!s::is_compound_v<const T>, "");
            static_assert(!s::is_compound_v<volatile T>, "");
            static_assert(!s::is_compound_v<const volatile T>, "");
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

void
kernel_test()
{
    cl::sycl::queue deviceQueue;
    test_is_compound<KernelTest1, char[3]>(deviceQueue);
    test_is_compound<KernelTest2, char[]>(deviceQueue);
    test_is_compound<KernelTest3, void*>(deviceQueue);
    test_is_compound<KernelTest4, FunctionPtr>(deviceQueue);
    test_is_compound<KernelTest5, int&>(deviceQueue);
    test_is_compound<KernelTest6, int&&>(deviceQueue);
    test_is_compound<KernelTest7, Union>(deviceQueue);
    test_is_compound<KernelTest8, Empty>(deviceQueue);
    test_is_compound<KernelTest9, incomplete_type>(deviceQueue);
    test_is_compound<KernelTest10, bit_zero>(deviceQueue);
    test_is_compound<KernelTest11, int*>(deviceQueue);
    test_is_compound<KernelTest12, const int*>(deviceQueue);
    test_is_compound<KernelTest13, Enum>(deviceQueue);

    test_is_not_compound<KernelTest14, std::nullptr_t>(deviceQueue);
    test_is_not_compound<KernelTest15, void>(deviceQueue);
    test_is_not_compound<KernelTest16, int>(deviceQueue);
    if (deviceQueue.get_device().has_extension("cl_khr_fp64"))
    {
        test_is_not_compound<KernelTest17, double>(deviceQueue);
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
