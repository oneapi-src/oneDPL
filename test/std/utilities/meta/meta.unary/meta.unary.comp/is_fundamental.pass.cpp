//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// is_fundamental

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
test_is_fundamental(cl::sycl::queue& deviceQueue)
{
    deviceQueue.submit([&](cl::sycl::handler& cgh) {
        cgh.single_task<KernelTest>([=]() {
            static_assert(s::is_fundamental<T>::value, "");
            static_assert(s::is_fundamental<const T>::value, "");
            static_assert(s::is_fundamental<volatile T>::value, "");
            static_assert(s::is_fundamental<const volatile T>::value, "");
#if TEST_STD_VER > 14
            static_assert(s::is_fundamental_v<T>, "");
            static_assert(s::is_fundamental_v<const T>, "");
            static_assert(s::is_fundamental_v<volatile T>, "");
            static_assert(s::is_fundamental_v<const volatile T>, "");
#endif
        });
    });
}

template <class KernelTest, class T>
void
test_is_not_fundamental(cl::sycl::queue& deviceQueue)
{
    deviceQueue.submit([&](cl::sycl::handler& cgh) {
        cgh.single_task<KernelTest>([=]() {
            static_assert(!s::is_fundamental<T>::value, "");
            static_assert(!s::is_fundamental<const T>::value, "");
            static_assert(!s::is_fundamental<volatile T>::value, "");
            static_assert(!s::is_fundamental<const volatile T>::value, "");
#if TEST_STD_VER > 14
            static_assert(!s::is_fundamental_v<T>, "");
            static_assert(!s::is_fundamental_v<const T>, "");
            static_assert(!s::is_fundamental_v<volatile T>, "");
            static_assert(!s::is_fundamental_v<const volatile T>, "");
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
class KernelTest27;
class KernelTest28;
class KernelTest29;
class KernelTest30;
class KernelTest31;

void
kernel_test()
{
    cl::sycl::queue deviceQueue;
    test_is_fundamental<KernelTest1, s::nullptr_t>(deviceQueue);
    test_is_fundamental<KernelTest2, void>(deviceQueue);
    test_is_fundamental<KernelTest3, short>(deviceQueue);
    test_is_fundamental<KernelTest4, unsigned short>(deviceQueue);
    test_is_fundamental<KernelTest5, int>(deviceQueue);
    test_is_fundamental<KernelTest6, unsigned int>(deviceQueue);
    test_is_fundamental<KernelTest7, long>(deviceQueue);
    test_is_fundamental<KernelTest8, unsigned long>(deviceQueue);
    test_is_fundamental<KernelTest9, long long>(deviceQueue);
    test_is_fundamental<KernelTest10, unsigned long long>(deviceQueue);
    test_is_fundamental<KernelTest11, bool>(deviceQueue);
    test_is_fundamental<KernelTest12, char>(deviceQueue);
    test_is_fundamental<KernelTest13, signed char>(deviceQueue);
    test_is_fundamental<KernelTest14, unsigned char>(deviceQueue);
    test_is_fundamental<KernelTest15, wchar_t>(deviceQueue);
    test_is_fundamental<KernelTest16, float>(deviceQueue);
    test_is_fundamental<KernelTest17, char16_t>(deviceQueue);
    test_is_fundamental<KernelTest18, char32_t>(deviceQueue);
    if (deviceQueue.get_device().has_extension("cl_khr_fp64"))
    {
        test_is_fundamental<KernelTest19, double>(deviceQueue);
    }

    test_is_not_fundamental<KernelTest20, Enum>(deviceQueue);
    test_is_not_fundamental<KernelTest21, char[3]>(deviceQueue);
    test_is_not_fundamental<KernelTest22, char[]>(deviceQueue);
    test_is_not_fundamental<KernelTest23, void*>(deviceQueue);
    test_is_not_fundamental<KernelTest24, int&>(deviceQueue);
    test_is_not_fundamental<KernelTest25, int&&>(deviceQueue);
    test_is_not_fundamental<KernelTest26, Union>(deviceQueue);
    test_is_not_fundamental<KernelTest27, Empty>(deviceQueue);
    test_is_not_fundamental<KernelTest28, incomplete_type>(deviceQueue);
    test_is_not_fundamental<KernelTest29, bit_zero>(deviceQueue);
    test_is_not_fundamental<KernelTest30, int*>(deviceQueue);
    test_is_not_fundamental<KernelTest31, const int*>(deviceQueue);
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
