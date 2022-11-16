//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// is_signed

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
test_is_signed(cl::sycl::queue deviceQueue)
{
    deviceQueue.submit([&](cl::sycl::handler& cgh) {
        cgh.single_task<KernelTest>([=]() {
            static_assert(s::is_signed<T>::value, "");
            static_assert(s::is_signed<const T>::value, "");
            static_assert(s::is_signed<volatile T>::value, "");
            static_assert(s::is_signed<const volatile T>::value, "");
#if TEST_STD_VER > 14
            static_assert(s::is_signed_v<T>, "");
            static_assert(s::is_signed_v<const T>, "");
            static_assert(s::is_signed_v<volatile T>, "");
            static_assert(s::is_signed_v<const volatile T>, "");
#endif
        });
    });
}

template <class KernelTest, class T>
void
test_is_not_signed(cl::sycl::queue& deviceQueue)
{
    deviceQueue.submit([&](cl::sycl::handler& cgh) {
        cgh.single_task<KernelTest>([=]() {
            static_assert(!s::is_signed<T>::value, "");
            static_assert(!s::is_signed<const T>::value, "");
            static_assert(!s::is_signed<volatile T>::value, "");
            static_assert(!s::is_signed<const volatile T>::value, "");
#if TEST_STD_VER > 14
            static_assert(!s::is_signed_v<T>, "");
            static_assert(!s::is_signed_v<const T>, "");
            static_assert(!s::is_signed_v<volatile T>, "");
            static_assert(!s::is_signed_v<const volatile T>, "");
#endif
        });
    });
}

class Class
{
  public:
    ~Class();
};

struct A; // incomplete

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

void
kernel_test()
{
    cl::sycl::queue deviceQueue;
    test_is_not_signed<KernelTest1, void>(deviceQueue);
    test_is_not_signed<KernelTest2, int&>(deviceQueue);
    test_is_not_signed<KernelTest3, Class>(deviceQueue);
    test_is_not_signed<KernelTest4, int*>(deviceQueue);
    test_is_not_signed<KernelTest5, const int*>(deviceQueue);
    test_is_not_signed<KernelTest6, char[3]>(deviceQueue);
    test_is_not_signed<KernelTest7, char[]>(deviceQueue);
    test_is_not_signed<KernelTest8, bool>(deviceQueue);
    test_is_not_signed<KernelTest9, unsigned>(deviceQueue);
    test_is_not_signed<KernelTest10, A>(deviceQueue);

    test_is_signed<KernelTest11, int>(deviceQueue);
    if (deviceQueue.get_device().has_extension("cl_khr_fp64"))
    {
        test_is_signed<KernelTest12, double>(deviceQueue);
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
