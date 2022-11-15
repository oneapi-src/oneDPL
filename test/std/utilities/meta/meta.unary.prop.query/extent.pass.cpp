//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// extent

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

constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;

template <class KernelTest, class T, unsigned A>
void
test_extent(cl::sycl::queue& deviceQueue)
{
    deviceQueue.submit([&](cl::sycl::handler& cgh) {
        cgh.single_task<KernelTest>([=]() {
            static_assert((s::extent<T>::value == A), "");
            static_assert((s::extent<const T>::value == A), "");
            static_assert((s::extent<volatile T>::value == A), "");
            static_assert((s::extent<const volatile T>::value == A), "");
#if TEST_STD_VER > 14
            static_assert((s::extent_v<T> == A), "");
            static_assert((s::extent_v<const T> == A), "");
            static_assert((s::extent_v<volatile T> == A), "");
            static_assert((s::extent_v<const volatile T> == A), "");
#endif
        });
    });
}

template <class KernelTest, class T, unsigned A>
void
test_extent1(cl::sycl::queue& deviceQueue)
{
    deviceQueue.submit([&](cl::sycl::handler& cgh) {
        cgh.single_task<KernelTest>([=]() {
            static_assert((s::extent<T, 1>::value == A), "");
            static_assert((s::extent<const T, 1>::value == A), "");
            static_assert((s::extent<volatile T, 1>::value == A), "");
            static_assert((s::extent<const volatile T, 1>::value == A), "");
#if TEST_STD_VER > 14
            static_assert((s::extent_v<T, 1> == A), "");
            static_assert((s::extent_v<const T, 1> == A), "");
            static_assert((s::extent_v<volatile T, 1> == A), "");
            static_assert((s::extent_v<const volatile T, 1> == A), "");
#endif
        });
    });
}

class Class
{
  public:
    ~Class();
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

void
kernel_test()
{
    cl::sycl::queue deviceQueue;
    test_extent<KernelTest1, void, 0>(deviceQueue);
    test_extent<KernelTest2, int&, 0>(deviceQueue);
    test_extent<KernelTest3, Class, 0>(deviceQueue);
    test_extent<KernelTest4, int*, 0>(deviceQueue);
    test_extent<KernelTest5, const int*, 0>(deviceQueue);
    test_extent<KernelTest6, int, 0>(deviceQueue);
    test_extent<KernelTest7, bool, 0>(deviceQueue);
    test_extent<KernelTest8, unsigned, 0>(deviceQueue);
    if (deviceQueue.get_device().has_extension("cl_khr_fp64"))
    {
        test_extent<KernelTest9, double, 0>(deviceQueue);
    }

    test_extent<KernelTest10, int[2], 2>(deviceQueue);
    test_extent<KernelTest11, int[2][4], 2>(deviceQueue);
    test_extent<KernelTest12, int[][4], 0>(deviceQueue);

    test_extent1<KernelTest13, int, 0>(deviceQueue);
    test_extent1<KernelTest14, int[2], 0>(deviceQueue);
    test_extent1<KernelTest15, int[2][4], 4>(deviceQueue);
    test_extent1<KernelTest16, int[][4], 4>(deviceQueue);
}

int
main()
{
    kernel_test();
    std::cout << "Pass" << std::endl;

    return 0;
}
