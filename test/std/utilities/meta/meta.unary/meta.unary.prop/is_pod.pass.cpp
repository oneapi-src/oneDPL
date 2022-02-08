//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// is_pod

#include "oneapi_std_test_config.h"
#include "test_macros.h"
#include <CL/sycl.hpp>
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

template <class T>
void
test_is_pod(cl::sycl::queue& deviceQueue)
{
    deviceQueue.submit([&](cl::sycl::handler& cgh) {
        cgh.single_task<T>([=]() {
            static_assert(s::is_pod<T>::value, "");
            static_assert(s::is_pod<const T>::value, "");
            static_assert(s::is_pod<volatile T>::value, "");
            static_assert(s::is_pod<const volatile T>::value, "");
#if TEST_STD_VER > 14
            static_assert(s::is_pod_v<T>, "");
            static_assert(s::is_pod_v<const T>, "");
            static_assert(s::is_pod_v<volatile T>, "");
            static_assert(s::is_pod_v<const volatile T>, "");
#endif
        });
    });
}

template <class T>
void
test_is_not_pod(cl::sycl::queue deviceQueue)
{
    deviceQueue.submit([&](cl::sycl::handler& cgh) {
        cgh.single_task<T>([=]() {
            static_assert(!s::is_pod<T>::value, "");
            static_assert(!s::is_pod<const T>::value, "");
            static_assert(!s::is_pod<volatile T>::value, "");
            static_assert(!s::is_pod<const volatile T>::value, "");
#if TEST_STD_VER > 14
            static_assert(!s::is_pod_v<T>, "");
            static_assert(!s::is_pod_v<const T>, "");
            static_assert(!s::is_pod_v<volatile T>, "");
            static_assert(!s::is_pod_v<const volatile T>, "");
#endif
        });
    });
}

class Class
{
  public:
    ~Class();
};

void
kernel_test()
{
    cl::sycl::queue deviceQueue;
    test_is_not_pod<void>(deviceQueue);
    test_is_not_pod<int&>(deviceQueue);
    test_is_not_pod<Class>(deviceQueue);

    test_is_pod<int>(deviceQueue);
    test_is_pod<int*>(deviceQueue);
    test_is_pod<const int*>(deviceQueue);
    test_is_pod<char[3]>(deviceQueue);
    test_is_pod<char[]>(deviceQueue);
    if (deviceQueue.get_device().has_extension("cl_khr_fp64"))
    {
        test_is_pod<double>(deviceQueue);
    }
}

int
main(int, char**)
{
    kernel_test();
    std::cout << "Pass" << std::endl;
    return 0;
}
