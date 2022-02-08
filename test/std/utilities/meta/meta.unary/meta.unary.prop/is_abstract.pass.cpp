//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// is_abstract

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
test_is_not_abstract(cl::sycl::queue& deviceQueue)
{
    deviceQueue.submit([&](cl::sycl::handler& cgh) {
        cgh.single_task<T>([=]() {
            static_assert(!s::is_abstract<T>::value, "");
            static_assert(!s::is_abstract<const T>::value, "");
            static_assert(!s::is_abstract<volatile T>::value, "");
            static_assert(!s::is_abstract<const volatile T>::value, "");
#if TEST_STD_VER > 14
            static_assert(!s::is_abstract_v<T>, "");
            static_assert(!s::is_abstract_v<const T>, "");
            static_assert(!s::is_abstract_v<volatile T>, "");
            static_assert(!s::is_abstract_v<const volatile T>, "");
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

void
kernel_test()
{
    cl::sycl::queue deviceQueue;
    test_is_not_abstract<void>(deviceQueue);
    test_is_not_abstract<int&>(deviceQueue);
    test_is_not_abstract<int>(deviceQueue);
    test_is_not_abstract<int*>(deviceQueue);
    test_is_not_abstract<const int*>(deviceQueue);
    test_is_not_abstract<char[3]>(deviceQueue);
    test_is_not_abstract<char[]>(deviceQueue);
    test_is_not_abstract<Union>(deviceQueue);
    test_is_not_abstract<Empty>(deviceQueue);
    test_is_not_abstract<bit_zero>(deviceQueue);
    if (deviceQueue.get_device().has_extension("cl_khr_fp64"))
    {
        test_is_not_abstract<double>(deviceQueue);
    }
}

int
main(int, char**)
{
    kernel_test();
    std::cout << "Pass" << std::endl;
    return 0;
}
