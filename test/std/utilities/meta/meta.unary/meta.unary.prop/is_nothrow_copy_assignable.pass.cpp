//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// is_nothrow_copy_assignable

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

template <class T>
void
test_has_nothrow_assign(cl::sycl::queue& deviceQueue)
{
    deviceQueue.submit([&](cl::sycl::handler& cgh) {
        cgh.single_task<T>([=]() {
            static_assert(s::is_nothrow_copy_assignable<T>::value, "");
#if TEST_STD_VER > 14
            static_assert(s::is_nothrow_copy_assignable_v<T>, "");
#endif
        });
    });
}

template <class T>
void
test_has_not_nothrow_assign(cl::sycl::queue& deviceQueue)
{
    deviceQueue.submit([&](cl::sycl::handler& cgh) {
        cgh.single_task<T>([=]() {
            static_assert(!s::is_nothrow_copy_assignable<T>::value, "");
#if TEST_STD_VER > 14
            static_assert(!s::is_nothrow_copy_assignable_v<T>, "");
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
    A&
    operator=(const A&);
};

void
kernel_test()
{
    cl::sycl::queue deviceQueue;
    test_has_nothrow_assign<int&>(deviceQueue);
    test_has_nothrow_assign<Union>(deviceQueue);
    test_has_nothrow_assign<Empty>(deviceQueue);
    test_has_nothrow_assign<int>(deviceQueue);
    test_has_nothrow_assign<int*>(deviceQueue);
    test_has_nothrow_assign<const int*>(deviceQueue);
    test_has_nothrow_assign<bit_zero>(deviceQueue);
    if (deviceQueue.get_device().has_extension("cl_khr_fp64"))
    {
        test_has_nothrow_assign<double>(deviceQueue);
    }

    test_has_not_nothrow_assign<const int>(deviceQueue);
    test_has_not_nothrow_assign<void>(deviceQueue);
    test_has_not_nothrow_assign<A>(deviceQueue);
}

int
main(int, char**)
{
    kernel_test();
    std::cout << "Pass" << std::endl;
    return 0;
}
