//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// is_nothrow_assignable

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

template <class T, class U>
void
test_is_nothrow_assignable()
{
    static_assert((s::is_nothrow_assignable<T, U>::value), "");
#if TEST_STD_VER > 14
    static_assert((s::is_nothrow_assignable_v<T, U>), "");
#endif
}

template <class T, class U>
void
test_is_not_nothrow_assignable()
{
    static_assert((!s::is_nothrow_assignable<T, U>::value), "");
#if TEST_STD_VER > 14
    static_assert((!s::is_nothrow_assignable_v<T, U>), "");
#endif
}

struct A
{
};

struct B
{
    void operator=(A);
};

struct C
{
    void
    operator=(C&); // not const
};

cl::sycl::cl_bool
kernel_test()
{
    test_is_nothrow_assignable<int&, int&>();
    test_is_nothrow_assignable<int&, int>();
    test_is_nothrow_assignable<int&, float>();

    test_is_not_nothrow_assignable<int, int&>();
    test_is_not_nothrow_assignable<int, int>();
    test_is_not_nothrow_assignable<B, A>();
    test_is_not_nothrow_assignable<A, B>();
    test_is_not_nothrow_assignable<C, C&>();
    return true;
}

int
main(int, char**)
{
    cl::sycl::queue deviceQueue;
    cl::sycl::cl_bool ret = false;
    cl::sycl::range<1> numOfItems{1};
    {
        cl::sycl::buffer<cl::sycl::cl_bool, 1> buffer1(&ret, numOfItems);
        deviceQueue.submit([&](cl::sycl::handler& cgh) {
            auto ret_access = buffer1.get_access<sycl_write>(cgh);
            cgh.single_task<class KernelTest>([=]() { ret_access[0] = kernel_test(); });
        });
    }

    if (ret)
    {
        std::cout << "Pass" << std::endl;
    }
    else
    {
        std::cout << "Fail" << std::endl;
    }

    return 0;
}
