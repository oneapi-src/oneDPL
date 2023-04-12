//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// remove_extent

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
constexpr sycl::access::mode sycl_read = sycl::access::mode::read;
constexpr sycl::access::mode sycl_write = sycl::access::mode::write;

enum Enum
{
    zero,
    one_
};

template <class T, class U>
void
test_remove_extent()
{
    ASSERT_SAME_TYPE(U, typename s::remove_extent<T>::type);
#if TEST_STD_VER > 11
    ASSERT_SAME_TYPE(U, s::remove_extent_t<T>);
#endif
}

sycl::cl_bool
kernel_test()
{
    test_remove_extent<int, int>();
    test_remove_extent<const Enum, const Enum>();
    test_remove_extent<int[], int>();
    test_remove_extent<const int[], const int>();
    test_remove_extent<int[3], int>();
    test_remove_extent<const int[3], const int>();
    test_remove_extent<int[][3], int[3]>();
    test_remove_extent<const int[][3], const int[3]>();
    test_remove_extent<int[2][3], int[3]>();
    test_remove_extent<const int[2][3], const int[3]>();
    test_remove_extent<int[1][2][3], int[2][3]>();
    test_remove_extent<const int[1][2][3], const int[2][3]>();

    return true;
}

class KernelTest;
#endif // TEST_DPCPP_BACKEND_PRESENT

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    sycl::queue deviceQueue;
    sycl::cl_bool ret = false;
    sycl::range<1> numOfItems{1};
    {
        sycl::buffer<sycl::cl_bool, 1> buffer1(&ret, numOfItems);
        deviceQueue.submit([&](sycl::handler& cgh) {
            auto ret_access = buffer1.get_access<sycl_write>(cgh);
            cgh.single_task<class KernelTest>([=]() { ret_access[0] = kernel_test(); });
        });
    }

    TestUtils::exitOnError(ret);
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
