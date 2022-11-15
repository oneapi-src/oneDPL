//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// is_base_of

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

template <class T, class U>
void
test_is_base_of()
{
    static_assert((s::is_base_of<T, U>::value), "");
    static_assert((s::is_base_of<const T, U>::value), "");
    static_assert((s::is_base_of<T, const U>::value), "");
    static_assert((s::is_base_of<const T, const U>::value), "");
#if TEST_STD_VER > 14
    static_assert((s::is_base_of_v<T, U>), "");
    static_assert((s::is_base_of_v<const T, U>), "");
    static_assert((s::is_base_of_v<T, const U>), "");
    static_assert((s::is_base_of_v<const T, const U>), "");
#endif
}

template <class T, class U>
void
test_is_not_base_of()
{
    static_assert((!s::is_base_of<T, U>::value), "");
}

struct B
{
};
struct B1 : B
{
};
struct B2 : B
{
};
struct D : private B1, private B2
{
};
struct I0; // incomplete

cl::sycl::cl_bool
kernel_test()
{
    test_is_base_of<B, D>();
    test_is_base_of<B1, D>();
    test_is_base_of<B2, D>();
    test_is_base_of<B, B1>();
    test_is_base_of<B, B2>();
    test_is_base_of<B, B>();

    test_is_not_base_of<D, B>();
    test_is_not_base_of<B&, D&>();
    test_is_not_base_of<B[3], D[3]>();
    test_is_not_base_of<int, int>();

    //  A scalar is never the base class of anything (including incomplete types)
    test_is_not_base_of<int, B>();
    test_is_not_base_of<int, B1>();
    test_is_not_base_of<int, B2>();
    test_is_not_base_of<int, D>();
    test_is_not_base_of<int, I0>();

    //  A scalar never has base classes (including incomplete types)
    test_is_not_base_of<B, int>();
    test_is_not_base_of<B1, int>();
    test_is_not_base_of<B2, int>();
    test_is_not_base_of<D, int>();
    test_is_not_base_of<I0, int>();

    return true;
}

class KernelTest;
#endif // TEST_DPCPP_BACKEND_PRESENT

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
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

    TestUtils::exitOnError(ret);
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
