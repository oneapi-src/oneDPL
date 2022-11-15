//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// template <class T, class... Args>
//   struct is_nothrow_constructible;

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

template <class T>
void
test_is_nothrow_constructible()
{
    static_assert((s::is_nothrow_constructible<T>::value), "");
#if TEST_STD_VER > 14
    static_assert((s::is_nothrow_constructible_v<T>), "");
#endif
}

template <class T, class A0>
void
test_is_nothrow_constructible()
{
    static_assert((s::is_nothrow_constructible<T, A0>::value), "");
#if TEST_STD_VER > 14
    static_assert((s::is_nothrow_constructible_v<T, A0>), "");
#endif
}

template <class T>
void
test_is_not_nothrow_constructible()
{
    static_assert((!s::is_nothrow_constructible<T>::value), "");
#if TEST_STD_VER > 14
    static_assert((!s::is_nothrow_constructible_v<T>), "");
#endif
}

template <class T, class A0>
void
test_is_not_nothrow_constructible()
{
    static_assert((!s::is_nothrow_constructible<T, A0>::value), "");
#if TEST_STD_VER > 14
    static_assert((!s::is_nothrow_constructible_v<T, A0>), "");
#endif
}

template <class T, class A0, class A1>
void
test_is_not_nothrow_constructible()
{
    static_assert((!s::is_nothrow_constructible<T, A0, A1>::value), "");
#if TEST_STD_VER > 14
    static_assert((!s::is_nothrow_constructible_v<T, A0, A1>), "");
#endif
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
    A(const A&);
};

struct C
{
    C(C&); // not const
    void
    operator=(C&); // not const
};

struct Tuple
{
    Tuple(Empty&&) noexcept {}
};

cl::sycl::cl_bool
kernel_test()
{
    test_is_nothrow_constructible<int>();
    test_is_nothrow_constructible<int, const int&>();
    test_is_nothrow_constructible<Empty>();
    test_is_nothrow_constructible<Empty, const Empty&>();

    test_is_not_nothrow_constructible<A, int>();
    test_is_not_nothrow_constructible<A, int, float>();
    test_is_not_nothrow_constructible<A>();
    test_is_not_nothrow_constructible<C>();
    test_is_nothrow_constructible<Tuple&&, Empty>(); // See bug #19616.

    static_assert(!s::is_constructible<Tuple&, Empty>::value, "");
    test_is_not_nothrow_constructible<Tuple&, Empty>(); // See bug #19616.
    return true;
}

#endif // TEST_DPCPP_BACKEND_PRESENT

int
main(int, char**)
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
