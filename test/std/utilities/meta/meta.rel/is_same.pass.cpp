//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// is_same

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
test_is_same()
{
    static_assert((s::is_same<T, U>::value), "");
    static_assert((!s::is_same<const T, U>::value), "");
    static_assert((!s::is_same<T, const U>::value), "");
    static_assert((s::is_same<const T, const U>::value), "");
#if TEST_STD_VER > 14
    static_assert((s::is_same_v<T, U>), "");
    static_assert((!s::is_same_v<const T, U>), "");
    static_assert((!s::is_same_v<T, const U>), "");
    static_assert((s::is_same_v<const T, const U>), "");
#endif
}

template <class T, class U>
void
test_is_same_ref()
{
    static_assert((s::is_same<T, U>::value), "");
    static_assert((s::is_same<const T, U>::value), "");
    static_assert((s::is_same<T, const U>::value), "");
    static_assert((s::is_same<const T, const U>::value), "");
#if TEST_STD_VER > 14
    static_assert((s::is_same_v<T, U>), "");
    static_assert((s::is_same_v<const T, U>), "");
    static_assert((s::is_same_v<T, const U>), "");
    static_assert((s::is_same_v<const T, const U>), "");
#endif
}

template <class T, class U>
void
test_is_not_same()
{
    static_assert((!s::is_same<T, U>::value), "");
}

class Class
{
  public:
    ~Class();
};

cl::sycl::cl_bool
kernel_test()
{
    test_is_same<int, int>();
    test_is_same<void, void>();
    test_is_same<Class, Class>();
    test_is_same<int*, int*>();
    test_is_same_ref<int&, int&>();

    test_is_not_same<int, void>();
    test_is_not_same<void, Class>();
    test_is_not_same<Class, int*>();
    test_is_not_same<int*, int&>();
    test_is_not_same<int&, int>();

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
