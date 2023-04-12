//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// has_virtual_destructor

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

template <class T>
void
test_has_not_virtual_destructor(sycl::queue& deviceQueue)
{
    deviceQueue.submit([&](sycl::handler& cgh) {
        cgh.single_task<T>([=]() {
            static_assert(!s::has_virtual_destructor<T>::value, "");
            static_assert(!s::has_virtual_destructor<const T>::value, "");
            static_assert(!s::has_virtual_destructor<volatile T>::value, "");
            static_assert(!s::has_virtual_destructor<const volatile T>::value, "");
#if TEST_STD_VER > 14
            static_assert(!s::has_virtual_destructor_v<T>, "");
            static_assert(!s::has_virtual_destructor_v<const T>, "");
            static_assert(!s::has_virtual_destructor_v<volatile T>, "");
            static_assert(!s::has_virtual_destructor_v<const volatile T>, "");
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
    ~A();
};

void
kernel_test()
{
    sycl::queue deviceQueue;
    test_has_not_virtual_destructor<void>(deviceQueue);
    test_has_not_virtual_destructor<A>(deviceQueue);
    test_has_not_virtual_destructor<int&>(deviceQueue);
    test_has_not_virtual_destructor<Union>(deviceQueue);
    test_has_not_virtual_destructor<Empty>(deviceQueue);
    test_has_not_virtual_destructor<int>(deviceQueue);
    test_has_not_virtual_destructor<int*>(deviceQueue);
    test_has_not_virtual_destructor<const int*>(deviceQueue);
    test_has_not_virtual_destructor<char[3]>(deviceQueue);
    test_has_not_virtual_destructor<char[]>(deviceQueue);
    test_has_not_virtual_destructor<bit_zero>(deviceQueue);
    if (TestUtils::has_type_support<double>(deviceQueue.get_device()))
    {
        test_has_not_virtual_destructor<double>(deviceQueue);
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
