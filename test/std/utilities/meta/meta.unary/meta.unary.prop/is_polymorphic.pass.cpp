//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// is_polymorphic

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
test_is_not_polymorphic(sycl::queue& deviceQueue)
{
    deviceQueue.submit([&](sycl::handler& cgh) {
        cgh.single_task<T>([=]() {
            static_assert(!s::is_polymorphic<T>::value, "");
            static_assert(!s::is_polymorphic<const T>::value, "");
            static_assert(!s::is_polymorphic<volatile T>::value, "");
            static_assert(!s::is_polymorphic<const volatile T>::value, "");
#if TEST_STD_VER > 14
            static_assert(!s::is_polymorphic_v<T>, "");
            static_assert(!s::is_polymorphic_v<const T>, "");
            static_assert(!s::is_polymorphic_v<volatile T>, "");
            static_assert(!s::is_polymorphic_v<const volatile T>, "");
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

class Final final
{
};

void
kernel_test()
{
    sycl::queue deviceQueue;
    test_is_not_polymorphic<void>(deviceQueue);
    test_is_not_polymorphic<int&>(deviceQueue);
    test_is_not_polymorphic<int>(deviceQueue);
    test_is_not_polymorphic<int*>(deviceQueue);
    test_is_not_polymorphic<const int*>(deviceQueue);
    test_is_not_polymorphic<char[3]>(deviceQueue);
    test_is_not_polymorphic<char[]>(deviceQueue);
    test_is_not_polymorphic<Union>(deviceQueue);
    test_is_not_polymorphic<Empty>(deviceQueue);
    test_is_not_polymorphic<bit_zero>(deviceQueue);
    test_is_not_polymorphic<Final>(deviceQueue);

    if (TestUtils::has_type_support<double>(deviceQueue.get_device()))
    {
        test_is_not_polymorphic<double>(deviceQueue);
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
