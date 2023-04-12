//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// floating_point

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
test_floating_point_imp(sycl::queue& deviceQueue)
{
    deviceQueue.submit([&](sycl::handler& cgh) {
        cgh.single_task<T>([=]() {
            static_assert(!s::is_reference<T>::value, "");
            static_assert(s::is_arithmetic<T>::value, "");
            static_assert(s::is_fundamental<T>::value, "");
            static_assert(s::is_object<T>::value, "");
            static_assert(s::is_scalar<T>::value, "");
            static_assert(!s::is_compound<T>::value, "");
            static_assert(!s::is_member_pointer<T>::value, "");
        });
    });
}

template <class T>
void
test_floating_point(sycl::queue& deviceQueue)
{
    test_floating_point_imp<T>(deviceQueue);
    test_floating_point_imp<const T>(deviceQueue);
    test_floating_point_imp<volatile T>(deviceQueue);
    test_floating_point_imp<const volatile T>(deviceQueue);
}

void
kernel_test()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    test_floating_point<float>(deviceQueue);
    if (TestUtils::has_type_support<double>(deviceQueue.get_device()))
    {
        test_floating_point<double>(deviceQueue);
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
