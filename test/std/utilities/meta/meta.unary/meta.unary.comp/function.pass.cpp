//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// function

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
test_function_imp(cl::sycl::queue& deviceQueue)
{
    deviceQueue.submit([&](cl::sycl::handler& cgh) {
        cgh.single_task<T>([=]() {
            static_assert(!s::is_reference<T>::value, "");
            static_assert(!s::is_arithmetic<T>::value, "");
            static_assert(!s::is_fundamental<T>::value, "");
            static_assert(!s::is_object<T>::value, "");
            static_assert(!s::is_scalar<T>::value, "");
            static_assert(s::is_compound<T>::value, "");
            static_assert(!s::is_member_pointer<T>::value, "");
        });
    });
}

template <class T>
void
test_function(cl::sycl::queue& deviceQueue)
{
    test_function_imp<T>(deviceQueue);
    test_function_imp<const T>(deviceQueue);
    test_function_imp<volatile T>(deviceQueue);
    test_function_imp<const volatile T>(deviceQueue);
}

void
kernel_test()
{
    cl::sycl::queue deviceQueue;
    if (deviceQueue.get_device().has_extension("cl_khr_fp64"))
    {
        test_function<int(double)>(deviceQueue);
        test_function<int(double, char)>(deviceQueue);
    }
    test_function<void()>(deviceQueue);
    test_function<void(int)>(deviceQueue);
}
#endif // TEST_DPCPP_BACKEND_PRESENT

int
main(int, char**)
{
#if TEST_DPCPP_BACKEND_PRESENT
    kernel_test();
    TestUtils::exitOnError(true);
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
