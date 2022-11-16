//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// alignment_of

#include "oneapi_std_test_config.h"
#include "test_macros.h"

#include <iostream>
#include <cstdint>
#ifdef USE_ONEAPI_STD
#    include _ONEAPI_STD_TEST_HEADER(type_traits)
namespace s = oneapi_cpp_ns;
#else
#    include <type_traits>
#    include <cstdint>
namespace s = std;
#endif

#if TEST_DPCPP_BACKEND_PRESENT
constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;

template <class T, unsigned A>
void
test_alignment_of(cl::sycl::queue& deviceQueue)
{
    deviceQueue.submit([&](cl::sycl::handler& cgh) {
        cgh.single_task<T>([=]() {
            const unsigned AlignofResult = TEST_ALIGNOF(T);
            static_assert(AlignofResult == A, "Golden value does not match result of alignof keyword");
            static_assert(s::alignment_of<T>::value == AlignofResult, "");
            static_assert(s::alignment_of<T>::value == A, "");
            static_assert(s::alignment_of<const T>::value == A, "");
            static_assert(s::alignment_of<volatile T>::value == A, "");
            static_assert(s::alignment_of<const volatile T>::value == A, "");
#if TEST_STD_VER > 14
            static_assert(s::alignment_of_v<T> == A, "");
            static_assert(s::alignment_of_v<const T> == A, "");
            static_assert(s::alignment_of_v<volatile T> == A, "");
            static_assert(s::alignment_of_v<const volatile T> == A, "");
#endif
        });
    });
}

class Class
{
  public:
    ~Class();
};

void
kernel_test()
{
    cl::sycl::queue deviceQueue;
    test_alignment_of<int&, 4>(deviceQueue);
    test_alignment_of<Class, 1>(deviceQueue);
    test_alignment_of<int*, sizeof(intptr_t)>(deviceQueue);
    test_alignment_of<const int*, sizeof(intptr_t)>(deviceQueue);
    test_alignment_of<char[3], 1>(deviceQueue);
    test_alignment_of<int, 4>(deviceQueue);
    test_alignment_of<unsigned, 4>(deviceQueue);
    if (deviceQueue.get_device().has_extension("cl_khr_fp64"))
    {
        test_alignment_of<double, TEST_ALIGNOF(double)>(deviceQueue);
    }
}
#endif // TEST_DPCPP_BACKEND_PRESENT

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    kernel_test();
    std::cout << "Pass" << std::endl;
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
