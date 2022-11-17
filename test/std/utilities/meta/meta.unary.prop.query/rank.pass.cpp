//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// rank

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

template <class T, unsigned A>
void
test_rank(cl::sycl::queue& deviceQueue)
{
    deviceQueue.submit([&](cl::sycl::handler& cgh) {
        cgh.single_task<T>([=]() {
            static_assert(s::rank<T>::value == A, "");
            static_assert(s::rank<const T>::value == A, "");
            static_assert(s::rank<volatile T>::value == A, "");
            static_assert(s::rank<const volatile T>::value == A, "");
#if TEST_STD_VER > 14
            static_assert(s::rank_v<T> == A, "");
            static_assert(s::rank_v<const T> == A, "");
            static_assert(s::rank_v<volatile T> == A, "");
            static_assert(s::rank_v<const volatile T> == A, "");
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
    test_rank<void, 0>(deviceQueue);
    test_rank<int&, 0>(deviceQueue);
    test_rank<Class, 0>(deviceQueue);
    test_rank<int*, 0>(deviceQueue);
    test_rank<const int*, 0>(deviceQueue);
    test_rank<int, 0>(deviceQueue);
    test_rank<bool, 0>(deviceQueue);
    test_rank<unsigned, 0>(deviceQueue);
    test_rank<char[3], 1>(deviceQueue);
    test_rank<char[][3], 2>(deviceQueue);
    test_rank<char[][4][3], 3>(deviceQueue);
    if (deviceQueue.get_device().has_extension("cl_khr_fp64"))
    {
        test_rank<double, 0>(deviceQueue);
    }
}

int
main()
{
    kernel_test();
    std::cout << "Pass" << std::endl;
    return 0;
}
