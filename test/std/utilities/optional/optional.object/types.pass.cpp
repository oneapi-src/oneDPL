//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11, c++14
// <optional>

// template <class T>
// class optional
// {
// public:
//     typedef T value_type;
//     ...

#include "oneapi_std_test_config.h"
#include "test_macros.h"
#include <CL/sycl.hpp>
#include <iostream>

#ifdef USE_ONEAPI_STD
#    include _ONEAPI_STD_TEST_HEADER(optional)
#    include _ONEAPI_STD_TEST_HEADER(type_traits)
namespace s = oneapi_cpp_ns;
#else
#    include <optional>
#    include <type_traits>
namespace s = std;
#endif

constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;
using s::optional;

template <class KernelTest, class Opt, class T>
void
test()
{
    cl::sycl::queue q;
    {

        q.submit([&](cl::sycl::handler& cgh) {
            cgh.single_task<KernelTest>([=]() { static_assert(s::is_same<typename Opt::value_type, T>::value, ""); });
        });
    }
}

class KernelTest1;
class KernelTest2;
class KernelTest3;
class KernelTest4;

int
main(int, char**)
{
    test<KernelTest1, optional<int>, int>();
    test<KernelTest2, optional<const int>, const int>();
    test<KernelTest3, optional<double>, double>();
    test<KernelTest4, optional<const double>, const double>();
    std::cout << "Pass" << std::endl;
    return 0;
}
