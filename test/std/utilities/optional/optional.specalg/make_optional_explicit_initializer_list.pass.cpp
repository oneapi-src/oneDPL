//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11, c++14
// <optional>

// template <class T, class U, class... Args>
//   constexpr optional<T> make_optional(initializer_list<U> il, Args&&... args);

#include "oneapi_std_test_config.h"
#include "test_macros.h"

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

struct TestT
{
    int x;
    int size;
    constexpr TestT(std::initializer_list<int> il) : x(*il.begin()), size(static_cast<int>(il.size())) {}
    constexpr TestT(std::initializer_list<int> il, const int*) : x(*il.begin()), size(static_cast<int>(il.size())) {}
};

void
kernel_test()
{
    cl::sycl::queue q;
    q.submit([&](cl::sycl::handler& cgh) {
        cgh.single_task<class KernelTest>([=]() {
            using s::make_optional;
            {
                constexpr auto opt = make_optional<TestT>({42, 2, 3});
                static_assert(opt->x == 42, "");
                static_assert(opt->size == 3, "");
            }
            {
                constexpr auto opt = make_optional<TestT>({42, 2, 3}, nullptr);
                static_assert(opt->x == 42, "");
                static_assert(opt->size == 3, "");
            }
        });
    });
}

int
main(int, char**)
{
    kernel_test();
    std::cout << "Pass" << std::endl;
    return 0;
}
