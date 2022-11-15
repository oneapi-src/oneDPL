//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <optional>
// UNSUPPORTED: c++98, c++03, c++11, c++14

// template<class T>
//   optional(T) -> optional<T>;

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

#if TEST_DPCPP_BACKEND_PRESENT
constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;

struct A
{
};

bool
kernel_test()
{
    cl::sycl::queue q;
    bool ret = true;
    cl::sycl::range<1> numOfItems1{1};
    {
        cl::sycl::buffer<bool, 1> buffer1(&ret, numOfItems1);

        q.submit([&](cl::sycl::handler& cgh) {
            auto ret_access = buffer1.get_access<sycl_write>(cgh);
            cgh.single_task<class KernelTest>([=]() {
                //  Test the explicit deduction guides
                {
                    //  optional(T)
                    s::optional opt(5);
                    static_assert(s::is_same_v<decltype(opt), s::optional<int>>, "");
                    ret_access[0] &= (static_cast<bool>(opt));
                    ret_access[0] &= (*opt == 5);
                }

                {
                    //  optional(T)
                    s::optional opt(A{});
                    static_assert(s::is_same_v<decltype(opt), s::optional<A>>, "");
                    ret_access[0] &= (static_cast<bool>(opt));
                }

                //  Test the implicit deduction guides
                {
                    //  optional(optional);
                    s::optional<char> source('A');
                    s::optional opt(source);
                    static_assert(s::is_same_v<decltype(opt), s::optional<char>>, "");
                    ret_access[0] &= (static_cast<bool>(opt) == static_cast<bool>(source));
                    ret_access[0] &= (*opt == *source);
                }
            });
        });
    }
    return ret;
}
#endif // TEST_DPCPP_BACKEND_PRESENT

int
main(int, char**)
{
#if TEST_DPCPP_BACKEND_PRESENT
    auto ret = kernel_test();
    if (ret)
        std::cout << "Pass" << std::endl;
    else
        std::cout << "Fail" << std::endl;
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
