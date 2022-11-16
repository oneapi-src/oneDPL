//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11, c++14
// XFAIL: dylib-has-no-bad_optional_access && !libcpp-no-exceptions

// <optional>
//
// template <class T>
//   constexpr optional<decay_t<T>> make_optional(T&& v);

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

bool
kernel_test()
{
    bool ret = true;
    {
        cl::sycl::queue q;
        cl::sycl::buffer<bool, 1> buf(&ret, cl::sycl::range<1>{1});
        q.submit([&](cl::sycl::handler& cgh) {
            auto ret_access = buf.get_access<cl::sycl::access::mode::write>(cgh);
            cgh.single_task<class KernelTest>([=]() {
                using s::optional;
                using s::make_optional;
                {
                    optional<int> opt = make_optional(2);
                    ret_access[0] &= (*opt == 2);
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
