//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11, c++14
// <optional>

// constexpr bool optional<T>::has_value() const noexcept;

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

void
kernel_test()
{
    cl::sycl::queue q;
    {

        q.submit([&](cl::sycl::handler& cgh) {
            cgh.single_task<class KernelTest>([=]() {
                using s::optional;
                {
                    const optional<int> opt;
                    ((void)opt);
                    static_assert(noexcept(opt.has_value()));
                    static_assert(s::is_same<decltype(opt.has_value()), bool>::value, "");
                }
                {
                    constexpr optional<int> opt;
                    static_assert(!opt.has_value(), "");
                }
                {
                    constexpr optional<int> opt(0);
                    static_assert(opt.has_value(), "");
                }
            });
        });
    }
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
