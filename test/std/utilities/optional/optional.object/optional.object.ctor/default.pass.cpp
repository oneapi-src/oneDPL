//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11, c++14
// <optional>

// constexpr optional() noexcept;

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
using s::optional;

template <class Opt>
void
test_constexpr()
{
    cl::sycl::queue q;
    {
        q.submit([&](cl::sycl::handler& cgh) {
            cgh.single_task<Opt>([=]() {
                static_assert(s::is_nothrow_default_constructible<Opt>::value, "");
                static_assert(s::is_trivially_destructible<Opt>::value, "");
                static_assert(s::is_trivially_destructible<typename Opt::value_type>::value, "");

                constexpr Opt opt;
                static_assert(static_cast<bool>(opt) == false, "");
            });
        });
    }
}
#endif // TEST_DPCPP_BACKEND_PRESENT

int
main(int, char**)
{
#if TEST_DPCPP_BACKEND_PRESENT
    test_constexpr<optional<int>>();
    test_constexpr<optional<int*>>();
    std::cout << "Pass" << std::endl;
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
