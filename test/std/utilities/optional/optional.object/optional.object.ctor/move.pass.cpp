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

// constexpr optional(optional<T>&& rhs);

#include "oneapi_std_test_config.h"
#include "test_macros.h"

#include <iostream>

#ifdef USE_ONEAPI_STD
#    include _ONEAPI_STD_TEST_HEADER(optional)
#    include _ONEAPI_STD_TEST_HEADER(type_traits)
#    include _ONEAPI_STD_TEST_HEADER(utility)
namespace s = oneapi_cpp_ns;
#else
#    include <optional>
#    include <type_traits>
#    include <utility>
namespace s = std;
#endif

constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;
using s::optional;

class KernelTest1;
class KernelTest2;

template <class KernelTest, class T, class... InitArgs>
bool
kernel_test1(InitArgs&&... args)
{
    cl::sycl::queue q;
    bool ret = true;
    const optional<T> orig(s::forward<InitArgs>(args)...);
    cl::sycl::range<1> numOfItems1{1};
    {
        cl::sycl::buffer<bool, 1> buffer1(&ret, numOfItems1);
        cl::sycl::buffer<const optional<T>, 1> buffer2(&orig, numOfItems1);

        q.submit([&](cl::sycl::handler& cgh) {
            auto ret_access = buffer1.get_access<sycl_write>(cgh);
            auto orig_access = buffer2.template get_access<sycl_write>(cgh);
            cgh.single_task<KernelTest>([=]() {
                optional<T> rhs(orig_access[0]);
                bool rhs_engaged = static_cast<bool>(rhs);
                optional<T> lhs = s::move(rhs);
                if (rhs_engaged)
                    ret_access[0] &= (*lhs == *orig_access[0]);
            });
        });
    }
    return ret;
}

bool
kernel_test2()
{
    cl::sycl::queue q;
    bool ret = true;
    cl::sycl::range<1> numOfItems1{1};
    {
        cl::sycl::buffer<bool, 1> buffer1(&ret, numOfItems1);

        q.submit([&](cl::sycl::handler& cgh) {
            auto ret_access = buffer1.get_access<sycl_write>(cgh);
            cgh.single_task<class KernelTest>([=]() {
                {
                    optional<const int> o(42);
                    optional<const int> o2(s::move(o));
                    ret_access[0] &= (*o2 == 42);
                }
                {
                    constexpr s::optional<int> o1{4};
                    constexpr s::optional<int> o2 = s::move(o1);
                    static_assert(*o2 == 4, "");
                }
            });
        });
    }
    return ret;
}

int
main(int, char**)
{
    auto ret = kernel_test1<KernelTest1, int>();
    ret &= kernel_test1<KernelTest2, int>(3);
    ret &= kernel_test2();
    if (ret)
        std::cout << "Pass" << std::endl;
    else
        std::cout << "Fail" << std::endl;
    return 0;
}
