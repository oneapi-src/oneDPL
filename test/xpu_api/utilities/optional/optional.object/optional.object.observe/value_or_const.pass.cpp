//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11, c++14
// <optional>

// template <class U> constexpr T optional<T>::value_or(U&& v) const&;

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

struct Y
{
    int i_;

    constexpr Y(int i) : i_(i) {}
};

struct X
{
    int i_;

    constexpr X(int i) : i_(i) {}
    constexpr X(const Y& y) : i_(y.i_) {}
    constexpr X(Y&& y) : i_(y.i_ + 1) {}
    friend constexpr bool
    operator==(const X& x, const X& y)
    {
        return x.i_ == y.i_;
    }
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
                {
                    constexpr optional<X> opt(2);
                    constexpr Y y(3);
                    static_assert(opt.value_or(y) == 2, "");
                }
                {
                    constexpr optional<X> opt(2);
                    static_assert(opt.value_or(Y(3)) == 2, "");
                }
                {
                    constexpr optional<X> opt;
                    constexpr Y y(3);
                    static_assert(opt.value_or(y) == 3, "");
                }
                {
                    constexpr optional<X> opt;
                    static_assert(opt.value_or(Y(3)) == 4, "");
                }
                {
                    const optional<X> opt(2);
                    const Y y(3);
                    ret_access[0] &= (opt.value_or(y) == 2);
                }
                {
                    const optional<X> opt(2);
                    ret_access[0] &= (opt.value_or(Y(3)) == 2);
                }
                {
                    const optional<X> opt;
                    const Y y(3);
                    ret_access[0] &= (opt.value_or(y) == 3);
                }
                {
                    const optional<X> opt;
                    ret_access[0] &= (opt.value_or(Y(3)) == 4);
                }
            });
        });
    }
    return ret;
}

int
main(int, char**)
{
    auto ret = kernel_test();
    if (ret)
        std::cout << "Pass" << std::endl;
    else
        std::cout << "Fail" << std::endl;
    return 0;
}
