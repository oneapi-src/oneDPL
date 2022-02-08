//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11, c++14
// <optional>

// template <class U, class... Args>
//     constexpr
//     explicit optional(in_place_t, initializer_list<U> il, Args&&... args);

#include "oneapi_std_test_config.h"
#include "test_macros.h"
#include <CL/sycl.hpp>
#include <iostream>

#ifdef USE_ONEAPI_STD
#    include _ONEAPI_STD_TEST_HEADER(optional)
#    include _ONEAPI_STD_TEST_HEADER(utility)
#    include _ONEAPI_STD_TEST_HEADER(type_traits)
#    include _ONEAPI_STD_TEST_HEADER(array)
namespace s = oneapi_cpp_ns;
#else
#    include <optional>
#    include <type_traits>
#    include <utility>
#    include <array>
namespace s = std;
#endif

constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;
using s::in_place;
using s::in_place_t;
using s::optional;

class X
{
    int i_;
    int j_ = 0;

  public:
    X() : i_(0) {}
    X(int i) : i_(i) {}
    X(int i, int j) : i_(i), j_(j) {}

    ~X() {}

    friend bool
    operator==(const X& x, const X& y)
    {
        return x.i_ == y.i_ && x.j_ == y.j_;
    }
};

class Y
{
    int i_;
    int j_ = 0;

  public:
    constexpr Y() : i_(0) {}
    constexpr Y(int i) : i_(i) {}
    constexpr Y(std::initializer_list<int> il) : i_(il.begin()[0]), j_(il.begin()[1]) {}

    friend constexpr bool
    operator==(const Y& x, const Y& y)
    {
        return x.i_ == y.i_ && x.j_ == y.j_;
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
                    static_assert(!s::is_constructible<X, std::initializer_list<int>&>::value, "");
                    static_assert(!s::is_constructible<optional<X>, std::initializer_list<int>&>::value, "");
                }
                {
                    static_assert(s::is_constructible<optional<Y>, std::initializer_list<int>&>::value, "");
                    constexpr optional<Y> opt(in_place, {3, 1});
                    static_assert(static_cast<bool>(opt) == true, "");
                    static_assert(*opt == Y{3, 1}, "");

                    struct test_constexpr_ctor : public optional<Y>
                    {
                        constexpr test_constexpr_ctor(in_place_t, std::initializer_list<int> i)
                            : optional<Y>(in_place, i)
                        {
                        }
                    };
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
