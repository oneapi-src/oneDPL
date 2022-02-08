//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11, c++14
// <optional>

// template <class... Args> T& optional<T>::emplace(Args&&... args);

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

class X
{
    int i_;
    int j_ = 0;

  public:
    X() : i_(0) {}
    X(int i) : i_(i) {}
    X(int i, int j) : i_(i), j_(j) {}

    friend bool
    operator==(const X& x, const X& y)
    {
        return x.i_ == y.i_ && x.j_ == y.j_;
    }
};

template <class T>
bool
test_one_arg()
{
    cl::sycl::queue q;
    bool ret = true;
    cl::sycl::range<1> numOfItems1{1};
    {
        cl::sycl::buffer<bool, 1> buffer1(&ret, numOfItems1);

        q.submit([&](cl::sycl::handler& cgh) {
            auto ret_access = buffer1.get_access<sycl_write>(cgh);
            cgh.single_task<T>([=]() {
                using Opt = s::optional<T>;
                {
                    Opt opt;
                    auto& v = opt.emplace();
                    static_assert(s::is_same_v<T&, decltype(v)>, "");
                    ret_access[0] &= (static_cast<bool>(opt) == true);
                    ret_access[0] &= (*opt == T(0));
                    ret_access[0] &= (&v == &*opt);
                }
                {
                    Opt opt;
                    auto& v = opt.emplace(1);
                    static_assert(s::is_same_v<T&, decltype(v)>, "");
                    ret_access[0] &= (static_cast<bool>(opt) == true);
                    ret_access[0] &= (*opt == T(1));
                    ret_access[0] &= (&v == &*opt);
                }
                {
                    Opt opt(2);
                    auto& v = opt.emplace();
                    static_assert(s::is_same_v<T&, decltype(v)>, "");
                    ret_access[0] &= (static_cast<bool>(opt) == true);
                    ret_access[0] &= (*opt == T(0));
                    ret_access[0] &= (&v == &*opt);
                }
                {
                    Opt opt(2);
                    auto& v = opt.emplace(1);
                    static_assert(s::is_same_v<T&, decltype(v)>, "");
                    ret_access[0] &= (static_cast<bool>(opt) == true);
                    ret_access[0] &= (*opt == T(1));
                    ret_access[0] &= (&v == &*opt);
                }
            });
        });
    }
    return ret;
}

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
            cgh.single_task<class KernelTest2>([=]() {
                optional<const int> opt;
                auto& v = opt.emplace(42);
                static_assert(s::is_same_v<const int&, decltype(v)>, "");
                ret_access[0] &= (*opt == 42);
                ret_access[0] &= (v == 42);
                opt.emplace();
                ret_access[0] &= (*opt == 0);
            });
        });
    }
    return ret;
}

int
main(int, char**)
{
    using T = int;
    auto ret = test_one_arg<T>();
    ret &= test_one_arg<const T>();
    ret &= kernel_test();
    if (ret)
        std::cout << "Pass" << std::endl;
    else
        std::cout << "Fail" << std::endl;
    return 0;
}
