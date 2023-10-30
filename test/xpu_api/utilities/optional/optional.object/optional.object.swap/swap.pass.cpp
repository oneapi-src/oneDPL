//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11, c++14
// <optional>

// void swap(optional&)
//     noexcept(is_nothrow_move_constructible<T>::value &&
//              is_nothrow_swappable<T>::value)

#include "oneapi_std_test_config.h"
#include "test_macros.h"
#include <CL/sycl.hpp>
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

class X
{
    int i_;

  public:
    X(int i) : i_(i) {}
    X(X&& x) = default;
    X&
    operator=(X&&) = default;
    ~X() {}

    friend bool
    operator==(const X& x, const X& y)
    {
        return x.i_ == y.i_;
    }
};

class Y
{
    int i_;

  public:
    Y(int i) : i_(i) {}
    Y(Y&&) = default;
    ~Y() {}

    friend constexpr bool
    operator==(const Y& x, const Y& y)
    {
        return x.i_ == y.i_;
    }
    friend void
    swap(Y& x, Y& y)
    {
        s::swap(x.i_, y.i_);
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
                    optional<int> opt1;
                    optional<int> opt2;
                    static_assert(noexcept(opt1.swap(opt2)) == true, "");
                    ret_access[0] &= (static_cast<bool>(opt1) == false);
                    ret_access[0] &= (static_cast<bool>(opt2) == false);
                    opt1.swap(opt2);
                    ret_access[0] &= (static_cast<bool>(opt1) == false);
                    ret_access[0] &= (static_cast<bool>(opt2) == false);
                }
                {
                    optional<int> opt1(1);
                    optional<int> opt2;
                    static_assert(noexcept(opt1.swap(opt2)) == true, "");
                    ret_access[0] &= (static_cast<bool>(opt1) == true);
                    ret_access[0] &= (*opt1 == 1);
                    ret_access[0] &= (static_cast<bool>(opt2) == false);
                    opt1.swap(opt2);
                    ret_access[0] &= (static_cast<bool>(opt1) == false);
                    ret_access[0] &= (static_cast<bool>(opt2) == true);
                    ret_access[0] &= (*opt2 == 1);
                }
                {
                    optional<int> opt1;
                    optional<int> opt2(2);
                    static_assert(noexcept(opt1.swap(opt2)) == true, "");
                    ret_access[0] &= (static_cast<bool>(opt1) == false);
                    ret_access[0] &= (static_cast<bool>(opt2) == true);
                    ret_access[0] &= (*opt2 == 2);
                    opt1.swap(opt2);
                    ret_access[0] &= (static_cast<bool>(opt1) == true);
                    ret_access[0] &= (*opt1 == 2);
                    ret_access[0] &= (static_cast<bool>(opt2) == false);
                }
                {
                    optional<int> opt1(1);
                    optional<int> opt2(2);
                    static_assert(noexcept(opt1.swap(opt2)) == true, "");
                    ret_access[0] &= (static_cast<bool>(opt1) == true);
                    ret_access[0] &= (*opt1 == 1);
                    ret_access[0] &= (static_cast<bool>(opt2) == true);
                    ret_access[0] &= (*opt2 == 2);
                    opt1.swap(opt2);
                    ret_access[0] &= (static_cast<bool>(opt1) == true);
                    ret_access[0] &= (*opt1 == 2);
                    ret_access[0] &= (static_cast<bool>(opt2) == true);
                    ret_access[0] &= (*opt2 == 1);
                }
                {
                    optional<X> opt1;
                    optional<X> opt2;
                    static_assert(noexcept(opt1.swap(opt2)) == true, "");
                    ret_access[0] &= (static_cast<bool>(opt1) == false);
                    ret_access[0] &= (static_cast<bool>(opt2) == false);
                    opt1.swap(opt2);
                    ret_access[0] &= (static_cast<bool>(opt1) == false);
                    ret_access[0] &= (static_cast<bool>(opt2) == false);
                }
                {
                    optional<X> opt1(1);
                    optional<X> opt2;
                    static_assert(noexcept(opt1.swap(opt2)) == true, "");
                    ret_access[0] &= (static_cast<bool>(opt1) == true);
                    ret_access[0] &= (*opt1 == 1);
                    ret_access[0] &= (static_cast<bool>(opt2) == false);
                    opt1.swap(opt2);
                    ret_access[0] &= (static_cast<bool>(opt1) == false);
                    ret_access[0] &= (static_cast<bool>(opt2) == true);
                    ret_access[0] &= (*opt2 == 1);
                }
                {
                    optional<X> opt1;
                    optional<X> opt2(2);
                    static_assert(noexcept(opt1.swap(opt2)) == true, "");
                    ret_access[0] &= (static_cast<bool>(opt1) == false);
                    ret_access[0] &= (static_cast<bool>(opt2) == true);
                    ret_access[0] &= (*opt2 == 2);
                    opt1.swap(opt2);
                    ret_access[0] &= (static_cast<bool>(opt1) == true);
                    ret_access[0] &= (*opt1 == 2);
                    ret_access[0] &= (static_cast<bool>(opt2) == false);
                }
                {
                    optional<X> opt1(1);
                    optional<X> opt2(2);
                    static_assert(noexcept(opt1.swap(opt2)) == true, "");
                    ret_access[0] &= (static_cast<bool>(opt1) == true);
                    ret_access[0] &= (*opt1 == 1);
                    ret_access[0] &= (static_cast<bool>(opt2) == true);
                    ret_access[0] &= (*opt2 == 2);
                    opt1.swap(opt2);
                    ret_access[0] &= (static_cast<bool>(opt1) == true);
                    ret_access[0] &= (*opt1 == 2);
                    ret_access[0] &= (static_cast<bool>(opt2) == true);
                    ret_access[0] &= (*opt2 == 1);
                }
                {
                    optional<Y> opt1;
                    optional<Y> opt2;
                    static_assert(noexcept(opt1.swap(opt2)) == false, "");
                    ret_access[0] &= (static_cast<bool>(opt1) == false);
                    ret_access[0] &= (static_cast<bool>(opt2) == false);
                    opt1.swap(opt2);
                    ret_access[0] &= (static_cast<bool>(opt1) == false);
                    ret_access[0] &= (static_cast<bool>(opt2) == false);
                }
                {
                    optional<Y> opt1(1);
                    optional<Y> opt2;
                    static_assert(noexcept(opt1.swap(opt2)) == false, "");
                    ret_access[0] &= (static_cast<bool>(opt1) == true);
                    ret_access[0] &= (*opt1 == 1);
                    ret_access[0] &= (static_cast<bool>(opt2) == false);
                    opt1.swap(opt2);
                    ret_access[0] &= (static_cast<bool>(opt1) == false);
                    ret_access[0] &= (static_cast<bool>(opt2) == true);
                    ret_access[0] &= (*opt2 == 1);
                }
                {
                    optional<Y> opt1;
                    optional<Y> opt2(2);
                    static_assert(noexcept(opt1.swap(opt2)) == false, "");
                    ret_access[0] &= (static_cast<bool>(opt1) == false);
                    ret_access[0] &= (static_cast<bool>(opt2) == true);
                    ret_access[0] &= (*opt2 == 2);
                    opt1.swap(opt2);
                    ret_access[0] &= (static_cast<bool>(opt1) == true);
                    ret_access[0] &= (*opt1 == 2);
                    ret_access[0] &= (static_cast<bool>(opt2) == false);
                }
                {
                    optional<Y> opt1(1);
                    optional<Y> opt2(2);
                    static_assert(noexcept(opt1.swap(opt2)) == false, "");
                    ret_access[0] &= (static_cast<bool>(opt1) == true);
                    ret_access[0] &= (*opt1 == 1);
                    ret_access[0] &= (static_cast<bool>(opt2) == true);
                    ret_access[0] &= (*opt2 == 2);
                    opt1.swap(opt2);
                    ret_access[0] &= (static_cast<bool>(opt1) == true);
                    ret_access[0] &= (*opt1 == 2);
                    ret_access[0] &= (static_cast<bool>(opt2) == true);
                    ret_access[0] &= (*opt2 == 1);
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
