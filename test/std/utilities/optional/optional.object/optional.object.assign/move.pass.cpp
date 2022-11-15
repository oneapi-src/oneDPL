//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11, c++14
// <optional>

// optional<T>& operator=(optional<T>&& rhs)
//     noexcept(is_nothrow_move_assignable<T>::value &&
//              is_nothrow_move_constructible<T>::value); // constexpr in C++20

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

#if TEST_DPCPP_BACKEND_PRESENT
constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;
using s::optional;

template <class Tp>
bool
assign_empty(optional<Tp>&& lhs)
{
    cl::sycl::queue q;
    bool ret = true;
    cl::sycl::range<1> numOfItems1{1};
    {
        cl::sycl::buffer<bool, 1> buffer1(&ret, numOfItems1);
        cl::sycl::buffer<optional<Tp>, 1> buffer2(&lhs, numOfItems1);

        q.submit([&](cl::sycl::handler& cgh) {
            auto ret_access = buffer1.get_access<sycl_write>(cgh);
            auto lhs_access = buffer2.template get_access<sycl_write>(cgh);
            cgh.single_task<class KernelTest1>([=]() {
                optional<Tp> rhs;
                lhs_access[0] = s::move(rhs);
                ret_access[0] &= !lhs_access[0].has_value() && !rhs.has_value();
            });
        });
    }
    return ret;
}

template <class Tp>
bool
assign_value(optional<Tp>&& lhs)
{
    cl::sycl::queue q;
    bool ret = true;
    cl::sycl::range<1> numOfItems1{1};
    {
        cl::sycl::buffer<bool, 1> buffer1(&ret, numOfItems1);
        cl::sycl::buffer<optional<Tp>, 1> buffer2(&lhs, numOfItems1);

        q.submit([&](cl::sycl::handler& cgh) {
            auto ret_access = buffer1.get_access<sycl_write>(cgh);
            auto lhs_access = buffer2.template get_access<sycl_write>(cgh);
            cgh.single_task<class KernelTest2>([=]() {
                optional<Tp> rhs(101);
                lhs_access[0] = s::move(rhs);
                ret_access[0] &= lhs_access[0].has_value() && rhs.has_value() && *lhs_access[0] == Tp{101};
                ;
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
            cgh.single_task<class KernelTest3>([=]() {
                {
                    static_assert(s::is_nothrow_move_assignable<optional<int>>::value, "");
                    optional<int> opt;
                    constexpr optional<int> opt2;
                    opt = s::move(opt2);
                    static_assert(static_cast<bool>(opt2) == false, "");
                    ret_access[0] &= (static_cast<bool>(opt) == static_cast<bool>(opt2));
                }
                {
                    optional<int> opt;
                    constexpr optional<int> opt2(2);
                    opt = s::move(opt2);
                    static_assert(static_cast<bool>(opt2) == true, "");
                    static_assert(*opt2 == 2, "");
                    ret_access[0] &= (static_cast<bool>(opt) == static_cast<bool>(opt2));
                    ret_access[0] &= (*opt == *opt2);
                }
                {
                    optional<int> opt(3);
                    constexpr optional<int> opt2;
                    opt = s::move(opt2);
                    static_assert(static_cast<bool>(opt2) == false, "");
                    ret_access[0] &= (static_cast<bool>(opt) == static_cast<bool>(opt2));
                }
                {
                    optional<int> opt(3);
                    constexpr optional<int> opt2(2);
                    opt = s::move(opt2);
                    static_assert(static_cast<bool>(opt2) == true, "");
                    static_assert(*opt2 == 2, "");
                    ret_access[0] &= (static_cast<bool>(opt) == static_cast<bool>(opt2));
                    ret_access[0] &= (*opt == *opt2);
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
    using O = optional<int>;
    auto ret = assign_empty(O{42});
    ret &= assign_value(O{42});
    ret &= kernel_test();
    TestUtils::exitOnError(ret);
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
