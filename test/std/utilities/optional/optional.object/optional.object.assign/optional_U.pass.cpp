//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11, c++14
// <optional>

// From LWG2451:
// template <class U>
// optional<T>& operator=(optional<U>&& rhs);

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

struct Y1
{
    Y1() = default;
    Y1(const int&) {}
    Y1&
    operator=(const Y1&) = delete;
};

struct Y2
{
    Y2() = default;
    Y2(const int&) = delete;
    Y2&
    operator=(const int&)
    {
        return *this;
    }
};

class B
{
};
class D : public B
{
};

template <class T>
struct AssignableFrom
{
    int type_constructed = 0;
    int type_assigned = 0;
    int int_constructed = 0;
    int int_assigned = 0;

    AssignableFrom() = default;

    explicit AssignableFrom(T) { ++type_constructed; }
    AssignableFrom& operator=(T)
    {
        ++type_assigned;
        return *this;
    }

    AssignableFrom(int) { ++int_constructed; }
    AssignableFrom&
    operator=(int)
    {
        ++int_assigned;
        return *this;
    }

  private:
    AssignableFrom(AssignableFrom const&) = delete;
    AssignableFrom&
    operator=(AssignableFrom const&) = delete;
};

bool
test_ambigious_assign()
{
    cl::sycl::queue q;
    bool ret = true;
    cl::sycl::range<1> numOfItems1{1};
    {
        cl::sycl::buffer<bool, 1> buffer1(&ret, numOfItems1);

        q.submit([&](cl::sycl::handler& cgh) {
            auto ret_access = buffer1.get_access<sycl_write>(cgh);
            cgh.single_task<class KernelTest1>([=]() {
                using OptInt = s::optional<int>;
                {
                    using T = AssignableFrom<OptInt&&>;
                    {
                        OptInt a(42);
                        s::optional<T> t;
                        t = s::move(a);
                        ret_access[0] &= (t->type_constructed == 1);
                        ret_access[0] &= (t->type_assigned == 0);
                        ret_access[0] &= (t->int_constructed == 0);
                        ret_access[0] &= (t->int_assigned == 0);
                    }
                    {
                        using Opt = s::optional<T>;
                        static_assert(!s::is_assignable<Opt&, const OptInt&&>::value, "");
                        static_assert(!s::is_assignable<Opt&, const OptInt&>::value, "");
                        static_assert(!s::is_assignable<Opt&, OptInt&>::value, "");
                    }
                }
                {
                    using T = AssignableFrom<OptInt const&&>;
                    {
                        const OptInt a(42);
                        s::optional<T> t;
                        t = s::move(a);
                        ret_access[0] &= (t->type_constructed == 1);
                        ret_access[0] &= (t->type_assigned == 0);
                        ret_access[0] &= (t->int_constructed == 0);
                        ret_access[0] &= (t->int_assigned == 0);
                    }
                    {
                        OptInt a(42);
                        s::optional<T> t;
                        t = s::move(a);
                        ret_access[0] &= (t->type_constructed == 1);
                        ret_access[0] &= (t->type_assigned == 0);
                        ret_access[0] &= (t->int_constructed == 0);
                        ret_access[0] &= (t->int_assigned == 0);
                    }
                    {
                        using Opt = s::optional<T>;
                        static_assert(s::is_assignable<Opt&, OptInt&&>::value, "");
                        static_assert(!s::is_assignable<Opt&, const OptInt&>::value, "");
                        static_assert(!s::is_assignable<Opt&, OptInt&>::value, "");
                    }
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
                {
                    optional<int> opt;
                    optional<short> opt2;
                    opt = s::move(opt2);
                    ret_access[0] &= (static_cast<bool>(opt2) == false);
                    ret_access[0] &= (static_cast<bool>(opt) == static_cast<bool>(opt2));
                }
                {
                    optional<int> opt;
                    optional<short> opt2(short{2});
                    opt = s::move(opt2);
                    ret_access[0] &= (static_cast<bool>(opt2) == true);
                    ret_access[0] &= (*opt2 == 2);
                    ret_access[0] &= (static_cast<bool>(opt) == static_cast<bool>(opt2));
                    ret_access[0] &= (*opt == *opt2);
                }
                {
                    optional<int> opt(3);
                    optional<short> opt2;
                    opt = s::move(opt2);
                    ret_access[0] &= (static_cast<bool>(opt2) == false);
                    ret_access[0] &= (static_cast<bool>(opt) == static_cast<bool>(opt2));
                }
                {
                    optional<int> opt(3);
                    optional<short> opt2(short{2});
                    opt = s::move(opt2);
                    ret_access[0] &= (static_cast<bool>(opt2) == true);
                    ret_access[0] &= (*opt2 == 2);
                    ret_access[0] &= (static_cast<bool>(opt) == static_cast<bool>(opt2));
                    ret_access[0] &= (*opt == *opt2);
                }
            });
        });
    }
    return ret;
}

int
main(int, char**)
{
    auto ret = test_ambigious_assign();
    ret &= kernel_test();
    if (ret)
        std::cout << "Pass" << std::endl;
    else
        std::cout << "Fail" << std::endl;
    return 0;
}
