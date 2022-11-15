//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11, c++14
// <optional>

// template <class U>
//   explicit optional(optional<U>&& rhs);

#include "oneapi_std_test_config.h"
#include "test_macros.h"

#include <iostream>

#ifdef USE_ONEAPI_STD
#    include _ONEAPI_STD_TEST_HEADER(optional)
#    include _ONEAPI_STD_TEST_HEADER(utility)
#    include _ONEAPI_STD_TEST_HEADER(type_traits)
namespace s = oneapi_cpp_ns;
#else
#    include <optional>
#    include <utility>
#    include <type_traits>
namespace s = std;
#endif

constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;
using s::optional;

template <class KernelTest, class T, class U>
bool
test(optional<U>&& rhs, bool is_going_to_throw = false)
{
    cl::sycl::queue q;
    bool ret = true;
    cl::sycl::range<1> numOfItems1{1};
    {
        cl::sycl::buffer<bool, 1> buffer1(&ret, numOfItems1);
        cl::sycl::buffer<optional<U>, 1> buffer2(&rhs, numOfItems1);

        q.submit([&](cl::sycl::handler& cgh) {
            auto ret_access = buffer1.get_access<sycl_write>(cgh);
            auto rhs_access = buffer2.template get_access<sycl_write>(cgh);
            cgh.single_task<KernelTest>([=]() {
                static_assert(!(s::is_convertible<optional<U>&&, optional<T>>::value), "");
                bool rhs_engaged = static_cast<bool>(rhs_access[0]);
                optional<T> lhs(s::move(rhs_access[0]));
                ret_access[0] &= (static_cast<bool>(lhs) == rhs_engaged);
            });
        });
    }
    return ret;
}

class X
{
    int i_;

  public:
    explicit X(int i) : i_(i) {}
    X(X&& x) : i_(s::exchange(x.i_, 0)) {}
    ~X() { i_ = 0; }
    friend bool
    operator==(const X& x, const X& y)
    {
        return x.i_ == y.i_;
    }
};

class KernelTest1;
class KernelTest2;

int
main(int, char**)
{
    bool ret = true;
    {
        optional<int> rhs;
        ret &= test<KernelTest1, X>(s::move(rhs));
    }
    {
        optional<int> rhs(3);
        ret &= test<KernelTest2, X>(s::move(rhs));
    }
    if (ret)
        std::cout << "Pass" << std::endl;
    else
        std::cout << "Fail" << std::endl;
    return 0;
}
