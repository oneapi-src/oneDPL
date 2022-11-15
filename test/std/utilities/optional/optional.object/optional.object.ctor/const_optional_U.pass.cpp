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
//   optional(const optional<U>& rhs);

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

template <class KernelTest, class T, class U>
bool
kernel_test(const optional<U>& rhs)
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
                bool rhs_engaged = static_cast<bool>(rhs_access[0]);
                optional<T> lhs = rhs_access[0];
                ret_access[0] &= (static_cast<bool>(lhs) == rhs_engaged);
                if (rhs_engaged)
                    ret_access[0] &= (*lhs == *rhs_access[0]);
            });
        });
    }
    return ret;
}

class X
{
    int i_;

  public:
    X(int i) : i_(i) {}
    X(const X& x) : i_(x.i_) {}
    ~X() { i_ = 0; }
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

    friend constexpr bool
    operator==(const Y& x, const Y& y)
    {
        return x.i_ == y.i_;
    }
};

class KernelTest1;
class KernelTest2;
class KernelTest3;
class KernelTest4;
class KernelTest5;
class KernelTest6;

bool
test()
{
    bool ret = true;
    {
        typedef short U;
        typedef int T;
        optional<U> rhs;
        ret &= kernel_test<KernelTest1, T>(rhs);
    }
    {
        typedef short U;
        typedef int T;
        optional<U> rhs(U{3});
        ret &= kernel_test<KernelTest2, T>(rhs);
    }
    {
        typedef X T;
        typedef int U;
        optional<U> rhs;
        ret &= kernel_test<KernelTest3, T>(rhs);
    }
    {
        typedef X T;
        typedef int U;
        optional<U> rhs(U{3});
        ret &= kernel_test<KernelTest4, T>(rhs);
    }
    {
        typedef Y T;
        typedef int U;
        optional<U> rhs;
        ret &= kernel_test<KernelTest5, T>(rhs);
    }
    {
        typedef Y T;
        typedef int U;
        optional<U> rhs(U{3});
        ret &= kernel_test<KernelTest6, T>(rhs);
    }
    return ret;
}
#endif // TEST_DPCPP_BACKEND_PRESENT

int
main(int, char**)
{
#if TEST_DPCPP_BACKEND_PRESENT
    auto ret = test();
    TestUtils::exitOnError(ret);
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
