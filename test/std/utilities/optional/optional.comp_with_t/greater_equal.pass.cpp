//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11, c++14
// <optional>

// template <class T, class U> constexpr bool operator>=(const optional<T>& x, const U& v);
// template <class T, class U> constexpr bool operator>=(const U& v, const optional<T>& x);

#include "oneapi_std_test_config.h"
#include "test_macros.h"

#include <iostream>

#ifdef USE_ONEAPI_STD
#    include _ONEAPI_STD_TEST_HEADER(optional)
namespace s = oneapi_cpp_ns;
#else
#    include <optional>
namespace s = std;
#endif

using s::optional;

#if TEST_DPCPP_BACKEND_PRESENT
constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;

struct X
{
    int i_;

    constexpr X(int i) : i_(i) {}
};

constexpr bool
operator>=(const X& lhs, const X& rhs)
{
    return lhs.i_ >= rhs.i_;
}

bool
kernel_test()
{
    cl::sycl::queue q;
    bool ret = true;
    typedef X T;
    typedef optional<T> O;
    T val(2);
    O ia[3] = {O{}, O{1}, O{val}};
    cl::sycl::range<1> numOfItems1{1};
    cl::sycl::range<1> numOfItems2{3};
    {
        cl::sycl::buffer<bool, 1> buffer1(&ret, numOfItems1);
        cl::sycl::buffer<O, 1> buffer2(ia, numOfItems2);
        q.submit([&](cl::sycl::handler& cgh) {
            auto ret_access = buffer1.get_access<sycl_write>(cgh);
            auto ia_acc = buffer2.get_access<sycl_write>(cgh);
            cgh.single_task<class KernelTest>([=]() {
                {

                    ret_access[0] &= (!(ia_acc[0] >= T(1)));
                    ret_access[0] &= ((ia_acc[1] >= T(1))); // equal
                    ret_access[0] &= ((ia_acc[2] >= T(1)));
                    ret_access[0] &= (!(ia_acc[1] >= val));
                    ret_access[0] &= ((ia_acc[2] >= val)); // equal
                    ret_access[0] &= (!(ia_acc[2] >= T(3)));

                    ret_access[0] &= ((T(1) >= ia_acc[0]));
                    ret_access[0] &= ((T(1) >= ia_acc[1])); // equal
                    ret_access[0] &= (!(T(1) >= ia_acc[2]));
                    ret_access[0] &= ((val >= ia_acc[1]));
                    ret_access[0] &= ((val >= ia_acc[2])); // equal
                    ret_access[0] &= ((T(3) >= ia_acc[2]));
                }
                {
                    using O = optional<int>;
                    constexpr O o1(42);
                    static_assert(o1 >= 42l, "");
                    static_assert(!(11l >= o1), "");
                }
                {
                    using O = optional<const int>;
                    constexpr O o1(42);
                    static_assert(o1 >= 42, "");
                    static_assert(!(11 >= o1), "");
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
    auto ret = kernel_test();
    if (ret)
        std::cout << "Pass" << std::endl;
    else
        std::cout << "Fail" << std::endl;
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
