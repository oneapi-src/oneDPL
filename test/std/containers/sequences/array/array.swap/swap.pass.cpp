//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <array>

// void swap(array& a);
// namespace std { void swap(array<T, N> &x, array<T, N> &y);

#include "oneapi_std_test_config.h"

#include <iostream>

#ifdef USE_ONEAPI_STD
#    include _ONEAPI_STD_TEST_HEADER(array)
#    include _ONEAPI_STD_TEST_HEADER(utility)
namespace s = oneapi_cpp_ns;
#else
#    include <array>
#    include <utility>
namespace s = std;
#endif

#if TEST_DPCPP_BACKEND_PRESENT
class KernelTest1;

bool
kernel_test()
{

    bool ret = true;
    {
        cl::sycl::buffer<bool, 1> buf(&ret, cl::sycl::range<1>{1});
        cl::sycl::queue q;
        q.submit([&](cl::sycl::handler& cgh) {
            auto ret_acc = buf.get_access<cl::sycl::access::mode::write>(cgh);
            cgh.single_task<KernelTest1>([=]() {
                {
                    typedef int T;
                    typedef s::array<T, 3> C;
                    C c1 = {1, 2, 35};
                    C c2 = {4, 5, 65};
                    c1.swap(c2);
                    ret_acc[0] &= (c1.size() == 3);
                    ret_acc[0] &= (c1[0] == 4);
                    ret_acc[0] &= (c1[1] == 5);
                    ret_acc[0] &= (c1[2] == 65);
                    ret_acc[0] &= (c2.size() == 3);
                    ret_acc[0] &= (c2[0] == 1);
                    ret_acc[0] &= (c2[1] == 2);
                    ret_acc[0] &= (c2[2] == 35);
                }
                {
                    typedef int T;
                    typedef s::array<T, 3> C;
                    C c1 = {1, 2, 35};
                    C c2 = {4, 5, 65};
                    s::swap(c1, c2);
                    ret_acc[0] &= (c1.size() == 3);
                    ret_acc[0] &= (c1[0] == 4);
                    ret_acc[0] &= (c1[1] == 5);
                    ret_acc[0] &= (c1[2] == 65);
                    ret_acc[0] &= (c2.size() == 3);
                    ret_acc[0] &= (c2[0] == 1);
                    ret_acc[0] &= (c2[1] == 2);
                    ret_acc[0] &= (c2[2] == 35);
                }
                {
                    typedef int T;
                    typedef s::array<T, 0> C;
                    C c1 = {};
                    C c2 = {};
                    c1.swap(c2);
                    ret_acc[0] &= (c1.size() == 0);
                    ret_acc[0] &= (c2.size() == 0);
                }
                {
                    typedef int T;
                    typedef s::array<T, 0> C;
                    C c1 = {};
                    C c2 = {};
                    s::swap(c1, c2);
                    ret_acc[0] &= (c1.size() == 0);
                    ret_acc[0] &= (c2.size() == 0);
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
    TestUtils::exitOnError(ret);
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
