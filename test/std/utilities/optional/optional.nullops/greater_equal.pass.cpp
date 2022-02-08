//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11, c++14
// <optional>

// template <class T> constexpr bool operator>=(const optional<T>& x, nullopt_t) noexcept;
// template <class T> constexpr bool operator>=(nullopt_t, const optional<T>& x) noexcept;

#include "oneapi_std_test_config.h"
#include "test_macros.h"
#include <CL/sycl.hpp>
#include <iostream>

#ifdef USE_ONEAPI_STD
#    include _ONEAPI_STD_TEST_HEADER(optional)
namespace s = oneapi_cpp_ns;
#else
#    include <optional>
namespace s = std;
#endif

constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;

bool
kernel_test()
{
    cl::sycl::queue q;
    bool ret = true;
    typedef int T;
    typedef s::optional<T> O;
    O ia[2] = {O{}, O{1}};
    cl::sycl::range<1> numOfItems1{1};
    cl::sycl::range<1> numOfItems2{2};
    {
        cl::sycl::buffer<bool, 1> buffer1(&ret, numOfItems1);
        cl::sycl::buffer<O, 1> buffer2(ia, numOfItems2);

        q.submit([&](cl::sycl::handler& cgh) {
            auto ret_access = buffer1.get_access<sycl_write>(cgh);
            auto ia_acc = buffer2.get_access<sycl_write>(cgh);
            cgh.single_task<class KernelTest>([=]() {
                using s::optional;
                using s::nullopt_t;
                using s::nullopt;

                ret_access[0] &= ((nullopt >= ia_acc[0]));
                ret_access[0] &= (!(nullopt >= ia_acc[1]));
                ret_access[0] &= ((ia_acc[0] >= nullopt));
                ret_access[0] &= ((ia_acc[1] >= nullopt));
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
