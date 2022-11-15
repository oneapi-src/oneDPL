//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11, c++14
// <optional>

// constexpr optional(const optional<T>& rhs);

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

class KernelTest1;
class KernelTest2;

template <class KernelTest, class T, class... InitArgs>
bool
test1(InitArgs&&... args)
{
    cl::sycl::queue q;
    bool ret = true;
    const optional<T> rhs(s::forward<InitArgs>(args)...);
    cl::sycl::range<1> numOfItems1{1};
    {
        cl::sycl::buffer<bool, 1> buffer1(&ret, numOfItems1);
        cl::sycl::buffer<optional<T>, 1> buffer2(&rhs, numOfItems1);

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

bool
test2()
{
    cl::sycl::queue q;
    bool ret = true;
    const optional<const int> o(42);
    cl::sycl::range<1> numOfItems1{1};
    {
        cl::sycl::buffer<bool, 1> buffer1(&ret, numOfItems1);
        cl::sycl::buffer<optional<const int>, 1> buffer2(&o, numOfItems1);

        q.submit([&](cl::sycl::handler& cgh) {
            auto ret_access = buffer1.get_access<sycl_write>(cgh);
            auto o_access = buffer2.template get_access<sycl_write>(cgh);
            cgh.single_task<class KernelTest>([=]() {
                optional<const int> o2(o_access[0]);
                ret_access[0] &= (*o2 == 42);
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
    auto ret = test1<KernelTest1, int>();
    ret &= test1<KernelTest2, int>(3);
    ret &= test2();
    if (ret)
        std::cout << "Pass" << std::endl;
    else
        std::cout << "Fail" << std::endl;
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
