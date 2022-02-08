//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iterator>

// move_iterator

// template <class U>
//   requires HasConstructor<Iter, const U&>
//   move_iterator(const move_iterator<U> &u);
//
//  constexpr in C++17

#include "oneapi_std_test_config.h"
#include <CL/sycl.hpp>
#include <iostream>
#include "test_macros.h"
#include "test_iterators.h"

#ifdef USE_ONEAPI_STD
#    include _ONEAPI_STD_TEST_HEADER(iterator)
#    include _ONEAPI_STD_TEST_HEADER(type_traits)
namespace s = oneapi_cpp_ns;
#else
#    include <iterator>
#    include <type_traits>
namespace s = std;
#endif

constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;

template <class It, class U>
bool
test(U u)
{
    const s::move_iterator<U> r2(u);
    s::move_iterator<It> r1 = r2;
    return (r1.base() == u);
}

struct Base
{
};
struct Derived : Base
{
};

bool
kernel_test()
{
    cl::sycl::queue deviceQueue;
    cl::sycl::cl_bool ret = true;
    {
        cl::sycl::range<1> numOfItems{1};
        cl::sycl::buffer<cl::sycl::cl_bool, 1> buffer1(&ret, numOfItems);
        deviceQueue.submit([&](cl::sycl::handler& cgh) {
            auto ret_access = buffer1.get_access<sycl_write>(cgh);
            cgh.single_task<class KernelTest>([=]() {
                Derived d;

                ret_access[0] &= test<input_iterator<Base*>>(input_iterator<Derived*>(&d));
                ret_access[0] &= test<forward_iterator<Base*>>(forward_iterator<Derived*>(&d));
                ret_access[0] &= test<bidirectional_iterator<Base*>>(bidirectional_iterator<Derived*>(&d));
                ret_access[0] &= test<random_access_iterator<const Base*>>(random_access_iterator<Derived*>(&d));
                ret_access[0] &= test<Base*>(&d);

#if TEST_STD_VER > 14
                {
                    constexpr const Derived* p = nullptr;
                    constexpr s::move_iterator<const Derived*> it1 = s::make_move_iterator(p);
                    constexpr s::move_iterator<const Base*> it2(it1);
                    static_assert(it2.base() == p);
                }
#endif
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
