//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iterator>

// reverse_iterator

// template <RandomAccessIterator Iter1, RandomAccessIterator Iter2>
//   requires HasMinus<Iter2, Iter1>
//   constexpr auto operator-(const reverse_iterator<Iter1>& x, const
//   reverse_iterator<Iter2>& y)
//   -> decltype(y.base() - x.base());
//
// constexpr in C++17

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

template <class It1, class It2>
bool
test(It1 l, It2 r, std::ptrdiff_t x)
{
    const s::reverse_iterator<It1> r1(l);
    const s::reverse_iterator<It2> r2(r);
    return ((r1 - r2) == x);
}

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
                char s[3] = {0};
                ret_access[0] &= test(random_access_iterator<const char*>(s), random_access_iterator<char*>(s), 0);
                ret_access[0] &= test(random_access_iterator<char*>(s), random_access_iterator<const char*>(s + 1), 1);
                ret_access[0] &= test(random_access_iterator<const char*>(s + 1), random_access_iterator<char*>(s), -1);
                ret_access[0] &= test(s, s, 0);
                ret_access[0] &= test(s, s + 1, 1);
                ret_access[0] &= test(s + 1, s, -1);

#if TEST_STD_VER > 14
                {
                    constexpr const char* p = "123456789";
                    typedef s::reverse_iterator<const char*> RI;
                    constexpr RI it1 = s::make_reverse_iterator(p);
                    constexpr RI it2 = s::make_reverse_iterator(p + 1);
                    static_assert(it1 - it2 == 1, "");
                    static_assert(it2 - it1 == -1, "");
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
