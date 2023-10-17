//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iterator>

//   All of these became constexpr in C++17
//
// template <InputIterator Iter>
//   constexpr void advance(Iter& i, Iter::difference_type n);
//
// template <BidirectionalIterator Iter>
//   constexpr void advance(Iter& i, Iter::difference_type n);
//
// template <RandomAccessIterator Iter>
//   constexpr void advance(Iter& i, Iter::difference_type n);

#include "oneapi_std_test_config.h"
#include <CL/sycl.hpp>
#include "test_iterators.h"
#include "test_macros.h"
#include <iostream>

#ifdef USE_ONEAPI_STD
#    include _ONEAPI_STD_TEST_HEADER(iterator)
namespace s = oneapi_cpp_ns;
#else
#    include <iterator>
namespace s = std;
#endif

constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;

template <class It>
bool
test(It i, typename std::iterator_traits<It>::difference_type n, It x)
{
    s::advance(i, n);
    return (i == x);
}

#if TEST_STD_VER > 14
template <class It>
constexpr bool
constepxr_test(It i, typename std::iterator_traits<It>::difference_type n, It x)
{
    s::advance(i, n);
    return i == x;
}
#endif

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
                {
                    const char* s = "1234567890";
                    ret_access[0] &= test(input_iterator<const char*>(s), 10, input_iterator<const char*>(s + 10));
                    ret_access[0] &= test(forward_iterator<const char*>(s), 10, forward_iterator<const char*>(s + 10));
                    ret_access[0] &= test(bidirectional_iterator<const char*>(s + 5), 5,
                                          bidirectional_iterator<const char*>(s + 10));
                    ret_access[0] &=
                        test(bidirectional_iterator<const char*>(s + 5), -5, bidirectional_iterator<const char*>(s));
                    ret_access[0] &= test(random_access_iterator<const char*>(s + 5), 5,
                                          random_access_iterator<const char*>(s + 10));
                    ret_access[0] &=
                        test(random_access_iterator<const char*>(s + 5), -5, random_access_iterator<const char*>(s));
                    ret_access[0] &= test(s + 5, 5, s + 10);
                    ret_access[0] &= test(s + 5, -5, s);
                }

#if TEST_STD_VER > 14
                {
                    constexpr const char* s = "1234567890";
                    static_assert(
                        constepxr_test(input_iterator<const char*>(s), 10, input_iterator<const char*>(s + 10)), "");
                    static_assert(
                        constepxr_test(forward_iterator<const char*>(s), 10, forward_iterator<const char*>(s + 10)),
                        "");
                    static_assert(constepxr_test(bidirectional_iterator<const char*>(s + 5), 5,
                                                 bidirectional_iterator<const char*>(s + 10)),
                                  "");
                    static_assert(constepxr_test(bidirectional_iterator<const char*>(s + 5), -5,
                                                 bidirectional_iterator<const char*>(s)),
                                  "");
                    static_assert(constepxr_test(random_access_iterator<const char*>(s + 5), 5,
                                                 random_access_iterator<const char*>(s + 10)),
                                  "");
                    static_assert(constepxr_test(random_access_iterator<const char*>(s + 5), -5,
                                                 random_access_iterator<const char*>(s)),
                                  "");
                    static_assert(constepxr_test(s + 5, 5, s + 10), "");
                    static_assert(constepxr_test(s + 5, -5, s), "");
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
