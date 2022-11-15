//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iterator>

// template <InputIterator Iter>
//   Iter::difference_type
//   distance(Iter first, Iter last);
//
// template <RandomAccessIterator Iter>
//   Iter::difference_type
//   distance(Iter first, Iter last);

#include "oneapi_std_test_config.h"

#include <iostream>
#include "test_iterators.h"
#include "test_macros.h"

#ifdef USE_ONEAPI_STD
#    include _ONEAPI_STD_TEST_HEADER(iterator)
namespace s = oneapi_cpp_ns;
#else
#    include <iterator>
namespace s = std;
#endif

#if TEST_DPCPP_BACKEND_PRESENT
constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;

template <class It>
bool
test(It first, It last, typename std::iterator_traits<It>::difference_type x)
{
    return (s::distance(first, last) == x);
}

#if TEST_STD_VER > 14
template <class It>
constexpr bool
constexpr_test(It first, It last, typename std::iterator_traits<It>::difference_type x)
{
    return s::distance(first, last) == x;
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
                    ret_access[0] &= test(input_iterator<const char*>(s), input_iterator<const char*>(s + 10), 10);
                    ret_access[0] &= test(forward_iterator<const char*>(s), forward_iterator<const char*>(s + 10), 10);
                    ret_access[0] &=
                        test(bidirectional_iterator<const char*>(s), bidirectional_iterator<const char*>(s + 10), 10);
                    ret_access[0] &=
                        test(random_access_iterator<const char*>(s), random_access_iterator<const char*>(s + 10), 10);
                    ret_access[0] &= test(s, s + 10, 10);
                }

#if TEST_STD_VER > 14
                {
                    constexpr const char* s = "1234567890";
                    static_assert(
                        constexpr_test(input_iterator<const char*>(s), input_iterator<const char*>(s + 10), 10), "");
                    static_assert(
                        constexpr_test(forward_iterator<const char*>(s), forward_iterator<const char*>(s + 10), 10),
                        "");
                    static_assert(constexpr_test(bidirectional_iterator<const char*>(s),
                                                 bidirectional_iterator<const char*>(s + 10), 10),
                                  "");
                    static_assert(constexpr_test(random_access_iterator<const char*>(s),
                                                 random_access_iterator<const char*>(s + 10), 10),
                                  "");
                    static_assert(constexpr_test(s, s + 10, 10), "");
                }
#endif
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
