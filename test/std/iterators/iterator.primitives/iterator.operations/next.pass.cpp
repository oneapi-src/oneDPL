//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iterator>

// template <InputIterator Iter>
//   Iter next(Iter x, Iter::difference_type n = 1);

// LWG #2353 relaxed the requirement on next from ForwardIterator to
// InputIterator

#include "oneapi_std_test_config.h"

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
test(It i, typename std::iterator_traits<It>::difference_type n, It x)
{
    return (s::next(i, n) == x);
}

template <class It>
bool
test(It i, It x)
{
    return (s::next(i) == x);
}

#if TEST_STD_VER > 14
template <class It>
constexpr bool
constexpr_test(It i, typename std::iterator_traits<It>::difference_type n, It x)
{
    return s::next(i, n) == x;
}

template <class It>
constexpr bool
constexpr_test(It i, It x)
{
    return s::next(i) == x;
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
                    ret_access[0] &=
                        test(bidirectional_iterator<const char*>(s), 10, bidirectional_iterator<const char*>(s + 10));
                    ret_access[0] &=
                        test(bidirectional_iterator<const char*>(s + 10), -10, bidirectional_iterator<const char*>(s));
                    ret_access[0] &=
                        test(random_access_iterator<const char*>(s), 10, random_access_iterator<const char*>(s + 10));
                    ret_access[0] &=
                        test(random_access_iterator<const char*>(s + 10), -10, random_access_iterator<const char*>(s));
                    ret_access[0] &= test(s, 10, s + 10);

                    ret_access[0] &= test(input_iterator<const char*>(s), input_iterator<const char*>(s + 1));
                    ret_access[0] &= test(forward_iterator<const char*>(s), forward_iterator<const char*>(s + 1));
                    ret_access[0] &=
                        test(bidirectional_iterator<const char*>(s), bidirectional_iterator<const char*>(s + 1));
                    ret_access[0] &=
                        test(random_access_iterator<const char*>(s), random_access_iterator<const char*>(s + 1));
                    ret_access[0] &= test(s, s + 1);
                }

#if TEST_STD_VER > 14
                {
                    constexpr const char* s = "1234567890";
                    static_assert(
                        constexpr_test(input_iterator<const char*>(s), 10, input_iterator<const char*>(s + 10)), "");
                    static_assert(
                        constexpr_test(forward_iterator<const char*>(s), 10, forward_iterator<const char*>(s + 10)),
                        "");
                    static_assert(constexpr_test(bidirectional_iterator<const char*>(s), 10,
                                                 bidirectional_iterator<const char*>(s + 10)),
                                  "");
                    static_assert(constexpr_test(bidirectional_iterator<const char*>(s + 10), -10,
                                                 bidirectional_iterator<const char*>(s)),
                                  "");
                    static_assert(constexpr_test(random_access_iterator<const char*>(s), 10,
                                                 random_access_iterator<const char*>(s + 10)),
                                  "");
                    static_assert(constexpr_test(random_access_iterator<const char*>(s + 10), -10,
                                                 random_access_iterator<const char*>(s)),
                                  "");
                    static_assert(constexpr_test(s, 10, s + 10), "");

                    static_assert(constexpr_test(input_iterator<const char*>(s), input_iterator<const char*>(s + 1)),
                                  "");
                    static_assert(
                        constexpr_test(forward_iterator<const char*>(s), forward_iterator<const char*>(s + 1)), "");
                    static_assert(constexpr_test(bidirectional_iterator<const char*>(s),
                                                 bidirectional_iterator<const char*>(s + 1)),
                                  "");
                    static_assert(constexpr_test(random_access_iterator<const char*>(s),
                                                 random_access_iterator<const char*>(s + 1)),
                                  "");
                    static_assert(constexpr_test(s, s + 1), "");
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
    if (ret)
        std::cout << "Pass" << std::endl;
    else
        std::cout << "Fail" << std::endl;
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
