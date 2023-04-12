//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iterator>

// reverse_iterator

// requires RandomAccessIterator<Iter>
//   unspecified operator[](difference_type n) const;

#include "oneapi_std_test_config.h"

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

#if TEST_DPCPP_BACKEND_PRESENT
constexpr sycl::access::mode sycl_read = sycl::access::mode::read;
constexpr sycl::access::mode sycl_write = sycl::access::mode::write;

template <class It>
bool
test(It i, typename s::iterator_traits<It>::difference_type n, typename s::iterator_traits<It>::value_type x)
{
    typedef typename s::iterator_traits<It>::value_type value_type;
    const s::reverse_iterator<It> r(i);
    value_type rr = r[n];
    return (rr == x);
}

bool
kernel_test()
{
    sycl::queue deviceQueue;
    sycl::cl_bool ret = true;
    {
        sycl::range<1> numOfItems{1};
        sycl::buffer<sycl::cl_bool, 1> buffer1(&ret, numOfItems);
        deviceQueue.submit([&](sycl::handler& cgh) {
            auto ret_access = buffer1.get_access<sycl_write>(cgh);
            cgh.single_task<class KernelTest>([=]() {
                const char* s = "1234567890";
                ret_access[0] &= test(random_access_iterator<const char*>(s + 5), 4, '1');
                ret_access[0] &= test(s + 5, 4, '1');

#if TEST_STD_VER > 14
                {
                    constexpr const char* p = "123456789";
                    typedef s::reverse_iterator<const char*> RI;
                    constexpr RI it1 = s::make_reverse_iterator(p + 5);
                    static_assert(it1[0] == '5', "");
                    static_assert(it1[4] == '1', "");
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
