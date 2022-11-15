//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iterator>

// reverse_iterator

// Test nested types and data member:

// template <BidirectionalIterator Iter>
// class reverse_iterator {
// protected:
//   Iter current;
// public:
//   iterator<typename iterator_traits<Iterator>::iterator_category,
//   typename iterator_traits<Iterator>::value_type,
//   typename iterator_traits<Iterator>::difference_type,
//   typename iterator_traits<Iterator>::pointer,
//   typename iterator_traits<Iterator>::reference> {
// };

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
constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;

template <class It>
struct find_current : private s::reverse_iterator<It>
{
    void
    test()
    {
        ++(this->current);
    }
};

template <class Tt>
class kernelTest;

template <class It>
void
test()
{
    cl::sycl::queue deviceQueue;
    {
        cl::sycl::range<1> numOfItems{1};
        deviceQueue.submit([&](cl::sycl::handler& cgh) {
            cgh.single_task<kernelTest<It>>([=]() {
                typedef s::reverse_iterator<It> R;
                typedef s::iterator_traits<It> T;
                find_current<It> q;
                q.test();
                static_assert((s::is_same<typename R::iterator_type, It>::value), "");
                static_assert((s::is_same<typename R::value_type, typename T::value_type>::value), "");
                static_assert((s::is_same<typename R::difference_type, typename T::difference_type>::value), "");
                static_assert((s::is_same<typename R::reference, typename T::reference>::value), "");
                static_assert((s::is_same<typename R::pointer, typename s::iterator_traits<It>::pointer>::value), "");
                static_assert((s::is_same<typename R::iterator_category, typename T::iterator_category>::value), "");
            });
        });
    }
}
#endif // TEST_DPCPP_BACKEND_PRESENT

int
main(int, char**)
{
#if TEST_DPCPP_BACKEND_PRESENT
    test<bidirectional_iterator<char*>>();
    test<random_access_iterator<char*>>();
    test<char*>();
    TestUtils::exitOnError(true);
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
