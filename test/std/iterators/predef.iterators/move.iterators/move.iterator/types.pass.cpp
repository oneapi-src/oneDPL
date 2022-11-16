//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iterator>

// move_iterator

// Test nested types:

// template <InputIterator Iter>
// class move_iterator {
// public:
//   typedef Iter                  iterator_type;
//   typedef Iter::difference_type difference_type;
//   typedef Iter                  pointer;
//   typedef Iter::value_type      value_type;
//   typedef value_type&&          reference;
// };

#include "oneapi_std_test_config.h"
#include "test_iterators.h"
#include "test_macros.h"

#include <iostream>

#ifdef USE_ONEAPI_STD
#    include _ONEAPI_STD_TEST_HEADER(iterator)
#    include _ONEAPI_STD_TEST_HEADER(type_traits)
#    include _ONEAPI_STD_TEST_HEADER(cstddef)
#    include _ONEAPI_STD_TEST_HEADER(functional)
namespace s = oneapi_cpp_ns;
#else
#    include <iterator>
#    include <type_traits>
namespace s = std;
#endif

#if TEST_DPCPP_BACKEND_PRESENT
constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;

template <class ValueType, class Reference>
struct DummyIt
{
    typedef s::forward_iterator_tag iterator_category;
    typedef ValueType value_type;
    typedef s::ptrdiff_t difference_type;
    typedef ValueType* pointer;
    typedef Reference reference;
};

template <class It>
void
test()
{
    typedef s::move_iterator<It> R;
    typedef s::iterator_traits<It> T;
    static_assert((s::is_same<typename R::iterator_type, It>::value), "");
    static_assert((s::is_same<typename R::difference_type, typename T::difference_type>::value), "");
    static_assert((s::is_same<typename R::pointer, It>::value), "");
    static_assert((s::is_same<typename R::value_type, typename T::value_type>::value), "");
    static_assert((s::is_same<typename R::reference, typename R::value_type&&>::value), "");
    static_assert((s::is_same<typename R::iterator_category, typename T::iterator_category>::value), "");
}

void
kernel_test()
{
    cl::sycl::queue deviceQueue;
    {
        cl::sycl::range<1> numOfItems{1};
        deviceQueue.submit([&](cl::sycl::handler& cgh) {
            cgh.single_task<class KernelTest>([=]() {
                test<input_iterator<char*>>();
                test<forward_iterator<char*>>();
                test<bidirectional_iterator<char*>>();
                test<random_access_iterator<char*>>();
                test<char*>();
                {
                    typedef DummyIt<int, int> T;
                    typedef s::move_iterator<T> It;
                    static_assert(s::is_same<It::reference, int>::value, "");
                }
                {
                    typedef DummyIt<int, s::reference_wrapper<int>> T;
                    typedef s::move_iterator<T> It;
                    static_assert(s::is_same<It::reference, s::reference_wrapper<int>>::value, "");
                }
                {
                    // Check that move_iterator uses whatever reference type it's given
                    // when it's not a reference.
                    typedef DummyIt<int, long> T;
                    typedef s::move_iterator<T> It;
                    static_assert(s::is_same<It::reference, long>::value, "");
                }
                {
                    typedef DummyIt<int, int&> T;
                    typedef s::move_iterator<T> It;
                    static_assert(s::is_same<It::reference, int&&>::value, "");
                }
                {
                    typedef DummyIt<int, int&&> T;
                    typedef s::move_iterator<T> It;
                    static_assert(s::is_same<It::reference, int&&>::value, "");
                }
            });
        });
    }
}
#endif // TEST_DPCPP_BACKEND_PRESENT

int
main(int, char**)
{
#if TEST_DPCPP_BACKEND_PRESENT
    kernel_test();
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
