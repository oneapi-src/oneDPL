//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iterator>

// template<class Category, class T, class Distance = ptrdiff_t,
//          class Pointer = T*, class Reference = T&>
// struct iterator
// {
//   typedef T         value_type;
//   typedef Distance  difference_type;
//   typedef Pointer   pointer;
//   typedef Reference reference;
//   typedef Category  iterator_category;
// };

#include "oneapi_std_test_config.h"
#include "test_macros.h"

#include <iostream>

#ifdef USE_ONEAPI_STD
#    include _ONEAPI_STD_TEST_HEADER(iterator)
#    include _ONEAPI_STD_TEST_HEADER(type_traits)
#    include _ONEAPI_STD_TEST_HEADER(cstddef)
namespace s = oneapi_cpp_ns;
#else
#    include <iterator>
namespace s = std;
#endif

#if TEST_DPCPP_BACKEND_PRESENT
struct A
{
};

template <class T>
class IteratorTest;

template <class T>
void
kernelTest()
{
    cl::sycl::queue q;
    q.submit([&](cl::sycl::handler& cgh) {
        cgh.single_task<IteratorTest<T>>([=]() {
            {
                typedef s::iterator<s::forward_iterator_tag, T> It;
                static_assert((s::is_same<typename It::value_type, T>::value), "");
                static_assert((s::is_same<typename It::difference_type, s::ptrdiff_t>::value), "");
                static_assert((s::is_same<typename It::pointer, T*>::value), "");
                static_assert((s::is_same<typename It::reference, T&>::value), "");
                static_assert((s::is_same<typename It::iterator_category, s::forward_iterator_tag>::value), "");
            }
            {
                typedef s::iterator<s::bidirectional_iterator_tag, T, short> It;
                static_assert((s::is_same<typename It::value_type, T>::value), "");
                static_assert((s::is_same<typename It::difference_type, short>::value), "");
                static_assert((s::is_same<typename It::pointer, T*>::value), "");
                static_assert((s::is_same<typename It::reference, T&>::value), "");
                static_assert((s::is_same<typename It::iterator_category, s::bidirectional_iterator_tag>::value), "");
            }
            {
                typedef s::iterator<s::random_access_iterator_tag, T, int, const T*> It;
                static_assert((s::is_same<typename It::value_type, T>::value), "");
                static_assert((s::is_same<typename It::difference_type, int>::value), "");
                static_assert((s::is_same<typename It::pointer, const T*>::value), "");
                static_assert((s::is_same<typename It::reference, T&>::value), "");
                static_assert((s::is_same<typename It::iterator_category, s::random_access_iterator_tag>::value), "");
            }
            {
                typedef s::iterator<s::input_iterator_tag, T, long, const T*, const T&> It;
                static_assert((s::is_same<typename It::value_type, T>::value), "");
                static_assert((s::is_same<typename It::difference_type, long>::value), "");
                static_assert((s::is_same<typename It::pointer, const T*>::value), "");
                static_assert((s::is_same<typename It::reference, const T&>::value), "");
                static_assert((s::is_same<typename It::iterator_category, s::input_iterator_tag>::value), "");
            }
        });
    });
}
#endif // TEST_DPCPP_BACKEND_PRESENT

int
main(int, char**)
{
#if TEST_DPCPP_BACKEND_PRESENT
    kernelTest<A>();
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
