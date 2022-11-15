//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iterator>

// template<class T>
// struct iterator_traits<T*>
// {
//   typedef ptrdiff_t                  difference_type;
//   typedef T                          value_type;
//   typedef T*                         pointer;
//   typedef T&                         reference;
//   typedef random_access_iterator_tag iterator_category;
//   typedef contiguous_iterator_tag iterator_category; // C++20
// };

#include "oneapi_std_test_config.h"

#include "test_macros.h"

#ifdef USE_ONEAPI_STD
#    include _ONEAPI_STD_TEST_HEADER(iterator)
#    include _ONEAPI_STD_TEST_HEADER(type_traits)
#    include _ONEAPI_STD_TEST_HEADER(cstddef)
namespace s = oneapi_cpp_ns;
#else
#    include <iterator>
#    include <type_traits>
namespace s = std;
#endif

#if TEST_DPCPP_BACKEND_PRESENT
struct A
{
};

void
kernelTest()
{
    cl::sycl::queue q;
    q.submit([&](cl::sycl::handler& cgh) {
        cgh.single_task<class IteratorTest>([=]() {
            typedef s::iterator_traits<A*> It;
            static_assert((s::is_same<It::difference_type, s::ptrdiff_t>::value), "");
            static_assert((s::is_same<It::value_type, A>::value), "");
            static_assert((s::is_same<It::pointer, A*>::value), "");
            static_assert((s::is_same<It::reference, A&>::value), "");
            static_assert((s::is_same<It::iterator_category, s::random_access_iterator_tag>::value), "");
#if TEST_STD_VER > 17
            ASSERT_SAME_TYPE(It::iterator_concept, s::contiguous_iterator_tag);
#endif
        });
    });
}
#endif // TEST_DPCPP_BACKEND_PRESENT

int
main(int, char**)
{
#if TEST_DPCPP_BACKEND_PRESENT
    kernelTest();
    TestUtils::exitOnError(true);
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
