//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iterator>

// template<class Iter>
// struct iterator_traits
// {
//   typedef typename Iter::difference_type difference_type;
//   typedef typename Iter::value_type value_type;
//   typedef typename Iter::pointer pointer;
//   typedef typename Iter::reference reference;
//   typedef typename Iter::iterator_category iterator_category;
// };

#include "oneapi_std_test_config.h"

#include "test_macros.h"

#ifdef USE_ONEAPI_STD
#    include _ONEAPI_STD_TEST_HEADER(iterator)
#    include _ONEAPI_STD_TEST_HEADER(type_traits)
namespace s = oneapi_cpp_ns;
#else
#    include <iterator>
#    include <type_traits>
namespace s = std;
#endif

struct A
{
};

struct test_iterator
{
    typedef int difference_type;
    typedef A value_type;
    typedef A* pointer;
    typedef A& reference;
    typedef std::forward_iterator_tag iterator_category;
};

void
kernelTest()
{
    cl::sycl::queue q;
    q.submit([&](cl::sycl::handler& cgh) {
        cgh.single_task<class IteratorTest>([=]() {
            typedef s::iterator_traits<test_iterator> It;
            static_assert((s::is_same<It::difference_type, int>::value), "");
            static_assert((s::is_same<It::value_type, A>::value), "");
            static_assert((s::is_same<It::pointer, A*>::value), "");
            static_assert((s::is_same<It::reference, A&>::value), "");
            static_assert((s::is_same<It::iterator_category, s::forward_iterator_tag>::value), "");
        });
    });
}

int
main(int, char**)
{
    kernelTest();
    std::cout << "Pass" << std::endl;
    return 0;
}
