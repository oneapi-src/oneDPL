//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iterator>

// template<class NotAnIterator>
// struct iterator_traits
// {
// };

#include "oneapi_std_test_config.h"

#include <iostream>
#include "test_macros.h"

#ifdef USE_ONEAPI_STD
#    include _ONEAPI_STD_TEST_HEADER(iterator)
namespace s = oneapi_cpp_ns;
#else
#    include <iterator>
namespace s = std;
#endif

struct not_an_iterator
{
};

template <class T>
struct has_value_type
{
  private:
    struct two
    {
        char lx;
        char lxx;
    };
    template <class U>
    static two
    test(...);
    template <class U>
    static char
    test(typename U::value_type* = 0);

  public:
    static const bool value = sizeof(test<T>(0)) == 1;
};

void
kernelTest()
{
    cl::sycl::queue q;
    q.submit([&](cl::sycl::handler& cgh) {
        cgh.single_task<class IteratorTest>([=]() {
            typedef s::iterator_traits<not_an_iterator> It;
            static_assert(!(has_value_type<It>::value), "");
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
