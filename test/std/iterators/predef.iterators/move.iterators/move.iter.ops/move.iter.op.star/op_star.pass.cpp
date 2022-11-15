//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iterator>

// move_iterator

// reference operator*() const;
//
//  constexpr in C++17

#include "oneapi_std_test_config.h"

#include <iostream>
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

#if TEST_DPCPP_BACKEND_PRESENT
constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;

class A
{
    int data_;

  public:
    A() : data_(1) {}
    ~A() { data_ = -1; }

    friend bool
    operator==(const A& x, const A& y)
    {
        return x.data_ == y.data_;
    }
};

template <class It>
bool
test(It i, typename s::iterator_traits<It>::value_type x)
{
    s::move_iterator<It> r(i);
    auto ret = (*r == x);
    typename s::iterator_traits<It>::value_type x2 = *r;
    ret &= (x2 == x);
    return ret;
}

struct do_nothing
{
    void
    operator()(void*) const
    {
    }
};

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
                    A a;
                    ret_access[0] &= test(&a, A());
                }
#if TEST_STD_VER > 14
                {
                    constexpr const char* p = "123456789";
                    typedef s::move_iterator<const char*> MI;
                    constexpr MI it1 = s::make_move_iterator(p);
                    constexpr MI it2 = s::make_move_iterator(p + 1);
                    static_assert(*it1 == p[0], "");
                    static_assert(*it2 == p[1], "");
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
