//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iterator>

// reverse_iterator

// constexpr pointer operator->() const;
//
// constexpr in C++17

// Be sure to respect LWG 198:
//    http://www.open-std.org/jtc1/sc22/wg21/docs/lwg-defects.html#198
// LWG 198 was superseded by LWG 2360
//    http://www.open-std.org/jtc1/sc22/wg21/docs/lwg-defects.html#2360

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

constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;

class A
{
    int data_;

  public:
    A() : data_(1) {}
    ~A() { data_ = -1; }

    int
    get() const
    {
        return data_;
    }

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
    s::reverse_iterator<It> r(i);
    return (r->get() == x.get());
}

class B
{
    int data_;

  public:
    B(int d = 1) : data_(d) {}
    ~B() { data_ = -1; }

    int
    get() const
    {
        return data_;
    }

    friend bool
    operator==(const B& x, const B& y)
    {
        return x.data_ == y.data_;
    }
    const B* operator&() const { return nullptr; }
    B* operator&() { return nullptr; }
};

class C
{
    int data_;

  public:
    TEST_CONSTEXPR
    C() : data_(1) {}

    TEST_CONSTEXPR int
    get() const
    {
        return data_;
    }

    friend TEST_CONSTEXPR bool
    operator==(const C& x, const C& y)
    {
        return x.data_ == y.data_;
    }
};

TEST_CONSTEXPR C gC;

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
                A a;
                ret_access[0] &= test(&a + 1, A());

#if TEST_STD_VER > 14
                {
                    typedef s::reverse_iterator<const C*> RI;
                    constexpr RI it1 = s::make_reverse_iterator(&gC + 1);

                    static_assert(it1->get() == gC.get(), "");
                }
#endif
                {
                    ((void)gC);
                }
            });
        });
    }
    return ret;
}

int
main(int, char**)
{
    auto ret = kernel_test();
    if (ret)
        std::cout << "Pass" << std::endl;
    else
        std::cout << "Fail" << std::endl;
    return 0;
}
