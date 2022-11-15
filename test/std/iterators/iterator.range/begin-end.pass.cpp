//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iterator>
// template <class C> constexpr auto begin(C& c) -> decltype(c.begin());
// template <class C> constexpr auto begin(const C& c) -> decltype(c.begin());
// template <class C> constexpr auto cbegin(const C& c) ->
// decltype(std::begin(c)); // C++14 template <class C> constexpr auto
// cend(const C& c) -> decltype(std::end(c));     // C++14 template <class C>
// constexpr auto end  (C& c) -> decltype(c.end()); template <class C> constexpr
// auto end  (const C& c) -> decltype(c.end()); template <class E> constexpr
// reverse_iterator<const E*> rbegin(initializer_list<E> il); template <class E>
// constexpr reverse_iterator<const E*> rend  (initializer_list<E> il);
//
// template <class C> auto constexpr rbegin(C& c) -> decltype(c.rbegin()); //
// C++14 template <class C> auto constexpr rbegin(const C& c) ->
// decltype(c.rbegin());           // C++14 template <class C> auto constexpr
// rend(C& c) -> decltype(c.rend());                     // C++14 template
// <class C> constexpr auto rend(const C& c) -> decltype(c.rend()); // C++14
// template <class T, size_t N> reverse_iterator<T*> constexpr rbegin(T
// (&array)[N]);      // C++14 template <class T, size_t N> reverse_iterator<T*>
// constexpr rend(T (&array)[N]);        // C++14 template <class C> constexpr
// auto crbegin(const C& c) -> decltype(std::rbegin(c));      // C++14 template
// <class C> constexpr auto crend(const C& c) -> decltype(std::rend(c)); //
// C++14
//
//  All of these are constexpr in C++17

#include "oneapi_std_test_config.h"

#include "test_macros.h"
#include <iostream>
#include <initializer_list>
#ifdef USE_ONEAPI_STD
#    include _ONEAPI_STD_TEST_HEADER(iterator)
#    include _ONEAPI_STD_TEST_HEADER(array)
namespace s = oneapi_cpp_ns;
#else
#    include <iterator>
#    include <array>
namespace s = std;
#endif
// std::array is explicitly allowed to be initialized with A a = { init-list };.
// Disable the missing braces warning for this reason.
#include "disable_missing_braces_warning.h"

#if TEST_DPCPP_BACKEND_PRESENT
constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;

template <typename C>
class ConstContainerTest1;
template <typename C>
class ContainerTest1;
template <typename C>
class ConstArrayTest;

template <typename C>
bool
test_const_container(const C& c, typename C::value_type val)
{
    cl::sycl::queue q;
    auto ret = true;
    cl::sycl::range<1> numOfItems{1};
    cl::sycl::buffer<C, 1> buffer1(&c, numOfItems);
    cl::sycl::buffer<bool, 1> buffer2(&ret, numOfItems);
    q.submit([&](cl::sycl::handler& cgh) {
        auto c_access = buffer1.template get_access<sycl_write>(cgh);
        auto ret_access = buffer2.template get_access<sycl_write>(cgh);
        cgh.single_task<ConstContainerTest1<C>>([=]() {
            ret_access[0] &= (s::begin(c_access[0]) == c_access[0].begin());
            ret_access[0] &= (*s::begin(c_access[0]) == val);
            ret_access[0] &= (s::begin(c_access[0]) != c_access[0].end());
            ret_access[0] &= (s::end(c_access[0]) == c_access[0].end());
#if TEST_STD_VER > 11
            ret_access[0] &= (s::cbegin(c_access[0]) == c_access[0].cbegin());
            ret_access[0] &= (s::cbegin(c_access[0]) != c_access[0].cend());
            ret_access[0] &= (s::cend(c_access[0]) == c_access[0].cend());
            ret_access[0] &= (s::rbegin(c_access[0]) == c_access[0].rbegin());
            ret_access[0] &= (s::rbegin(c_access[0]) != c_access[0].rend());
            ret_access[0] &= (s::rend(c_access[0]) == c_access[0].rend());
            ret_access[0] &= (s::crbegin(c_access[0]) == c_access[0].crbegin());
            ret_access[0] &= (s::crbegin(c_access[0]) != c_access[0].crend());
            ret_access[0] &= (s::crend(c_access[0]) == c_access[0].crend());
#endif
        });
    });
    return ret;
}

bool
test_initializer_list()
{
    cl::sycl::queue q;
    auto ret = true;
    cl::sycl::range<1> numOfItems{1};
    cl::sycl::buffer<bool, 1> buffer1(&ret, numOfItems);
    q.submit([&](cl::sycl::handler& cgh) {
        auto ret_access = buffer1.template get_access<sycl_write>(cgh);
        cgh.single_task<class KernelTest1>([=]() {
            {
                std::initializer_list<int> il = {4};
                ret_access[0] &= (s::begin(il) == il.begin());
                ret_access[0] &= (*s::begin(il) == 4);
                ret_access[0] &= (s::begin(il) != il.end());
                ret_access[0] &= (s::end(il) == il.end());
            }
        });
    });
    return ret;
}

template <typename C>
bool
test_container(C& c, typename C::value_type val)
{
    cl::sycl::queue q;
    auto ret = true;
    cl::sycl::range<1> numOfItems{1};
    cl::sycl::buffer<C, 1> buffer1(&c, numOfItems);
    cl::sycl::buffer<bool, 1> buffer2(&ret, numOfItems);
    q.submit([&](cl::sycl::handler& cgh) {
        auto c_access = buffer1.template get_access<sycl_write>(cgh);
        auto ret_access = buffer2.template get_access<sycl_write>(cgh);
        cgh.single_task<ContainerTest1<C>>([=]() {
            ret_access[0] &= (s::begin(c_access[0]) == c_access[0].begin());
            ret_access[0] &= (*s::begin(c_access[0]) == val);
            ret_access[0] &= (s::begin(c_access[0]) != c_access[0].end());
            ret_access[0] &= (s::end(c_access[0]) == c_access[0].end());
#if TEST_STD_VER > 11
            ret_access[0] &= (s::cbegin(c_access[0]) == c_access[0].cbegin());
            ret_access[0] &= (s::cbegin(c_access[0]) != c_access[0].cend());
            ret_access[0] &= (s::cend(c_access[0]) == c_access[0].cend());
            ret_access[0] &= (s::rbegin(c_access[0]) == c_access[0].rbegin());
            ret_access[0] &= (s::rbegin(c_access[0]) != c_access[0].rend());
            ret_access[0] &= (s::rend(c_access[0]) == c_access[0].rend());
            ret_access[0] &= (s::crbegin(c_access[0]) == c_access[0].crbegin());
            ret_access[0] &= (s::crbegin(c_access[0]) != c_access[0].crend());
            ret_access[0] &= (s::crend(c_access[0]) == c_access[0].crend());
#endif
        });
    });
    return ret;
}

void
kernel_test()
{
    cl::sycl::queue q;
    q.submit([&](cl::sycl::handler& cgh) {
        cgh.single_task<class KernelTest>([=]() {
#if TEST_STD_VER > 14
            {
                typedef std::array<int, 5> C;
                constexpr const C c{0, 1, 2, 3, 4};

                static_assert(c.begin() == s::begin(c), "");
                static_assert(c.cbegin() == s::cbegin(c), "");
                static_assert(c.end() == s::end(c), "");
                static_assert(c.cend() == s::cend(c), "");

                static_assert(c.rbegin() == s::rbegin(c), "");
                static_assert(c.crbegin() == s::crbegin(c), "");
                static_assert(c.rend() == s::rend(c), "");
                static_assert(c.crend() == s::crend(c), "");

                static_assert(s::begin(c) != s::end(c), "");
                static_assert(s::rbegin(c) != s::rend(c), "");
                static_assert(s::cbegin(c) != s::cend(c), "");
                static_assert(s::crbegin(c) != s::crend(c), "");

                static_assert(*c.begin() == 0, "");
                static_assert(*c.rbegin() == 4, "");

                static_assert(*s::begin(c) == 0, "");
                static_assert(*s::cbegin(c) == 0, "");
                static_assert(*s::rbegin(c) == 4, "");
                static_assert(*s::crbegin(c) == 4, "");
            }

            {
                static constexpr const int c[] = {0, 1, 2, 3, 4};

                static_assert(*s::begin(c) == 0, "");
                static_assert(*s::cbegin(c) == 0, "");
                static_assert(*s::rbegin(c) == 4, "");
                static_assert(*s::crbegin(c) == 4, "");
            }
#endif
        });
    });
}
#endif // TEST_DPCPP_BACKEND_PRESENT

int
main(int, char**)
{
#if TEST_DPCPP_BACKEND_PRESENT
    std::array<int, 1> a;
    a[0] = 3;

    auto ret = test_container(a, 3);
    ret &= test_const_container(a, 3);
    ret &= test_initializer_list();

    kernel_test();
    TestUtils::exitOnError(true);
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
