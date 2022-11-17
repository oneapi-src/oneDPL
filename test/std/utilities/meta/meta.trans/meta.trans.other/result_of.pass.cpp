//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <functional>

// result_of<Fn(ArgTypes...)>

#include "oneapi_std_test_config.h"
#include "test_macros.h"

#include <iostream>

#ifdef USE_ONEAPI_STD
#    include _ONEAPI_STD_TEST_HEADER(type_traits)
#    include _ONEAPI_STD_TEST_HEADER(functional)
namespace s = oneapi_cpp_ns;
#else
#    include <type_traits>
namespace s = std;
#endif

constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;

struct S
{
    typedef short (*FreeFunc)(long);
    operator FreeFunc() const;
    double
    operator()(char, int&);
    double const&
    operator()(char, int&) const;
    double volatile&
    operator()(char, int&) volatile;
    double const volatile&
    operator()(char, int&) const volatile;
};

struct SD : public S
{
};

struct NotDerived
{
};

template <class Tp>
struct Voider
{
    typedef void type;
};

template <class T, class = void>
struct HasType : s::false_type
{
};

template <class T>
struct HasType<T, typename Voider<typename T::type>::type> : s::true_type
{
};

#if TEST_STD_VER > 14
template <typename T, typename U>
struct test_invoke_result;

template <typename Fn, typename... Args, typename Ret>
struct test_invoke_result<Fn(Args...), Ret>
{
    static void
    call()
    {
        static_assert(s::is_invocable<Fn, Args...>::value, "");
        static_assert(s::is_invocable_r<Ret, Fn, Args...>::value, "");
        ASSERT_SAME_TYPE(Ret, typename s::invoke_result<Fn, Args...>::type);
    }
};
#endif

template <class KernelTest, class T, class U>
void
test_result_of(cl::sycl::queue& deviceQueue)
{
    deviceQueue.submit([&](cl::sycl::handler& cgh) {
        cgh.single_task<KernelTest>([=]() {
            ASSERT_SAME_TYPE(U, typename s::result_of<T>::type);
#if TEST_STD_VER > 14
            test_invoke_result<T, U>::call();
#endif
        });
    });
}

#if TEST_STD_VER > 14
template <typename T>
struct test_invoke_no_result;

template <typename Fn, typename... Args>
struct test_invoke_no_result<Fn(Args...)>
{
    static void
    call()
    {
        static_assert(s::is_invocable<Fn, Args...>::value == false, "");
        static_assert((!HasType<s::invoke_result<Fn, Args...>>::value), "");
    }
};
#endif

template <class KernelTest, class T>
void
test_no_result(cl::sycl::queue& deviceQueue)
{
    deviceQueue.submit([&](cl::sycl::handler& cgh) {
        cgh.single_task<KernelTest>([=]() {
            static_assert((!HasType<s::result_of<T>>::value), "");
#if TEST_STD_VER > 14
            test_invoke_no_result<T>::call();
#endif
        });
    });
}

class KernelTest1;
class KernelTest2;
class KernelTest3;
class KernelTest4;
class KernelTest5;
class KernelTest6;
class KernelTest7;
class KernelTest8;
class KernelTest9;
class KernelTest10;
class KernelTest11;
class KernelTest12;
class KernelTest13;
class KernelTest14;
class KernelTest15;
class KernelTest16;
class KernelTest17;
class KernelTest18;
class KernelTest19;
class KernelTest20;
class KernelTest21;

void
kernel_test()
{
    cl::sycl::queue deviceQueue;
    typedef NotDerived ND;
    { // functor object
        test_result_of<KernelTest1, S(int), short>(deviceQueue);
        if (deviceQueue.get_device().has_extension("cl_khr_fp64"))
        {
            test_result_of<KernelTest2, S&(unsigned char, int&), double>(deviceQueue);
            test_result_of<KernelTest3, S const&(unsigned char, int&), double const&>(deviceQueue);
            test_result_of<KernelTest4, S volatile&(unsigned char, int&), double volatile&>(deviceQueue);
            test_result_of<KernelTest5, S const volatile&(unsigned char, int&), double const volatile&>(deviceQueue);
        }
    }
    { // pointer to member data
        typedef char S::*PMD;
        test_result_of<KernelTest6, PMD(S&), char&>(deviceQueue);
        test_result_of<KernelTest7, PMD(S*), char&>(deviceQueue);
        test_result_of<KernelTest8, PMD(S* const), char&>(deviceQueue);
        test_result_of<KernelTest9, PMD(const S&), const char&>(deviceQueue);
        test_result_of<KernelTest10, PMD(const S*), const char&>(deviceQueue);
        test_result_of<KernelTest11, PMD(volatile S&), volatile char&>(deviceQueue);
        test_result_of<KernelTest12, PMD(volatile S*), volatile char&>(deviceQueue);
        test_result_of<KernelTest13, PMD(const volatile S&), const volatile char&>(deviceQueue);
        test_result_of<KernelTest14, PMD(const volatile S*), const volatile char&>(deviceQueue);
        test_result_of<KernelTest15, PMD(SD&), char&>(deviceQueue);
        test_result_of<KernelTest16, PMD(SD const&), const char&>(deviceQueue);
        test_result_of<KernelTest17, PMD(SD*), char&>(deviceQueue);
        test_result_of<KernelTest18, PMD(const SD*), const char&>(deviceQueue);
        test_result_of<KernelTest19, PMD(s::reference_wrapper<S>), char&>(deviceQueue);
        test_result_of<KernelTest20, PMD(s::reference_wrapper<S const>), const char&>(deviceQueue);
        test_no_result<KernelTest21, PMD(ND&)>(deviceQueue);
    }
}

int
main()
{
    kernel_test();
    std::cout << "Pass" << std::endl;
    return 0;
}
