//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: c++98, c++03
//
// <functional>
//
// result_of<Fn(ArgTypes...)>

#include "oneapi_std_test_config.h"
#include "test_macros.h"
#include <CL/sycl.hpp>
#include <iostream>

#ifdef USE_ONEAPI_STD
#    include _ONEAPI_STD_TEST_HEADER(type_traits)
#    include _ONEAPI_STD_TEST_HEADER(utility)
#    include _ONEAPI_STD_TEST_HEADER(functional)
namespace s = oneapi_cpp_ns;
#else
#    include <type_traits>
#    include <utility>
namespace s = std;
#endif

constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;

struct wat
{
    wat& operator*() { return *this; }
    void
    foo();
};

struct F
{
};
struct FD : public F
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
        ASSERT_SAME_TYPE(Ret, s::invoke_result_t<Fn, Args...>);
    }
};
#endif

template <class T, class U>
void
test_result_of_imp()
{
    ASSERT_SAME_TYPE(U, typename s::result_of<T>::type);
#if TEST_STD_VER > 11
    ASSERT_SAME_TYPE(U, s::result_of_t<T>);
#endif
#if TEST_STD_VER > 14
    test_invoke_result<T, U>::call();
#endif
}

cl::sycl::cl_bool
kernel_test()
{
    {
        typedef char F::*PMD;
        test_result_of_imp<PMD(F&), char&>();
        test_result_of_imp<PMD(F const&), char const&>();
        test_result_of_imp<PMD(F volatile&), char volatile&>();
        test_result_of_imp<PMD(F const volatile&), char const volatile&>();

        test_result_of_imp<PMD(F &&), char&&>();
        test_result_of_imp<PMD(F const&&), char const&&>();
        test_result_of_imp<PMD(F volatile &&), char volatile&&>();
        test_result_of_imp<PMD(F const volatile&&), char const volatile&&>();

        test_result_of_imp<PMD(F), char&&>();
        test_result_of_imp<PMD(F const), char&&>();
        test_result_of_imp<PMD(F volatile), char&&>();
        test_result_of_imp<PMD(F const volatile), char&&>();

        test_result_of_imp<PMD(FD&), char&>();
        test_result_of_imp<PMD(FD const&), char const&>();
        test_result_of_imp<PMD(FD volatile&), char volatile&>();
        test_result_of_imp<PMD(FD const volatile&), char const volatile&>();

        test_result_of_imp<PMD(FD &&), char&&>();
        test_result_of_imp<PMD(FD const&&), char const&&>();
        test_result_of_imp<PMD(FD volatile &&), char volatile&&>();
        test_result_of_imp<PMD(FD const volatile&&), char const volatile&&>();

        test_result_of_imp<PMD(FD), char&&>();
        test_result_of_imp<PMD(FD const), char&&>();
        test_result_of_imp<PMD(FD volatile), char&&>();
        test_result_of_imp<PMD(FD const volatile), char&&>();

        test_result_of_imp<PMD(s::reference_wrapper<F>), char&>();
        test_result_of_imp<PMD(s::reference_wrapper<F const>), const char&>();
        test_result_of_imp<PMD(s::reference_wrapper<FD>), char&>();
        test_result_of_imp<PMD(s::reference_wrapper<FD const>), const char&>();
    }
    {
        test_result_of_imp<int (F::*(F&))()&, int>();
        test_result_of_imp<int (F::*(F&))() const&, int>();
        test_result_of_imp<int (F::*(F&))() volatile&, int>();
        test_result_of_imp<int (F::*(F&))() const volatile&, int>();
        test_result_of_imp<int (F::*(F const&))() const&, int>();
        test_result_of_imp<int (F::*(F const&))() const volatile&, int>();
        test_result_of_imp<int (F::*(F volatile&))() volatile&, int>();
        test_result_of_imp<int (F::*(F volatile&))() const volatile&, int>();
        test_result_of_imp<int (F::*(F const volatile&))() const volatile&, int>();

        test_result_of_imp<int (F::*(F &&))()&&, int>();
        test_result_of_imp<int (F::*(F &&))() const&&, int>();
        test_result_of_imp<int (F::*(F &&))() volatile&&, int>();
        test_result_of_imp<int (F::*(F &&))() const volatile&&, int>();
        test_result_of_imp<int (F::*(F const&&))() const&&, int>();
        test_result_of_imp<int (F::*(F const&&))() const volatile&&, int>();
        test_result_of_imp<int (F::*(F volatile &&))() volatile&&, int>();
        test_result_of_imp<int (F::*(F volatile &&))() const volatile&&, int>();
        test_result_of_imp<int (F::*(F const volatile&&))() const volatile&&, int>();

        test_result_of_imp<int (F::*(F))()&&, int>();
        test_result_of_imp<int (F::*(F))() const&&, int>();
        test_result_of_imp<int (F::*(F))() volatile&&, int>();
        test_result_of_imp<int (F::*(F))() const volatile&&, int>();
        test_result_of_imp<int (F::*(F const))() const&&, int>();
        test_result_of_imp<int (F::*(F const))() const volatile&&, int>();
        test_result_of_imp<int (F::*(F volatile))() volatile&&, int>();
        test_result_of_imp<int (F::*(F volatile))() const volatile&&, int>();
        test_result_of_imp<int (F::*(F const volatile))() const volatile&&, int>();
    }
    {
        test_result_of_imp<int (F::*(FD&))()&, int>();
        test_result_of_imp<int (F::*(FD&))() const&, int>();
        test_result_of_imp<int (F::*(FD&))() volatile&, int>();
        test_result_of_imp<int (F::*(FD&))() const volatile&, int>();
        test_result_of_imp<int (F::*(FD const&))() const&, int>();
        test_result_of_imp<int (F::*(FD const&))() const volatile&, int>();
        test_result_of_imp<int (F::*(FD volatile&))() volatile&, int>();
        test_result_of_imp<int (F::*(FD volatile&))() const volatile&, int>();
        test_result_of_imp<int (F::*(FD const volatile&))() const volatile&, int>();

        test_result_of_imp<int (F::*(FD &&))()&&, int>();
        test_result_of_imp<int (F::*(FD &&))() const&&, int>();
        test_result_of_imp<int (F::*(FD &&))() volatile&&, int>();
        test_result_of_imp<int (F::*(FD &&))() const volatile&&, int>();
        test_result_of_imp<int (F::*(FD const&&))() const&&, int>();
        test_result_of_imp<int (F::*(FD const&&))() const volatile&&, int>();
        test_result_of_imp<int (F::*(FD volatile &&))() volatile&&, int>();
        test_result_of_imp<int (F::*(FD volatile &&))() const volatile&&, int>();
        test_result_of_imp<int (F::*(FD const volatile&&))() const volatile&&, int>();

        test_result_of_imp<int (F::*(FD))()&&, int>();
        test_result_of_imp<int (F::*(FD))() const&&, int>();
        test_result_of_imp<int (F::*(FD))() volatile&&, int>();
        test_result_of_imp<int (F::*(FD))() const volatile&&, int>();
        test_result_of_imp<int (F::*(FD const))() const&&, int>();
        test_result_of_imp<int (F::*(FD const))() const volatile&&, int>();
        test_result_of_imp<int (F::*(FD volatile))() volatile&&, int>();
        test_result_of_imp<int (F::*(FD volatile))() const volatile&&, int>();
        test_result_of_imp<int (F::*(FD const volatile))() const volatile&&, int>();
    }
    {
        test_result_of_imp<int (F::*(s::reference_wrapper<F>))(), int>();
        test_result_of_imp<int (F::*(s::reference_wrapper<const F>))() const, int>();
    }
    test_result_of_imp<decltype (&wat::foo)(wat), void>();

    return true;
}

class KernelTest;

int
main()
{
    cl::sycl::queue deviceQueue;
    cl::sycl::cl_bool ret = false;
    cl::sycl::range<1> numOfItems{1};
    {
        cl::sycl::buffer<cl::sycl::cl_bool, 1> buffer1(&ret, numOfItems);
        deviceQueue.submit([&](cl::sycl::handler& cgh) {
            auto ret_access = buffer1.get_access<sycl_write>(cgh);
            cgh.single_task<class KernelTest>([=]() { ret_access[0] = kernel_test(); });
        });
    }

    if (ret)
    {
        std::cout << "Pass" << std::endl;
    }
    else
    {
        std::cout << "Fail" << std::endl;
    }

    return 0;
}
