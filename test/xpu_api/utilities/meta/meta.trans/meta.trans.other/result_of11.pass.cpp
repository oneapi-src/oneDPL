// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This file incorporates work covered by the following copyright and permission
// notice:
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

//
// <functional>
//
// result_of<Fn(ArgTypes...)>

#include "support/test_config.h"

#include <oneapi/dpl/type_traits>
#include <oneapi/dpl/utility>
#include <oneapi/dpl/functional>

#include "support/test_macros.h"
#include "support/utils.h"

#if TEST_DPCPP_BACKEND_PRESENT
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

template <typename T, typename U>
struct test_invoke_result;

template <typename Fn, typename... Args, typename Ret>
struct test_invoke_result<Fn(Args...), Ret>
{
    static void
    call()
    {
        static_assert(dpl::is_invocable<Fn, Args...>::value);
        static_assert(dpl::is_invocable_r<Ret, Fn, Args...>::value);
        ASSERT_SAME_TYPE(Ret, typename dpl::invoke_result<Fn, Args...>::type);
        ASSERT_SAME_TYPE(Ret, dpl::invoke_result_t<Fn, Args...>);
    }
};

template <class T, class U>
void
test_result_of_imp()
{
    ASSERT_SAME_TYPE(U, typename dpl::result_of<T>::type);
    ASSERT_SAME_TYPE(U, dpl::result_of_t<T>);
    test_invoke_result<T, U>::call();
}

bool
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

        test_result_of_imp<PMD(dpl::reference_wrapper<F>), char&>();
        test_result_of_imp<PMD(dpl::reference_wrapper<F const>), const char&>();
        test_result_of_imp<PMD(dpl::reference_wrapper<FD>), char&>();
        test_result_of_imp<PMD(dpl::reference_wrapper<FD const>), const char&>();
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
        test_result_of_imp<int (F::*(dpl::reference_wrapper<F>))(), int>();
        test_result_of_imp<int (F::*(dpl::reference_wrapper<const F>))() const, int>();
    }
    test_result_of_imp<decltype (&wat::foo)(wat), void>();

    return true;
}

class KernelTest;

#endif // TEST_DPCPP_BACKEND_PRESENT

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    bool ret = false;
    sycl::range<1> numOfItems{1};
    {
        sycl::buffer<bool, 1> buffer1(&ret, numOfItems);
        deviceQueue.submit([&](sycl::handler& cgh) {
            auto ret_access = buffer1.get_access<sycl::access::mode::write>(cgh);
            cgh.single_task<class KernelTest>([=]() { ret_access[0] = kernel_test(); });
        });
    }

    EXPECT_TRUE(ret, "Wrong result of work with dpl::is_invocable");
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
