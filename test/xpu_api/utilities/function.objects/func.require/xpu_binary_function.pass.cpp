//===-- xpu_binary_function.pass.cpp --------------------------------------------===//
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

#include <oneapi/dpl/functional>
#include <oneapi/dpl/type_traits>

#include "support/utils.h"

#include <cassert>
#include <iostream>
#include <CL/sycl.hpp>

constexpr sycl::access::mode sycl_read = sycl::access::mode::read;
constexpr sycl::access::mode sycl_write = sycl::access::mode::write;

#if __cplusplus >= 201703L
struct IsSame : public oneapi::dpl::binary_function<int, int, bool>
{
    bool
    operator()(int a, int b)
    {
        return (a == b);
    }
};

class KernelBinaryFunctionTest;

void
kernel_test(sycl::queue& deviceQueue)
{
    sycl::cl_bool ret = false;
    {
        sycl::range<1> numOfItems{1};
        sycl::buffer<sycl::cl_bool, 1> buffer1(&ret, numOfItems);
        deviceQueue.submit([&](sycl::handler& cgh) {
            auto ret_access = buffer1.get_access<sycl_write>(cgh);
            cgh.single_task<class KernelUnaryFunctionTest>([=]() {
                IsSame IsSame_Obj;
                IsSame::first_argument_type input1;
                IsSame::second_argument_type input2;
                IsSame::result_type result;

                result = !IsSame_Obj(53, 54);

                ret_access[0] = result;

                result = IsSame_Obj(42, 42);

                ret_access[0] &= result;

                result = IsSame_Obj(1001, 1001);

                ret_access[0] &= result;

                result = !IsSame_Obj(1, 0);

                ret_access[0] &= result;
            });
        });
    }
    assert(ret);
}
#endif

int
main()
{
#if __cplusplus >= 201703L
    sycl::queue deviceQueue;
    kernel_test(deviceQueue);
    std::cout << "done" << std::endl;
#else
    std::cout << TestUtils::done(0) << ::std::endl;
#endif
    return 0;
}
