//===-- xpu_unary_function.pass.cpp --------------------------------------------===//
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

#include <cassert>
#include <iostream>
#include <CL/sycl.hpp>

constexpr sycl::access::mode sycl_read = sycl::access::mode::read;
constexpr sycl::access::mode sycl_write = sycl::access::mode::write;

struct IsOdd : public oneapi::dpl::unary_function<int, bool>
{
    bool
    operator()(int number)
    {
        return (number % 2 != 0);
    }
};

class KernelUnaryFunctionTest;

void
kernel_test(sycl::queue& deviceQueue)
{
    sycl::cl_bool ret = false;
    {
        sycl::range<1> numOfItems{1};
        sycl::buffer<sycl::cl_bool, 1> buffer1(&ret, numOfItems);
        int dev_num = 5;

        sycl::buffer<sycl::cl_int, 1> dev_num_buffer(&dev_num, numOfItems);
        deviceQueue.submit([&](sycl::handler& cgh) {
            auto ret_access = buffer1.get_access<sycl_write>(cgh);
            auto num_acc = dev_num_buffer.get_access<sycl_read>(cgh);
            cgh.single_task<class KernelUnaryFunctionTest>([=]() {
                IsOdd IsOdd_Obj;
                IsOdd::argument_type input;
                IsOdd::result_type result;

                result = IsOdd_Obj(53);

                ret_access[0] = result;

                result = IsOdd_Obj(42);

                ret_access[0] &= !result;

                result = IsOdd_Obj(1001);

                ret_access[0] &= result;

                result = IsOdd_Obj(num_acc[0]);

                ret_access[0] &= result;
            });
        });
    }
    assert(ret);
}

int
main()
{

    sycl::queue deviceQueue;
    kernel_test(deviceQueue);
    std::cout << "done" << std::endl;
    return 0;
}
