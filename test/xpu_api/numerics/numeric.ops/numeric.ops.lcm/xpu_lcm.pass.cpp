// -*- C++ -*-
//===-- xpu_lcm.pass.cpp --------------------------------------------===//
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

#include <iostream>

#include <CL/sycl.hpp>
#include <oneapi/dpl/numeric>
#include <oneapi/dpl/type_traits>

constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;

using oneapi::dpl::is_same_v;
using oneapi::dpl::lcm;

template <typename Input1, typename Input2, typename Output>
bool
test0(int in1, int in2, int out)
{
    auto value1 = static_cast<Input1>(in1);
    auto value2 = static_cast<Input2>(in2);
    static_assert(is_same_v<Output, decltype(lcm(value1, value2))>, "");
    static_assert(is_same_v<Output, decltype(lcm(value2, value1))>, "");
    return static_cast<Output>(out) == lcm(value1, value2);
}

template <typename KernelTest, typename Input1, typename Input2 = Input1>
void
do_test()
{
    cl::sycl::queue deviceQueue;
    bool res = true;
    cl::sycl::range<1> numOfItems1{1};

    {
        cl::sycl::buffer<bool, 1> buffer1(&res, numOfItems1);
        deviceQueue.submit([&](cl::sycl::handler& cgh) {
            auto out = buffer1.get_access<sycl_write>(cgh);
            cgh.single_task<KernelTest>([=]() {
                constexpr struct
                {
                    int x;
                    int y;
                    int expect;
                } Cases[] = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0},   {1, 1, 1},
                             {2, 3, 6}, {2, 4, 4}, {3, 17, 51}, {36, 18, 36}};
                using S1 = oneapi::dpl::make_signed_t<Input1>;
                using S2 = oneapi::dpl::make_signed_t<Input2>;
                using U1 = oneapi::dpl::make_unsigned_t<Input1>;
                using U2 = oneapi::dpl::make_unsigned_t<Input2>;
                for (auto TC : Cases)
                {
                    { // Test with two signed types
                        using Output = oneapi::dpl::common_type_t<S1, S2>;
                        out[0] &= test0<S1, S2, Output>(TC.x, TC.y, TC.expect);
                        out[0] &= test0<S1, S2, Output>(-TC.x, TC.y, TC.expect);
                        out[0] &= test0<S1, S2, Output>(TC.x, -TC.y, TC.expect);
                        out[0] &= test0<S1, S2, Output>(-TC.x, -TC.y, TC.expect);
                        out[0] &= test0<S2, S1, Output>(TC.x, TC.y, TC.expect);
                        out[0] &= test0<S2, S1, Output>(-TC.x, TC.y, TC.expect);
                        out[0] &= test0<S2, S1, Output>(TC.x, -TC.y, TC.expect);
                        out[0] &= test0<S2, S1, Output>(-TC.x, -TC.y, TC.expect);
                    }

                    { // test with two unsigned types
                        using Output = oneapi::dpl::common_type_t<U1, U2>;
                        out[0] &= test0<U1, U2, Output>(TC.x, TC.y, TC.expect);
                        out[0] &= test0<U2, U1, Output>(TC.x, TC.y, TC.expect);
                    }
                    { // Test with mixed signs
                        using Output = oneapi::dpl::common_type_t<S1, U2>;
                        out[0] &= test0<S1, U2, Output>(TC.x, TC.y, TC.expect);
                        out[0] &= test0<U2, S1, Output>(TC.x, TC.y, TC.expect);
                        out[0] &= test0<S1, U2, Output>(-TC.x, TC.y, TC.expect);
                        out[0] &= test0<U2, S1, Output>(TC.x, -TC.y, TC.expect);
                    }
                    { // Test with mixed signs
                        using Output = oneapi::dpl::common_type_t<S2, U1>;
                        out[0] &= test0<S2, U1, Output>(TC.x, TC.y, TC.expect);
                        out[0] &= test0<U1, S2, Output>(TC.x, TC.y, TC.expect);
                        out[0] &= test0<S2, U1, Output>(-TC.x, TC.y, TC.expect);
                        out[0] &= test0<U1, S2, Output>(TC.x, -TC.y, TC.expect);
                    }
                }
                { //  LWG#2837
                    auto res1 = oneapi::dpl::lcm(static_cast<std::int64_t>(1234), INT32_MIN);
                    out[0] &= (res1 == 1324997410816LL);
                }
            });
        });
    }

    if (res)
        std::cout << "pass" << std::endl;
    else
        std::cout << "fail" << std::endl;
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

int
main()
{
    do_test<KernelTest1, signed char>();
    do_test<KernelTest2, short>();
    do_test<KernelTest3, int>();
    do_test<KernelTest4, long>();
    do_test<KernelTest5, long long>();
    do_test<KernelTest6, std::int8_t>();
    do_test<KernelTest7, std::int16_t>();
    do_test<KernelTest8, std::int32_t>();
    do_test<KernelTest9, std::int64_t>();
    do_test<KernelTest10, signed char, int>();
    do_test<KernelTest11, int, signed char>();
    do_test<KernelTest12, short, int>();
    do_test<KernelTest13, int, short>();
    do_test<KernelTest14, int, long>();
    do_test<KernelTest15, long, int>();
    do_test<KernelTest16, int, long long>();
    do_test<KernelTest17, long long, int>();

    std::cout << "done" << std::endl;

    return 0;
}
