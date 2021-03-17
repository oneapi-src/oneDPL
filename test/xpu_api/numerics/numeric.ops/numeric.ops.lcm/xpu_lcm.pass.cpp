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

#include <oneapi/dpl/numeric>
#include <oneapi/dpl/type_traits>

#include <iostream>
#include <CL/sycl.hpp>

constexpr sycl::access::mode sycl_read = sycl::access::mode::read;
constexpr sycl::access::mode sycl_write = sycl::access::mode::write;

using oneapi::dpl::is_same;
using oneapi::dpl::lcm;

template <typename Input1, typename Input2, typename Output>
bool
test0(int in1, int in2, int out)
{
    auto value1 = static_cast<Input1>(in1);
    auto value2 = static_cast<Input2>(in2);
    static_assert(is_same<Output, decltype(lcm(value1, value2))>::value, "");
    static_assert(is_same<Output, decltype(lcm(value2, value1))>::value, "");
    return static_cast<Output>(out) == lcm(value1, value2);
}

template <typename KernelTest, typename Input1, typename Input2 = Input1>
void
do_test()
{
    sycl::queue deviceQueue;
    bool res = true;
    sycl::range<1> numOfItems1{1};

    {
        sycl::buffer<bool, 1> buffer1(&res, numOfItems1);
        deviceQueue.submit([&](sycl::handler& cgh) {
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
                {
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

class SignedChar2;
class Short2;
class Int2;
class Long2;
class LongLong2;
class Int8t2;
class Int16t2;
class Int32t2;
class Int64t2;
class SignedCharAndInt;
class IntAndSignedChar;
class ShortAndInt;
class IntAndShort;
class IntAndLong;
class LongAndInt;
class IntAndLongLong;
class LongLongAndInt;

int
main()
{
    do_test<SignedChar2, signed char>();
    do_test<Short2, short>();
    do_test<Int2, int>();
    do_test<Long2, long>();
    do_test<LongLong2, long long>();
    do_test<Int8t2, std::int8_t>();
    do_test<Int16t2, std::int16_t>();
    do_test<Int32t2, std::int32_t>();
    do_test<Int64t2, std::int64_t>();
    do_test<SignedCharAndInt, signed char, int>();
    do_test<IntAndSignedChar, int, signed char>();
    do_test<ShortAndInt, short, int>();
    do_test<IntAndShort, int, short>();
    do_test<IntAndLong, int, long>();
    do_test<LongAndInt, long, int>();
    do_test<IntAndLongLong, int, long long>();
    do_test<LongLongAndInt, long long, int>();

    std::cout << "done" << std::endl;

    return 0;
}
