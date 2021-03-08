// -*- C++ -*-
//===-- xpu_gcd3.pass.cpp --------------------------------------------===//
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

using oneapi::dpl::common_type_t;
using oneapi::dpl::gcd;
using oneapi::dpl::is_same_v;

class KernelTest;

template <typename P, typename Q>
bool
check(P p, Q q, unsigned long long r)
{
    using R = common_type_t<P, Q>;
    static_assert(is_same_v<decltype(gcd(p, q)), R>);
    static_assert(is_same_v<decltype(gcd(q, p)), R>);
    R r1 = gcd(p, q);
    bool res = true;
    // Check non-negative, so conversion to unsigned long doesn't alter value.
    res &= r1 >= 0;
    // Check for expected result
    res &= (unsigned long long)r1 == r;
    // Check reversing arguments doesn't change result
    res &= gcd(q, p) == r1;

    P pabs = p < 0 ? -p : p;
    res &= gcd(p, p) == pabs;
    res &= gcd(0, p) == pabs;
    res &= gcd(p, 0) == pabs;
    res &= gcd(1, p) == 1;
    res &= gcd(p, 1) == 1;
    Q qabs = q < 0 ? -q : q;
    res &= gcd(q, q) == qabs;
    res &= gcd(0, q) == qabs;
    res &= gcd(q, 0) == qabs;
    res &= gcd(1, q) == 1;
    res &= gcd(q, 1) == 1;
    res &= gcd(r, r) == r;
    res &= gcd(0, r) == r;
    res &= gcd(r, 0) == r;
    res &= gcd(1, r) == 1;
    res &= gcd(r, 1) == 1;

    return res;
}

void
test()
{
    cl::sycl::queue deviceQueue;
    bool res = true;
    cl::sycl::range<1> numOfItems1{1};

    {
        cl::sycl::buffer<bool, 1> buffer1(&res, numOfItems1);
        deviceQueue.submit([&](cl::sycl::handler& cgh) {
            auto out = buffer1.get_access<sycl_write>(cgh);
            cgh.single_task<KernelTest>([=]() {
                constexpr struct testcase
                {
                    unsigned long long p, q, r;
                } testcases[] = {

                    {5, 8, 1},
                    {6, 35, 1},
                    {30, 42, 6},
                    {24, 60, 12},
                    {55, 144, 1},
                    {105, 252, 21},
                    {253, 22121, 11},
                    {1386, 3213, 63},
                    {2028, 2049, 3},
                    {46391, 62527, 2017},
                    {63245986, 39088169, 1},
                    {77160074263, 47687519812, 1},
                    {77160074264, 47687519812, 4},
                };

                for (auto t : testcases)
                {
                    out[0] &= check(t.p, t.q, t.r);

                    if (t.p <= LONG_MAX && t.q <= LONG_MAX)
                    {
                        out[0] &= check((long)t.p, (long)t.p, t.p);
                        out[0] &= check(-(long)t.p, (long)t.p, t.p);
                        out[0] &= check(-(long)t.p, -(long)t.p, t.p);

                        out[0] &= check((long)t.p, t.q, t.r);
                        out[0] &= check(-(long)t.p, t.q, t.r);

                        out[0] &= check(t.p, (long)t.q, t.r);
                        out[0] &= check(t.p, -(long)t.q, t.r);

                        out[0] &= check((long)t.p, (long)t.q, t.r);
                        out[0] &= check((long)t.p, -(long)t.q, t.r);
                        out[0] &= check(-(long)t.p, (long)t.q, t.r);
                        out[0] &= check(-(long)t.p, -(long)t.q, t.r);
                    }

                    if (t.p <= INT_MAX && t.q <= INT_MAX)
                    {
                        out[0] &= check((long)t.p, (int)t.q, t.r);
                        out[0] &= check(-(int)t.p, (long)t.q, t.r);

                        out[0] &= check((int)t.p, (unsigned)t.q, t.r);
                        out[0] &= check(-(int)t.p, (unsigned)t.q, t.r);

                        out[0] &= check(-(int)t.p, -(int)t.q, t.r);
                        out[0] &= check(-(int)t.p, -(long)t.q, t.r);
                    }

                    if (t.p <= SHRT_MAX && t.q <= SHRT_MAX)
                    {
                        out[0] &= check((long)t.p, (short)t.q, t.r);
                        out[0] &= check(-(short)t.p, (long)t.q, t.r);

                        out[0] &= check((short)t.p, (unsigned short)t.q, t.r);
                        out[0] &= check(-(short)t.p, (unsigned short)t.q, t.r);

                        out[0] &= check(-(short)t.p, -(short)t.q, t.r);
                        out[0] &= check(-(short)t.p, -(long)t.q, t.r);
                    }
                }
            });
        });
    }

    if (res)
        std::cout << "pass\n";
    else
        std::cout << "fail\n";
}

int
main()
{
    test();
    std::cout << "done\n";

    return 0;
}
