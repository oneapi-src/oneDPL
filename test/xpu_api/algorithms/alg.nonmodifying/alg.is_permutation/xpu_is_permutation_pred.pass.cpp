//===-- xpu_is_permutation_pred.pass.cpp ----------------------------------===//
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

#include <oneapi/dpl/algorithm>

#include "support/test_iterators.h"

#include <cassert>
#include <CL/sycl.hpp>

constexpr sycl::access::mode sycl_read = sycl::access::mode::read;
constexpr sycl::access::mode sycl_write = sycl::access::mode::write;

using oneapi::dpl::is_permutation;

struct S
{
    S(int i) : i_(i) {}
    bool
    operator==(const S& other) = delete;
    int i_;
};

struct eq
{
    bool
    operator()(const S& a, const S& b)
    {
        return a.i_ == b.i_;
    }
};

void
kernel_test1()
{
    sycl::queue deviceQueue;
    bool ret = true;
    sycl::range<1> item1{1};
    {
        sycl::buffer<bool, 1> buffer1(&ret, item1);
        deviceQueue.submit([&](sycl::handler& cgh) {
            auto ret_acc = buffer1.get_access<sycl_write>(cgh);
            cgh.single_task<class KernelTest1>([=]() {
                {
                    const int ia[] = {0};
                    const int ib[] = {0};
                    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
                    ret_acc[0] = (is_permutation(forward_iterator<const int*>(ia), forward_iterator<const int*>(ia + 0),
                                                 forward_iterator<const int*>(ib), std::equal_to<const int>()) == true);
                    ret_acc[0] &=
                        (is_permutation(forward_iterator<const int*>(ia), forward_iterator<const int*>(ia + sa),
                                        forward_iterator<const int*>(ib), std::equal_to<const int>()) == true);
                }
                {
                    const int ia[] = {0};
                    const int ib[] = {1};
                    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
                    ret_acc[0] &=
                        (is_permutation(forward_iterator<const int*>(ia), forward_iterator<const int*>(ia + sa),
                                        forward_iterator<const int*>(ib), std::equal_to<const int>()) == false);
                }

                {
                    const int ia[] = {0, 0};
                    const int ib[] = {0, 0};
                    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
                    ret_acc[0] &=
                        (is_permutation(forward_iterator<const int*>(ia), forward_iterator<const int*>(ia + sa),
                                        forward_iterator<const int*>(ib), std::equal_to<const int>()) == true);
                }
                {
                    const int ia[] = {0, 0};
                    const int ib[] = {0, 1};
                    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
                    ret_acc[0] &=
                        (is_permutation(forward_iterator<const int*>(ia), forward_iterator<const int*>(ia + sa),
                                        forward_iterator<const int*>(ib), std::equal_to<const int>()) == false);
                }
                {
                    const int ia[] = {0, 0};
                    const int ib[] = {1, 0};
                    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
                    ret_acc[0] &=
                        (is_permutation(forward_iterator<const int*>(ia), forward_iterator<const int*>(ia + sa),
                                        forward_iterator<const int*>(ib), std::equal_to<const int>()) == false);
                }
                {
                    const int ia[] = {0, 0};
                    const int ib[] = {1, 1};
                    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
                    ret_acc[0] &=
                        (is_permutation(forward_iterator<const int*>(ia), forward_iterator<const int*>(ia + sa),
                                        forward_iterator<const int*>(ib), std::equal_to<const int>()) == false);
                }
                {
                    const int ia[] = {0, 1};
                    const int ib[] = {0, 0};
                    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
                    ret_acc[0] &=
                        (is_permutation(forward_iterator<const int*>(ia), forward_iterator<const int*>(ia + sa),
                                        forward_iterator<const int*>(ib), std::equal_to<const int>()) == false);
                }
                {
                    const int ia[] = {0, 1};
                    const int ib[] = {0, 1};
                    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
                    ret_acc[0] &=
                        (is_permutation(forward_iterator<const int*>(ia), forward_iterator<const int*>(ia + sa),
                                        forward_iterator<const int*>(ib), std::equal_to<const int>()) == true);
                }
                {
                    const int ia[] = {0, 1};
                    const int ib[] = {1, 0};
                    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
                    ret_acc[0] &=
                        (is_permutation(forward_iterator<const int*>(ia), forward_iterator<const int*>(ia + sa),
                                        forward_iterator<const int*>(ib), std::equal_to<const int>()) == true);
                }
                {
                    const int ia[] = {0, 1};
                    const int ib[] = {1, 1};
                    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
                    ret_acc[0] &=
                        (is_permutation(forward_iterator<const int*>(ia), forward_iterator<const int*>(ia + sa),
                                        forward_iterator<const int*>(ib), std::equal_to<const int>()) == false);
                }
                {
                    const int ia[] = {1, 0};
                    const int ib[] = {0, 0};
                    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
                    ret_acc[0] &=
                        (is_permutation(forward_iterator<const int*>(ia), forward_iterator<const int*>(ia + sa),
                                        forward_iterator<const int*>(ib), std::equal_to<const int>()) == false);
                }
                {
                    const int ia[] = {1, 0};
                    const int ib[] = {0, 1};
                    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
                    ret_acc[0] &=
                        (is_permutation(forward_iterator<const int*>(ia), forward_iterator<const int*>(ia + sa),
                                        forward_iterator<const int*>(ib), std::equal_to<const int>()) == true);
                }
                {
                    const int ia[] = {1, 0};
                    const int ib[] = {1, 0};
                    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
                    ret_acc[0] &=
                        (is_permutation(forward_iterator<const int*>(ia), forward_iterator<const int*>(ia + sa),
                                        forward_iterator<const int*>(ib), std::equal_to<const int>()) == true);
                }
                {
                    const int ia[] = {1, 0};
                    const int ib[] = {1, 1};
                    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
                    ret_acc[0] &=
                        (is_permutation(forward_iterator<const int*>(ia), forward_iterator<const int*>(ia + sa),
                                        forward_iterator<const int*>(ib), std::equal_to<const int>()) == false);
                }
                {
                    const int ia[] = {1, 1};
                    const int ib[] = {0, 0};
                    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
                    ret_acc[0] &=
                        (is_permutation(forward_iterator<const int*>(ia), forward_iterator<const int*>(ia + sa),
                                        forward_iterator<const int*>(ib), std::equal_to<const int>()) == false);
                }
                {
                    const int ia[] = {1, 1};
                    const int ib[] = {0, 1};
                    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
                    ret_acc[0] &=
                        (is_permutation(forward_iterator<const int*>(ia), forward_iterator<const int*>(ia + sa),
                                        forward_iterator<const int*>(ib), std::equal_to<const int>()) == false);
                }
                {
                    const int ia[] = {1, 1};
                    const int ib[] = {1, 0};
                    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
                    ret_acc[0] &=
                        (is_permutation(forward_iterator<const int*>(ia), forward_iterator<const int*>(ia + sa),
                                        forward_iterator<const int*>(ib), std::equal_to<const int>()) == false);
                }
                {
                    const int ia[] = {1, 1};
                    const int ib[] = {1, 1};
                    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
                    ret_acc[0] &=
                        (is_permutation(forward_iterator<const int*>(ia), forward_iterator<const int*>(ia + sa),
                                        forward_iterator<const int*>(ib), std::equal_to<const int>()) == true);
                }

                {
                    const int ia[] = {0, 0, 0};
                    const int ib[] = {1, 0, 0};
                    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
                    ret_acc[0] &=
                        (is_permutation(forward_iterator<const int*>(ia), forward_iterator<const int*>(ia + sa),
                                        forward_iterator<const int*>(ib), std::equal_to<const int>()) == false);
                }
                {
                    const int ia[] = {0, 0, 0};
                    const int ib[] = {1, 0, 1};
                    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
                    ret_acc[0] &=
                        (is_permutation(forward_iterator<const int*>(ia), forward_iterator<const int*>(ia + sa),
                                        forward_iterator<const int*>(ib), std::equal_to<const int>()) == false);
                }
                {
                    const int ia[] = {0, 0, 0};
                    const int ib[] = {1, 0, 2};
                    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
                    ret_acc[0] &=
                        (is_permutation(forward_iterator<const int*>(ia), forward_iterator<const int*>(ia + sa),
                                        forward_iterator<const int*>(ib), std::equal_to<const int>()) == false);
                }
                {
                    const int ia[] = {0, 0, 0};
                    const int ib[] = {1, 1, 0};
                    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
                    ret_acc[0] &=
                        (is_permutation(forward_iterator<const int*>(ia), forward_iterator<const int*>(ia + sa),
                                        forward_iterator<const int*>(ib), std::equal_to<const int>()) == false);
                }
                {
                    const int ia[] = {0, 0, 0};
                    const int ib[] = {1, 1, 1};
                    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
                    ret_acc[0] &=
                        (is_permutation(forward_iterator<const int*>(ia), forward_iterator<const int*>(ia + sa),
                                        forward_iterator<const int*>(ib), std::equal_to<const int>()) == false);
                }
                {
                    const int ia[] = {0, 0, 0};
                    const int ib[] = {1, 1, 2};
                    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
                    ret_acc[0] &=
                        (is_permutation(forward_iterator<const int*>(ia), forward_iterator<const int*>(ia + sa),
                                        forward_iterator<const int*>(ib), std::equal_to<const int>()) == false);
                }
                {
                    const int ia[] = {0, 0, 0};
                    const int ib[] = {1, 2, 0};
                    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
                    ret_acc[0] &=
                        (is_permutation(forward_iterator<const int*>(ia), forward_iterator<const int*>(ia + sa),
                                        forward_iterator<const int*>(ib), std::equal_to<const int>()) == false);
                }
                {
                    const int ia[] = {0, 0, 0};
                    const int ib[] = {1, 2, 1};
                    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
                    ret_acc[0] &=
                        (is_permutation(forward_iterator<const int*>(ia), forward_iterator<const int*>(ia + sa),
                                        forward_iterator<const int*>(ib), std::equal_to<const int>()) == false);
                }
                {
                    const int ia[] = {0, 0, 0};
                    const int ib[] = {1, 2, 2};
                    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
                    ret_acc[0] &=
                        (is_permutation(forward_iterator<const int*>(ia), forward_iterator<const int*>(ia + sa),
                                        forward_iterator<const int*>(ib), std::equal_to<const int>()) == false);
                }
                {
                    const int ia[] = {0, 0, 1};
                    const int ib[] = {1, 0, 0};
                    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
                    ret_acc[0] &=
                        (is_permutation(forward_iterator<const int*>(ia), forward_iterator<const int*>(ia + sa),
                                        forward_iterator<const int*>(ib), std::equal_to<const int>()) == true);
                }
                {
                    const int ia[] = {0, 0, 1};
                    const int ib[] = {1, 0, 1};
                    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
                    ret_acc[0] &=
                        (is_permutation(forward_iterator<const int*>(ia), forward_iterator<const int*>(ia + sa),
                                        forward_iterator<const int*>(ib), std::equal_to<const int>()) == false);
                }
                {
                    const int ia[] = {0, 1, 2};
                    const int ib[] = {1, 0, 2};
                    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
                    ret_acc[0] &=
                        (is_permutation(forward_iterator<const int*>(ia), forward_iterator<const int*>(ia + sa),
                                        forward_iterator<const int*>(ib), std::equal_to<const int>()) == true);
                }
                {
                    const int ia[] = {0, 1, 2};
                    const int ib[] = {1, 2, 0};
                    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
                    ret_acc[0] &=
                        (is_permutation(forward_iterator<const int*>(ia), forward_iterator<const int*>(ia + sa),
                                        forward_iterator<const int*>(ib), std::equal_to<const int>()) == true);
                }
                {
                    const int ia[] = {0, 1, 2};
                    const int ib[] = {2, 1, 0};
                    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
                    ret_acc[0] &=
                        (is_permutation(forward_iterator<const int*>(ia), forward_iterator<const int*>(ia + sa),
                                        forward_iterator<const int*>(ib), std::equal_to<const int>()) == true);
                }
                {
                    const int ia[] = {0, 1, 2};
                    const int ib[] = {2, 0, 1};
                    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
                    ret_acc[0] &=
                        (is_permutation(forward_iterator<const int*>(ia), forward_iterator<const int*>(ia + sa),
                                        forward_iterator<const int*>(ib), std::equal_to<const int>()) == true);
                }
                {
                    const int ia[] = {0, 0, 1};
                    const int ib[] = {1, 0, 1};
                    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
                    ret_acc[0] &=
                        (is_permutation(forward_iterator<const int*>(ia), forward_iterator<const int*>(ia + sa),
                                        forward_iterator<const int*>(ib), std::equal_to<const int>()) == false);
                }
                {
                    const int ia[] = {0, 0, 1};
                    const int ib[] = {1, 0, 0};
                    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
                    ret_acc[0] &=
                        (is_permutation(forward_iterator<const int*>(ia), forward_iterator<const int*>(ia + sa),
                                        forward_iterator<const int*>(ib), std::equal_to<const int>()) == true);
                }
                {
                    const int ia[] = {0, 1, 2, 3, 0, 5, 6, 2, 4, 4};
                    const int ib[] = {4, 2, 3, 0, 1, 4, 0, 5, 6, 2};
                    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
                    ret_acc[0] &=
                        (is_permutation(forward_iterator<const int*>(ia), forward_iterator<const int*>(ia + sa),
                                        forward_iterator<const int*>(ib), std::equal_to<const int>()) == true);
                }
                {
                    const int ia[] = {0, 1, 2, 3, 0, 5, 6, 2, 4, 4};
                    const int ib[] = {4, 2, 3, 0, 1, 4, 0, 5, 6, 0};
                    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
                    ret_acc[0] &=
                        (is_permutation(forward_iterator<const int*>(ia), forward_iterator<const int*>(ia + sa),
                                        forward_iterator<const int*>(ib), std::equal_to<const int>()) == false);
                }
                {
                    const S a[] = {S(0), S(1)};
                    const S b[] = {S(1), S(0)};
                    const unsigned sa = sizeof(a) / sizeof(a[0]);
                    ret_acc[0] &= (is_permutation(forward_iterator<const S*>(a), forward_iterator<const S*>(a + sa),
                                                  forward_iterator<const S*>(b), eq()));
                }
            });
        });
    }
    assert(ret);
}

int
main()
{
    kernel_test1();
    return 0;
}
