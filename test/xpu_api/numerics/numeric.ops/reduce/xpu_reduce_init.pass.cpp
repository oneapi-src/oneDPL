//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <numeric>
// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: clang-8
// UNSUPPORTED: gcc-9

// Became constexpr in C++20
// template<class InputIterator, class T>
//   T reduce(InputIterator first, InputIterator last, T init);
#include <oneapi/dpl/numeric>
#include <cassert>
#include <CL/sycl.hpp>

#include "support/test_iterators.h"

constexpr sycl::access::mode sycl_read = sycl::access::mode::read;
constexpr sycl::access::mode sycl_write = sycl::access::mode::write;

template <class T>
class KernelTest;

template <typename _T1, typename _T2>
void
ASSERT_EQUAL(_T1&& X, _T2&& Y)
{
    if (X != Y)
        std::cout << "CHECK CORRECTNESS (STL WITH SYCL): fail (" << X << "," << Y << ")" << std::endl;
}

template <class Iter>
void
test(sycl::queue& deviceQueue)
{
    int input[] = {1, 2, 3, 4, 5, 6};
    int output[8];
    int ref[8] = {0, 1, 1, 3, 3, 6, 21, 25};
    sycl::range<1> numOfItems{6};
    {
        sycl::buffer<int, 1> buffer1(input, sycl::range<1>{6});
        sycl::buffer<int, 1> buffer2(output, sycl::range<1>{8});
        deviceQueue.submit(
            [&](sycl::handler& cgh)
            {
                auto in = buffer1.get_access<sycl_read>(cgh);
                auto out = buffer2.get_access<sycl_write>(cgh);
                cgh.single_task<KernelTest<Iter>>(
                    [=]()
                    {
                        out[0] = dpl::reduce(Iter(&in[0]), Iter(&in[0]), 0);
                        out[1] = dpl::reduce(Iter(&in[0]), Iter(&in[0]), 1);
                        out[2] = dpl::reduce(Iter(&in[0]), Iter(&in[0] + 1), 0);
                        out[3] = dpl::reduce(Iter(&in[0]), Iter(&in[0] + 1), 2);
                        out[4] = dpl::reduce(Iter(&in[0]), Iter(&in[0] + 2), 0);
                        out[5] = dpl::reduce(Iter(&in[0]), Iter(&in[0] + 2), 3);
                        out[6] = dpl::reduce(Iter(&in[0]), Iter(&in[0] + 6), 0);
                        out[7] = dpl::reduce(Iter(&in[0]), Iter(&in[0] + 6), 4);
                    });
            });
    }

    for (size_t idx = 0; idx < 8; ++idx)
    {
        ASSERT_EQUAL(ref[idx], output[idx]);
    }
}

void
test(sycl::queue& deviceQueue)
{
    test<input_iterator<const int*>>(deviceQueue);
    test<forward_iterator<const int*>>(deviceQueue);
    test<bidirectional_iterator<const int*>>(deviceQueue);
    test<random_access_iterator<const int*>>(deviceQueue);
    test<const int*>(deviceQueue);
}

int
main(int, char**)
{
    sycl::queue deviceQueue;
    test(deviceQueue);
    return 0;
}
