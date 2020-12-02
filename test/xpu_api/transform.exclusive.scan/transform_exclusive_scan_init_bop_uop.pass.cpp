// -*- C++ -*-
//===-- transform_exclusive_scan_init_bop_uop.pass.cpp
//--------------------------------------------===//
//
// Copyright (C) 2020 Intel Corporation
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

#include <CL/sycl.hpp>
#include <iostream>
#include <oneapi/dpl/functional>
#include <oneapi/dpl/iterator>
#include <oneapi/dpl/numeric>

#include "test_iterators.h"

constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;

template <typename _T1, typename _T2> void ASSERT_EQUAL(_T1 &&X, _T2 &&Y) {
  if (X != Y)
    std::cout << "CHECK CORRECTNESS (STL WITH SYCL): fail (" << X << "," << Y
              << ")" << std::endl;
}

template <class T> class KernelTest;

struct add_one {
  template <typename T> constexpr auto operator()(T x) const noexcept {
    return static_cast<T>(x + 1);
  }
};

template <class Iter> void test() {
  cl::sycl::queue deviceQueue;
  int input[5] = {1, 3, 5, 7, 9};
  int output1[5] = {};
  int output2[5] = {};
  int output3[5] = {};
  int output4[5] = {};
  cl::sycl::range<1> numOfItems1{5};
  {
    cl::sycl::buffer<int, 1> buffer1(input, numOfItems1);
    cl::sycl::buffer<int, 1> buffer2(output1, numOfItems1);
    cl::sycl::buffer<int, 1> buffer3(output2, numOfItems1);
    cl::sycl::buffer<int, 1> buffer4(output3, numOfItems1);
    cl::sycl::buffer<int, 1> buffer5(output4, numOfItems1);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto in = buffer1.get_access<sycl_read>(cgh);
      auto out1 = buffer2.template get_access<sycl_write>(cgh);
      auto out2 = buffer3.template get_access<sycl_write>(cgh);
      auto out3 = buffer4.template get_access<sycl_write>(cgh);
      auto out4 = buffer5.template get_access<sycl_write>(cgh);
      cgh.single_task<KernelTest<Iter>>([=]() {
        oneapi::dpl::transform_exclusive_scan(Iter(&in[0]), Iter(&in[0] + 5),
                                              &out1[0], 0,
                                              oneapi::dpl::plus<>(), add_one{});
        oneapi::dpl::transform_exclusive_scan(
            Iter(&in[0]), Iter(&in[0] + 5), &out2[0], 0,
            oneapi::dpl::multiplies<>(), add_one{});
        oneapi::dpl::transform_exclusive_scan(
            Iter(&in[0]), Iter(&in[0] + 5), &out3[0], 0, oneapi::dpl::plus<>(),
            oneapi::dpl::negate<>());
        oneapi::dpl::transform_exclusive_scan(
            Iter(&in[0]), Iter(&in[0] + 5), &out4[0], 0,
            oneapi::dpl::multiplies<>(), oneapi::dpl::negate<>());
      });
    });
  }
  const int ref1[5] = {0, 2, 6, 12, 20};
  const int ref2[5] = {0, 0, 0, 0, 0};
  const int ref3[5] = {0, -1, -4, -9, -16};
  const int ref4[5] = {0, 0, 0, 0, 0};
  for (int i = 0; i < 5; ++i) {
    ASSERT_EQUAL(ref1[i], output1[i]);
    ASSERT_EQUAL(ref2[i], output2[i]);
    ASSERT_EQUAL(ref3[i], output3[i]);
    ASSERT_EQUAL(ref4[i], output4[i]);
  }
}

int main() {
#if (_MSC_VER >= 1912 && _MSVC_LANG >= 201703L) ||                             \
    (_GLIBCXX_RELEASE >= 9 && __GLIBCXX__ >= 20190503 &&                       \
     __cplusplus >= 201703L)
  test<input_iterator<const int *>>();
  test<forward_iterator<const int *>>();
  test<bidirectional_iterator<const int *>>();
  test<random_access_iterator<const int *>>();
  test<const int *>();
  test<int *>();
  std::cout << "done" << std::endl;
#else
  std::cout << "skipping test" << std::endl;
#endif
  return 0;
}
