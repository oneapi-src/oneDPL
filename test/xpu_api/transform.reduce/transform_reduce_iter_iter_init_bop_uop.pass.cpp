// -*- C++ -*-
//===-- transform_reduce_iter_iter_init_bop_uop.pass.cpp
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
#include <oneapi/dpl/type_traits>
#include <oneapi/dpl/utility>

#include "test_iterators.h"

constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;

template <typename _T1, typename _T2> void ASSERT_EQUAL(_T1 &&X, _T2 &&Y) {
  if (X != Y)
    std::cout << "CHECK CORRECTNESS (STL WITH SYCL): fail (" << X << "," << Y
              << ")" << std::endl;
}

struct identity {
  template <class T> constexpr decltype(auto) operator()(T &&x) const {
    return oneapi::dpl::forward<T>(x);
  }
};

struct twice {
  template <class T> constexpr auto operator()(const T &x) const {
    return 2 * x;
  }
};

template <class Iter1> void test() {
  cl::sycl::queue deviceQueue;
  int input[6] = {1, 2, 3, 4, 5, 6};
  int output[8] = {};
  cl::sycl::range<1> numOfItems1{6};
  cl::sycl::range<1> numOfItems2{8};

  {
    cl::sycl::buffer<int, 1> buffer1(input, numOfItems1);
    cl::sycl::buffer<int, 1> buffer2(output, numOfItems2);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto in = buffer1.get_access<sycl_read>(cgh);
      auto out = buffer2.get_access<sycl_write>(cgh);
      cgh.single_task<Iter1>([=]() {
        out[0] = oneapi::dpl::transform_reduce(
            Iter1(&in[0]), Iter1(&in[0]), 0, oneapi::dpl::plus<>(), identity());
        out[1] = oneapi::dpl::transform_reduce(Iter1(&in[0]), Iter1(&in[0]), 1,
                                               oneapi::dpl::multiplies<>(),
                                               identity());
        out[2] = oneapi::dpl::transform_reduce(Iter1(&in[0]), Iter1(&in[0] + 1),
                                               0, oneapi::dpl::multiplies<>(),
                                               identity());
        out[3] =
            oneapi::dpl::transform_reduce(Iter1(&in[0]), Iter1(&in[0] + 1), 2,
                                          oneapi::dpl::plus<>(), identity());
        out[4] = oneapi::dpl::transform_reduce(Iter1(&in[0]), Iter1(&in[0] + 6),
                                               4, oneapi::dpl::multiplies<>(),
                                               identity());
        out[5] =
            oneapi::dpl::transform_reduce(Iter1(&in[0]), Iter1(&in[0] + 6), 4,
                                          oneapi::dpl::plus<>(), identity());
        out[6] =
            oneapi::dpl::transform_reduce(Iter1(&in[0]), Iter1(&in[0] + 2), 0,
                                          oneapi::dpl::plus<>(), twice());
        out[7] =
            oneapi::dpl::transform_reduce(Iter1(&in[0]), Iter1(&in[0] + 6), 4,
                                          oneapi::dpl::plus<>(), twice());
      });
    });
  }
  int ref[8] = {0, 1, 0, 3, 2880, 25, 6, 46};
  for (int i = 0; i < 8; ++i) {

    ASSERT_EQUAL(ref[i], output[i]);
  }
}

template <typename T, typename Init, typename KernelName>
void test_return_type() {
  cl::sycl::queue deviceQueue;
  {
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      cgh.single_task<KernelName>([=]() {
        T *p = nullptr;
        static_assert(
            oneapi::dpl::is_same_v<
                Init, decltype(oneapi::dpl::transform_reduce(
                          p, p, Init{}, oneapi::dpl::plus<>(), identity()))>);
      });
    });
  }
}

class KernelTest1;
class KernelTest2;
class KernelTest3;
class KernelTest4;
class KernelTest5;

int main() {
#if (_MSC_VER >= 1912 && _MSVC_LANG >= 201703L) ||                             \
    (_GLIBCXX_RELEASE >= 9 && __GLIBCXX__ >= 20190503 &&                       \
     __cplusplus >= 201703L)
  test_return_type<char, int, KernelTest1>();
  test_return_type<int, int, KernelTest2>();
  test_return_type<int, unsigned long, KernelTest3>();
  test_return_type<float, int, KernelTest4>();
  test_return_type<short, float, KernelTest5>();

  //  All the iterator categories
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
