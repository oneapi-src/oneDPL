// -*- C++ -*-
//===-- adjacent_difference_op.pass.cpp
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

#include <iostream>

#include <CL/sycl.hpp>
#include <oneapi/dpl/functional>
#include <oneapi/dpl/numeric>

#include "test_iterators.h"

template <typename _T1, typename _T2> void ASSERT_EQUAL(_T1 &&X, _T2 &&Y) {
  if (X != Y)
    std::cout << "CHECK CORRECTNESS (STL WITH SYCL): fail (" << X << "," << Y
              << ")" << std::endl;
}

constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;

template <class InIter, class OutIter, class Test> void test() {
  cl::sycl::queue deviceQueue;
  int input[5] = {15, 10, 6, 3, 1};
  int output[5] = {0};
  cl::sycl::range<1> numOfItems1{5};

  {
    cl::sycl::buffer<int, 1> buffer1(input, numOfItems1);
    cl::sycl::buffer<int, 1> buffer2(output, numOfItems1);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto in = buffer1.get_access<sycl_read>(cgh);
      auto out = buffer2.get_access<sycl_write>(cgh);
      cgh.single_task<Test>([=]() {
        OutIter r = oneapi::dpl::adjacent_difference(
            InIter(&in[0]), InIter(&in[0] + 5), OutIter(&out[0]),
            oneapi::dpl::plus<int>());
      });
    });
  }
  int ref[5] = {15, 25, 16, 9, 4};
  for (int i = 0; i < 5; ++i)
    ASSERT_EQUAL(ref[i], output[i]);
}

class Y;

class X {
  int i_;

  X &operator=(const X &);

public:
  explicit X(int i) : i_(i) {}
  X(const X &x) : i_(x.i_) {}
  X &operator=(X &&x) {
    i_ = x.i_;
    x.i_ = -1;
    return *this;
  }

  friend X operator-(const X &x, const X &y) { return X(x.i_ - y.i_); }

  friend class Y;
};

class Y {
  int i_;

  Y &operator=(const Y &);

public:
  explicit Y(int i) : i_(i) {}
  Y(const Y &y) : i_(y.i_) {}
  void operator=(const X &x) { i_ = x.i_; }
};

void kernel_test() {
  cl::sycl::queue deviceQueue;

  {
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      cgh.single_task<class KernelTest>([=]() {
        X x[3] = {X(1), X(2), X(3)};
        Y y[3] = {Y(1), Y(2), Y(3)};
        oneapi::dpl::adjacent_difference(x, x + 3, y, std::minus<X>());
      });
    });
  }
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
class KernelTest18;
class KernelTest19;
class KernelTest20;
class KernelTest21;
class KernelTest22;
class KernelTest23;
class KernelTest24;
class KernelTest25;

int main() {
  test<input_iterator<const int *>, input_iterator<int *>, KernelTest1>();

  test<input_iterator<const int *>, forward_iterator<int *>, KernelTest2>();
  test<input_iterator<const int *>, bidirectional_iterator<int *>,
       KernelTest3>();
  test<input_iterator<const int *>, random_access_iterator<int *>,
       KernelTest4>();
  test<input_iterator<const int *>, int *, KernelTest5>();

  test<forward_iterator<const int *>, input_iterator<int *>, KernelTest6>();
  test<forward_iterator<const int *>, forward_iterator<int *>, KernelTest7>();
  test<forward_iterator<const int *>, bidirectional_iterator<int *>,
       KernelTest8>();
  test<forward_iterator<const int *>, random_access_iterator<int *>,
       KernelTest9>();
  test<forward_iterator<const int *>, int *, KernelTest10>();

  test<bidirectional_iterator<const int *>, input_iterator<int *>,
       KernelTest11>();
  test<bidirectional_iterator<const int *>, forward_iterator<int *>,
       KernelTest12>();
  test<bidirectional_iterator<const int *>, bidirectional_iterator<int *>,
       KernelTest13>();
  test<bidirectional_iterator<const int *>, random_access_iterator<int *>,
       KernelTest14>();
  test<bidirectional_iterator<const int *>, int *, KernelTest15>();

  test<random_access_iterator<const int *>, input_iterator<int *>,
       KernelTest16>();
  test<random_access_iterator<const int *>, forward_iterator<int *>,
       KernelTest17>();
  test<random_access_iterator<const int *>, bidirectional_iterator<int *>,
       KernelTest18>();
  test<random_access_iterator<const int *>, random_access_iterator<int *>,
       KernelTest19>();
  test<random_access_iterator<const int *>, int *, KernelTest20>();

  test<const int *, input_iterator<int *>, KernelTest21>();
  test<const int *, forward_iterator<int *>, KernelTest22>();
  test<const int *, bidirectional_iterator<int *>, KernelTest23>();
  test<const int *, random_access_iterator<int *>, KernelTest24>();
  test<const int *, int *, KernelTest25>();
  kernel_test();
  std::cout << "done" << std::endl;
  return 0;
}
