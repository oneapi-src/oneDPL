// -*- C++ -*-
//===----------------------------------------------------------------------===//
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

// oneDPL headers should be included before standard headers
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/numeric>

#include <iostream>
#include <random>
#include <vector>

double random_number_generator() {
  // usage of thread local random engines allows running the generator in
  // concurrent mode
  thread_local static std::default_random_engine rd;
  std::uniform_real_distribution<double> dist(0, 1);
  return dist(rd);
}

int main(int argc, char* argv[]) {
  const size_t size = 10000000;

  std::vector<double> v1(size), v2(size);

  // initialize vectors with random numbers
  // Note: we use par policy due to functor's difficulty for vectorization
  std::generate(oneapi::dpl::execution::par, v1.begin(), v1.end(),
                random_number_generator);
  std::generate(oneapi::dpl::execution::par, v2.begin(), v2.end(),
                random_number_generator);

  // the dot product calculation
  double res = std::transform_reduce(oneapi::dpl::execution::par_unseq,
                                     v1.cbegin(), v1.cend(), v2.cbegin(), .0);

  std::cout << "success" << std::endl;
  std::cout << "The dot product is: " << res << std::endl;

  return 0;
}
