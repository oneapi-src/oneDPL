//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <CL/sycl.hpp>
#include <iostream>

#include <dpstd/iterator>
#include <dpstd/algorithm>
#include <dpstd/execution>

using namespace cl::sycl;

auto AsyncHandler = [](exception_list ex_list) {
  for (auto& ex : ex_list) {
    try {
      std::rethrow_exception(ex);
    } catch (exception& ex) {
      std::cerr << ex.what() << std::endl;
      std::exit(1);
    }
  }
};

int main() {
  const int n = 10000;
  buffer<int> keys_buf{n};  // buffer with keys
  buffer<int> vals_buf{n};  // buffer with values

  // create objects to iterate over buffers
  auto keys_begin = dpstd::begin(keys_buf);
  auto vals_begin = dpstd::begin(vals_buf);

  auto counting_begin = dpstd::counting_iterator<int>{0};
  // use default policy for algorithms execution
  auto policy = dpstd::execution::make_device_policy(
      queue(default_selector{}, AsyncHandler));

  // 1. Initialization of buffers
  // let keys_buf contain {n, n, n-2, n-2, ..., 4, 4, 2, 2}
  std::transform(policy, counting_begin, counting_begin + n, keys_begin,
                 [n](int i) { return n - (i / 2) * 2; });
  // fill vals_buf with the analogue of std::iota using counting_iterator
  std::copy(policy, counting_begin, counting_begin + n, vals_begin);

  // 2. Sorting
  auto zipped_begin = dpstd::make_zip_iterator(keys_begin, vals_begin);
  // stable sort by keys
  std::stable_sort(
      policy, zipped_begin, zipped_begin + n,
      // Generic lambda is needed because type of lhs and rhs is unspecified.
      [](auto lhs, auto rhs) {
        using std::get;
        return get<0>(lhs) < get<0>(rhs);
      });

  // 3.Checking results
  auto host_keys = keys_buf.get_access<access::mode::read>();
  auto host_vals = vals_buf.get_access<access::mode::read>();

  // expected output:
  // keys: {2, 2, 4, 4, ..., n - 2, n - 2, n, n}
  // vals: {n - 2, n - 1, n - 4, n - 3, ..., 2, 3, 0, 1}
  for (int i = 0; i < n; ++i) {
    if (host_keys[i] != (i / 2) * 2 &&
        host_vals[i] != n - (i / 2) * 2 - (i % 2 == 0 ? 2 : 1)) {
      std::cout << "fail: i = " << i << ", host_keys[i] = " << host_keys[i]
                << ", host_vals[i] = " << host_vals[i] << "\n";
      return 1;
    }
  }
  std::cout << "success. Run on "
            << policy.queue().get_device().get_info<info::device::name>()
            << "\n";
  return 0;
}
