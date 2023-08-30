// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//


#include <iostream>
#include "oneapi/dpl/dynamic_selection"
#include "support/test_dynamic_selection_utils.h"
#include "support/sycl_sanity.h"

template<bool selection_option, typename Policy, typename UniverseContainer, typename ResourceFunction>
int tests_with_offset(UniverseContainer u, ResourceFunction&& f, int universe_size) {
    bool cond = false;
    for(int offset=0;offset<universe_size;offset++){
        cond = cond
               || test_submit_and_wait_on_event<selection_option, Policy>(u, f, offset)
               || test_submit_and_wait<selection_option, Policy>(u, f, offset)
               || test_submit_and_wait_on_group<selection_option, Policy>(u, f, offset);
        if(cond==1) return 1;
    }
    return cond;
}

int main() {
  using policy_t = oneapi::dpl::experimental::round_robin_policy<oneapi::dpl::experimental::sycl_backend>;
  std::vector<sycl::queue> u;
  build_universe(u);
  if (u.empty()) {
    std::cout << "PASS\n";
    return 0;
  }

  auto n = u.size();
  std::cout << "UNIVERSE SIZE " << n << std::endl;
  if(n==0) return 0;

  auto f = [u, n](int i, int offset=0) { return u[(i+offset-1)%n]; };


  constexpr bool just_call_submit = false;
  constexpr bool call_select_before_submit = true;
  if ( test_initialization<policy_t, sycl::queue>(u)
       || test_select<policy_t, decltype(u), decltype(f)&, false>(u, f)
       || tests_with_offset<just_call_submit, policy_t>(u, f, n)
       || tests_with_offset<call_select_before_submit, policy_t>(u, f, n)
     ) {
    std::cout << "FAIL\n";
    return 1;
  } else {
    std::cout << "PASS\n";
    return 0;
  }
}


