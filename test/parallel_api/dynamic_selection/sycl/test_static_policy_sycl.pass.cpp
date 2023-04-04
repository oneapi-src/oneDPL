/*
    Copyright 2021 Intel Corporation.  All Rights Reserved.

    The source code contained or described herein and all documents related
    to the source code ("Material") are owned by Intel Corporation or its
    suppliers or licensors.  Title to the Material remains with Intel
    Corporation or its suppliers and licensors.  The Material is protected
    by worldwide copyright laws and treaty provisions.  No part of the
    Material may be used, copied, reproduced, modified, published, uploaded,
    posted, transmitted, distributed, or disclosed in any way without
    Intel's prior express written permission.

    No license under any patent, copyright, trade secret or other
    intellectual property right is granted to or conferred upon you by
    disclosure or delivery of the Materials, either expressly, by
    implication, inducement, estoppel or otherwise.  Any license under such
    intellectual property rights must be express and approved by Intel in
    writing.
*/

#include <iostream>
#include "oneapi/dpl/dynamic_selection"
#include "support/sycl_sanity.h"
#include "support/test_ds_utils.h"

int main() {
  using policy_t = oneapi::dpl::experimental::static_policy;
  std::vector<sycl::queue> u; 
  build_universe(u);
  std::cout << "UNIVERSE SIZE " << u.size() << std::endl;
  if (u.empty()) {
    std::cout << "PASS\n";
    return 0;
  }
  sycl::queue test_resource = u[0];
  auto f = [test_resource](int i) { return test_resource; };

  if (test_cout<policy_t>()
      || test_properties<policy_t>(u, test_resource)
      || test_invoke<policy_t>(u, f)
      || test_invoke_async_and_wait_on_policy<policy_t>(u, f)
      || test_invoke_async_and_wait_on_sync<policy_t>(u, f)
      || test_invoke_async_and_get_wait_list<policy_t>(u, f)
      || test_invoke_async_and_get_wait_list_single_element<policy_t>(u, f)
      || test_invoke_async_and_get_wait_list_empty<policy_t>(u, f)
      || test_select<policy_t>(u, f)
      || test_select_and_wait_on_policy<policy_t>(u, f)
      || test_select_and_wait_on_sync<policy_t>(u, f)
      || test_select_invoke<policy_t>(u, f)
     ) {
    std::cout << "FAIL\n";
    return 1;
  } else {
    std::cout << "PASS\n";
    return 0;
  }
}


