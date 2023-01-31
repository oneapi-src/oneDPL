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

#pragma once

// Check the user-defined macro for parallel policies
// define _DS_BACKEND_SYCL 1 when we compile with the Compiler that supports SYCL
#if !defined(_DS_BACKEND_SYCL)
#    if (defined(CL_SYCL_LANGUAGE_VERSION) || defined(SYCL_LANGUAGE_VERSION))
#        define _DS_BACKEND_SYCL 1
#    else
#        define _DS_BACKEND_SYCL 0
#    endif // CL_SYCL_LANGUAGE_VERSION
#endif

#include "oneapi/dpl/dynamic_selection/ds_policy.h"
#include "oneapi/dpl/dynamic_selection/ds_scoring_policy.h"
#if _DS_BACKEND_SYCL
 // defines default_scheduler_t
 #include "oneapi/dpl/dynamic_selection/ds_scheduler.h"
#endif

#include "oneapi/dpl/dynamic_selection/ds_properties.h"
#include "oneapi/dpl/dynamic_selection/ds_algorithms.h"
namespace oneapi {
namespace dpl {
namespace experimental {

#if _DS_BACKEND_SYCL
  using static_policy = policy<static_policy_impl<default_scheduler_t>>;

  #if 0
  using round_robin_policy = policy<round_robin_policy_impl<default_scheduler_t>>;
  using dynamic_load_policy = policy<dynamic_load_policy_impl<default_scheduler_t>>;
  template<typename... KeyArgs> using auto_tune_policy = policy<auto_tune_policy_impl<default_scheduler_t, KeyArgs...>>;
    using static_per_task_policy_t = policy<static_per_task<default_scheduler_t>>;
  #endif
  inline static_policy default_policy;
#endif

  template<typename S> using static_policy_t = policy<static_policy_impl<S>>;
#if 0
  template<typename S> using round_robin_policy_t = policy<round_robin_policy_impl<S>>;
  template<typename S> using dynamic_load_policy_t = policy<dynamic_load_policy_impl<S>>;
  template<typename... Args> using auto_tune_policy_t = policy<auto_tune_policy_impl<Args...>>;
  template<typename S> using static_per_task_policy_t = policy<static_per_task<S>>;
#endif
} //namespace experimental
} //namespace dpl
} //namespace oneapi

