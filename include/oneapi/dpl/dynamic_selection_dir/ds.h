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

#ifndef _ONEDPL_DS_H
#define _ONEDPL_DS_H
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

//#if _DS_BACKEND_SYCL
  using static_policy = policy<static_policy_impl<default_scheduler_t>>;

  //#if 0
  using round_robin_policy = policy<round_robin_policy_impl<default_scheduler_t>>;
  /*using dynamic_load_policy = policy<dynamic_load_policy_impl<default_scheduler_t>>;
  template<typename... KeyArgs> using auto_tune_policy = policy<auto_tune_policy_impl<default_scheduler_t, KeyArgs...>>;
    using static_per_task_policy_t = policy<static_per_task<default_scheduler_t>>;
  #endif
  inline static_policy default_policy;
#endif
*/
  template<typename S> using static_policy_t = policy<static_policy_impl<S>>;
  template<typename S> using round_robin_policy_t = policy<round_robin_policy_impl<S>>;
#if 0
  template<typename S> using round_robin_policy_t = policy<round_robin_policy_impl<S>>;
  template<typename S> using dynamic_load_policy_t = policy<dynamic_load_policy_impl<S>>;
  template<typename... Args> using auto_tune_policy_t = policy<auto_tune_policy_impl<Args...>>;
  template<typename S> using static_per_task_policy_t = policy<static_per_task<S>>;
#endif
} //namespace experimental
} //namespace dpl
} //namespace oneapi

#endif /* ONEDPL_DS_H */
