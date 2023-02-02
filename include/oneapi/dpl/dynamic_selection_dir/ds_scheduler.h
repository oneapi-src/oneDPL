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


#ifndef _ONEDPL_DS_SCHEDULER_DEFS_H
#define _ONEDPL_DS_SCHEDULER_DEFS_H
#pragma once

#include "oneapi/dpl/internal/dynamic_selection_impl/sycl_scheduler.h"
#include <atomic>
#if 0
#include "custom/tbb_numa_scheduler.h"
#endif

namespace oneapi {
namespace dpl {
namespace experimental {
  using default_scheduler_t = oneapi::dpl::experimental::sycl_scheduler;
} // namespace experimental
} // namespace dpl
} //namespace oneapi


#endif /*_ONEDPL_DS_SCHEDULER_DEFS_H*/
