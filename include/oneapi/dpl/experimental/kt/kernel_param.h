// -*- C++ -*-
//===-- kernel_param.h ------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------------------------------------------------===//

#ifndef _ONEDPL_KT_KERNEL_PARAM_H
#define _ONEDPL_KT_KERNEL_PARAM_H

#include <cstdint>

#include "../../pstl/hetero/dpcpp/execution_sycl_defs.h"

namespace oneapi::dpl::experimental::kt
{

template <std::uint16_t _DataPerWorkItem, std::uint16_t _WorkGroupSize,
          typename _KernelName = oneapi::dpl::execution::DefaultKernelName>
struct kernel_param
{
    static constexpr std::uint16_t data_per_workitem = _DataPerWorkItem;
    static constexpr std::uint16_t workgroup_size = _WorkGroupSize;
    using kernel_name = _KernelName;
};

} // namespace oneapi::dpl::experimental::kt

#endif // _ONEDPL_KT_KERNEL_PARAM_H
