// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _ONEDPL_PSTL_OFFLOAD_TOP_LEVEL_ONEAPI_TBB_PARTITIONER_H

#ifndef _ONEDPL_PSTL_OFFLOAD_TOP_LEVEL
#    define _ONEDPL_PSTL_OFFLOAD_TOP_LEVEL_ONEAPI_TBB_PARTITIONER_H
#    define _ONEDPL_PSTL_OFFLOAD_TOP_LEVEL
#endif // _ONEDPL_PSTL_OFFLOAD_TOP_LEVEL

#include_next <oneapi/tbb/partitioner.h>

#ifdef _ONEDPL_PSTL_OFFLOAD_TOP_LEVEL_ONEAPI_TBB_PARTITIONER_H
#    include "internal/usm_memory_replacement.h"
#    undef _ONEDPL_PSTL_OFFLOAD_TOP_LEVEL_ONEAPI_TBB_PARTITIONER_H
#    undef _ONEDPL_PSTL_OFFLOAD_TOP_LEVEL
#endif // _ONEDPL_PSTL_OFFLOAD_TOP_LEVEL_ONEAPI_TBB_PARTITIONER_H

#endif // _ONEDPL_PSTL_OFFLOAD_TOP_LEVEL_ONEAPI_TBB_PARTITIONER_H
