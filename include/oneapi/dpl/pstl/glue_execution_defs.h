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

#ifndef _ONEDPL_GLUE_EXECUTION_DEFS_H
#define _ONEDPL_GLUE_EXECUTION_DEFS_H

#include <type_traits>

#include "execution_defs.h"

#if _ONEDPL_BACKEND_SYCL
#    include "hetero/algorithm_impl_hetero.h"
#    include "hetero/numeric_impl_hetero.h"
#endif

#include "algorithm_impl.h"
#include "numeric_impl.h"
#include "parallel_backend.h"

#endif /* _ONEDPL_GLUE_EXECUTION_DEFS_H */
