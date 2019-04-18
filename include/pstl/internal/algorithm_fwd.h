// -*- C++ -*-
//===-- algorithm_fwd.h ---------------------------------------------------===//
//
// Copyright (C) 2017-2019 Intel Corporation
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

#ifndef __PSTL_algorithm_fwd_H
#define __PSTL_algorithm_fwd_H

namespace __pstl
{
namespace internal
{

template<typename... _ExecutionPolicy>
struct brick_copy_n;

template<typename... _ExecutionPolicy>
struct brick_copy;

template<typename... _ExecutionPolicy>
struct brick_move;

}
}

#endif
