// -*- C++ -*-
//===-- memory_fwd.h ------------------------------------------------------===//
//
// Copyright (C) 2019-2020 Intel Corporation
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

#ifndef _PSTL_memory_fwd_H
#define _PSTL_memory_fwd_H

namespace pstl
{
namespace __internal
{

template <typename... _ExecutionPolicy>
struct __op_uninitialized_copy;

template <typename... _ExecutionPolicy>
struct __op_uninitialized_move;

template <typename _SourceT, typename... _ExecutionPolicy>
struct __op_uninitialized_fill;

template <typename... _ExecutionPolicy>
struct __op_destroy;

template <typename... _ExecutionPolicy>
struct __op_uninitialized_default_construct;

template <typename... _ExecutionPolicy>
struct __op_uninitialized_value_construct;

} // namespace __internal
} // namespace pstl

#endif /*_PSTL_memory_fwd_H*/
