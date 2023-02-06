// -*- C++ -*-
//===-- memory_fwd.h ------------------------------------------------------===//
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

#ifndef _ONEDPL_MEMORY_FWD_H
#define _ONEDPL_MEMORY_FWD_H

namespace oneapi
{
namespace dpl
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
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_MEMORY_FWD_H
