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

#ifndef _ONEDPL_SCHEDULER_DEFS_H
#define _ONEDPL_SCHEDULER_DEFS_H

#pragma once

namespace oneapi {
namespace dpl{
namespace experimental{

  template<typename NativeContext>
  struct nop_execution_resource_t {
    using native_resource_t = NativeContext;
    native_resource_t native_resource_;
    nop_execution_resource_t() : native_resource_(native_resource_t{}) {}
    nop_execution_resource_t(native_resource_t nc) : native_resource_(nc) {}
    native_resource_t get_native() const { return native_resource_; }
    bool operator==(const nop_execution_resource_t& e) const {
      return native_resource_ == e.native_resource_;
    }
    bool operator==(const native_resource_t& e) const {
      return native_resource_ == e;
    }
  };

} // namespace experimental

} // namespace dpl

} // namespace oneapi

#endif //_ONEDPL_SCHEDULER_DEFS_H
