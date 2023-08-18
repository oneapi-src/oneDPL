// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _ONEDPL_BACKEND_DEFS_H
#define _ONEDPL_BACKEND_DEFS_H

namespace oneapi {
namespace dpl{
namespace experimental{

  template<typename NativeContext>
  struct basic_execution_resource_t {
    using native_resource_t = NativeContext;
    native_resource_t native_resource_;
    basic_execution_resource_t() : native_resource_(native_resource_t{}) {}
    basic_execution_resource_t(native_resource_t nc) : native_resource_(nc) {}
    native_resource_t get_native() const { return native_resource_; }
    bool operator==(const basic_execution_resource_t& e) const {
      return native_resource_ == e.native_resource_;
    }
    bool operator==(const native_resource_t& e) const {
      return native_resource_ == e;
    }
  };

} // namespace experimental

} // namespace dpl

} // namespace oneapi

#endif //_ONEDPL_BACKEND_DEFS_H
