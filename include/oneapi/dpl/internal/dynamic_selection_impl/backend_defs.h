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

  template<typename Resource>
  class basic_execution_resource_t {
    using resource_t = Resource;
    resource_t resource_;
  public:
    basic_execution_resource_t() : resource_(resource_t{}) {}
    basic_execution_resource_t(resource_t r) : resource_(r) {}
    resource_t unwrap() const { return resource_; }
    bool operator==(const basic_execution_resource_t& e) const {
      return resource_ == e.resource_;
    }
    bool operator==(const resource_t& e) const {
      return resource_ == e;
    }
  };

} // namespace experimental

} // namespace dpl

} // namespace oneapi

#endif //_ONEDPL_BACKEND_DEFS_H
