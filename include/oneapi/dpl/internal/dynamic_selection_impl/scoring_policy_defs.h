// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _ONEDPL_SCORING_POLICY_DEFS_H
#define _ONEDPL_SCORING_POLICY_DEFS_H

namespace oneapi {
namespace dpl {
namespace experimental {
  template<typename Resource>
  class basic_selection_handle_t {
    Resource e_;
  public:
    explicit basic_selection_handle_t(Resource e = Resource{}) : e_(e) {}
    auto unwrap() { return oneapi::dpl::experimental::unwrap(e_); }
  };

} // namespace experimental
} // namespace dpl
} //namespace oneapi

#endif /* _ONEDPL_SCORING_POLICY_DEFS_H */
