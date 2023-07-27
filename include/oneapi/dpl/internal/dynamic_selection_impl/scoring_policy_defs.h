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
  struct basic_property_handle_t {
    static constexpr bool should_report_task_submission = false;
    static constexpr bool should_report_task_completion = false;
    static constexpr bool should_report_task_execution_time = false;
    template<typename Property>
    static constexpr void report(const Property&) noexcept { return; }
    template<typename Property, typename ValueType>
    static constexpr void report(const Property&, ValueType) noexcept { return; }
  };
  inline constexpr basic_property_handle_t basic_property_handle;

  template<typename ExecutionContext>
  class basic_selection_handle_t {
    ExecutionContext e_;
  public:
    using property_handle_t = oneapi::dpl::experimental::basic_property_handle_t;
    using execution_resource_t = ExecutionContext;
    using native_resource_t = typename execution_resource_t::native_resource_t;

    explicit basic_selection_handle_t(execution_resource_t e = execution_resource_t{}) : e_(e) {}
    native_resource_t get_native() { return e_.get_native(); }
    property_handle_t get_property_handle() { return oneapi::dpl::experimental::basic_property_handle; }
  };

} // namespace experimental

} // namespace dpl

} //namespace oneapi


#endif /* _ONEDPL_SCORING_POLICY_DEFS_H */
