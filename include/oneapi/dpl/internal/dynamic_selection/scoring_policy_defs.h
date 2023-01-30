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


#ifndef _SCORING_POLICY_DEFS_H
#define _SCORING_POLICY_DEFS_H

#pragma once

namespace oneapi {
namespace dpl {
namespace experimental {
  struct nop_property_handle_t {
    static constexpr bool should_report_task_submission = false;
    static constexpr bool should_report_task_completion = false;
    static constexpr bool should_report_task_execution_time = false;
    template<typename Property>
    static constexpr void report(const Property&) noexcept { return; }
    template<typename Property, typename ValueType>
    static constexpr void report(const Property&, ValueType) noexcept { return; }
  };
  inline constexpr nop_property_handle_t nop_property_handle;

  template<typename ExecutionContext>
  class nop_selection_handle_t {
    ExecutionContext e_;
  public:
    using property_handle_t = oneapi::dpl::experimental::nop_property_handle_t;
    using execution_resource_t = ExecutionContext;
    using native_resource_t = typename execution_resource_t::native_resource_t;

    nop_selection_handle_t(execution_resource_t e = execution_resource_t{}) : e_(e) {}
    native_resource_t get_native() { return e_.get_native(); }
    property_handle_t get_property_handle() { return oneapi::dpl::experimental::nop_property_handle; }
  };

} // namespace experimental

} // namespace dpl

} //namespace oneapi


#endif /* _SCORING_POLICY_DEFS_H */
