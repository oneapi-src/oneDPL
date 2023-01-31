/*
    Copyright 2021 Intel Corporation.  All Rights Reserved.

    The source code contained or described herein and all documents related
    to the source code ("Material") are owned by Intel Corporation or its
    suppliers or licensors.  Title to the Material remains with Intel
    Corporation or its suppliers and licensors.  The Material is protected
    by worldwide copyright laws and treaty provisions.  No part of the
    Material may be used, copied, reproduced, modified, published, uploaded,
    posted, transmitted, distributed, or disclosed in any way without
    Intel's prior express written permission.

    No license under any patent, copyright, trade secret or other
    intellectual property right is granted to or conferred upon you by
    disclosure or delivery of the Materials, either expressly, by
    implication, inducement, estoppel or otherwise.  Any license under such
    intellectual property rights must be express and approved by Intel in
    writing.
*/

#pragma once

namespace ds {
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
    using property_handle_t = ds::nop_property_handle_t;
    using execution_resource_t = ExecutionContext;
    using native_resource_t = typename execution_resource_t::native_resource_t;

    nop_selection_handle_t(execution_resource_t e = execution_resource_t{}) : e_(e) {}
    native_resource_t get_native() { return e_.get_native(); }
    property_handle_t get_property_handle() { return ds::nop_property_handle; }
  };

}

