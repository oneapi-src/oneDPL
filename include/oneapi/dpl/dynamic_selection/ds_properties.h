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
  namespace property {
    struct universe_t {
      static constexpr bool is_property_v = true;
      static constexpr bool can_report_v = false;
    };
    inline constexpr universe_t universe;

    struct universe_size_t {
      static constexpr bool is_property_v = true;
      static constexpr bool can_report_v = false;
    };
    inline constexpr universe_size_t universe_size;

    struct dynamic_load_t {
      static constexpr bool is_property_v = true;
      static constexpr bool can_report_v = false;
    };
    inline constexpr dynamic_load_t dynamic_load;
  
    struct is_device_available_t {
      static constexpr bool is_property_v = true;
      static constexpr bool can_report_v = false;
    };
    inline constexpr is_device_available_t is_device_available;

    struct task_execution_time_t {
      static constexpr bool is_property_v = true;
      static constexpr bool can_report_v = true;
    };
    inline constexpr task_execution_time_t task_execution_time;

    struct task_submission_t {
      static constexpr bool is_property_v = true;
      static constexpr bool can_report_v = true;
    };
    inline constexpr task_submission_t task_submission;

    struct task_completion_t {
      static constexpr bool is_property_v = true;
      static constexpr bool can_report_v = true;
    };
    inline constexpr task_completion_t task_completion;

    template<typename T, typename Property>
    inline auto query(T& t, const Property& prop) {
      return t.query(prop);
    }

    template<typename T, typename Property, typename Argument>
    inline auto query(T& t, const Property& prop, const Argument& arg) {
      return t.query(prop, arg);
    }

    template<typename Handle, typename Property>
    inline auto report(Handle&& h, const Property& prop) {
      return std::forward<Handle>(h).report(prop);
    }

    template<typename Handle, typename Property, typename ValueType>
    inline auto report(Handle&& h, const Property& prop, const ValueType& v) {
      return std::forward<Handle>(h).report(prop, v);
    }
  }
}

