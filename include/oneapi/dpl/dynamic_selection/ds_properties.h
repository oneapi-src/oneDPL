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


#ifndef _ONEDPL_DS_PROPERTIES_DEFS_H
#define _ONEDPL_DS_PROPERTIES_DEFS_H

#pragma once

namespace oneapi {
namespace dpl {
namespace experimental {
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
} //namespace experimental
} //namespace dpl
} //namespace oneapi
#endif /* _ONEDPL_DS_PROPERTIES_DEFS_H*/
