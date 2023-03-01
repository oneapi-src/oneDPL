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

#include <memory>
#include <utility>
#include <iostream>
#include <sstream>

#ifndef _ONEDPL_DYNAMIC_SELECTION_DEFS_H
#define _ONEDPL_DYNAMIC_SELECTION_DEFS_H
#pragma once


namespace oneapi {
namespace dpl {
namespace experimental {
//ds_properties

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
  } //namespace property
}
}
}

namespace oneapi {
namespace dpl {
namespace experimental {
//ds_algorithms
  template<typename Handle>
  inline auto wait_for_all(Handle&& h) {
    return std::forward<Handle>(h).wait_for_all();
  }

  template<typename DSPolicy, typename... Args>
  inline auto select(DSPolicy&& dp, Args&&... args) {
    return std::forward<DSPolicy>(dp).select(std::forward<Args>(args)...);
  }

  template<typename DSPolicy, typename Function, typename... Args>
  inline auto invoke_async(DSPolicy&& dp, Function&&f, Args&&... args) {
    return std::forward<DSPolicy>(dp).invoke_async(std::forward<Function>(f), std::forward<Args>(args)...);
  }

  template<typename DSPolicy, typename Function, typename... Args>
  inline auto invoke(DSPolicy&& dp, Function&&f, Args&&... args) {
    return wait_for_all(invoke_async(std::forward<DSPolicy>(dp), std::forward<Function>(f), std::forward<Args>(args)...));
  }

  template<typename DSPolicy, typename Function, typename... Args>
  inline auto invoke_async(DSPolicy&& dp, typename DSPolicy::selection_handle_t e, Function&&f, Args&&... args) {
    return std::forward<DSPolicy>(dp).invoke_async(e, std::forward<Function>(f), std::forward<Args>(args)...);
  }

  template<typename DSPolicy, typename Function, typename... Args>
  inline auto invoke(DSPolicy&& dp, typename DSPolicy::selection_handle_t e, Function&&f, Args&&... args) {
    return wait_for_all(invoke_async(std::forward<DSPolicy>(dp), e, std::forward<Function>(f), std::forward<Args>(args)...));
  }
//ds_policy

  template<typename ScoringPolicy>
  struct policy {
    using scoring_policy_t = ScoringPolicy;
    using native_resource_t = typename scoring_policy_t::native_resource_t;
    using native_sync_t = typename scoring_policy_t::native_sync_t;
    using selection_handle_t = typename scoring_policy_t::selection_handle_t;
    std::shared_ptr<ScoringPolicy> scoring_policy_;

    template<typename... Args>
    policy(Args&&... args) : scoring_policy_{std::make_shared<scoring_policy_t>(std::forward<Args>(args)...)} {}

    template<typename Property>
    auto query(const Property &prop) const {
      return const_cast<const scoring_policy_t &>(*scoring_policy_).query(prop);
    }

    template<typename Property, typename Other>
    auto query(const Property &prop, const Other &other) const {
      return const_cast<const scoring_policy_t &>(*scoring_policy_).query(prop, other);
    }

    template<typename Property, typename Value>
    auto report(const Property &prop, const Value &value) const {
      return const_cast<const scoring_policy_t &>(*scoring_policy_).report(prop, value);
    }

    template<typename ...Args>
    auto select(Args&&... args) {
      return scoring_policy_->select(std::forward<Args>(args)...);
    }

    template<typename Function, typename ...Args>
    auto invoke_async(Function&& f, Args&&... args) {
      return scoring_policy_->invoke_async(std::forward<Function>(f), std::forward<Args>(args)...);
    }

    template<typename Function, typename ...Args>
    auto invoke_async(native_resource_t e, Function&& f, Args&&... args) {
      return scoring_policy_->invoke_async(e, std::forward<Function>(f), std::forward<Args>(args)...);
    }

    template<typename Function, typename ...Args>
    auto invoke(Function&& f, Args&&... args) {
      return scoring_policy_->invoke(std::forward<Function>(f), std::forward<Args>(args)...);
    }

    template<typename Function, typename ...Args>
    auto invoke(native_resource_t e, Function&& f, Args&&... args) {
      return scoring_policy_->invoke(e, std::forward<Function>(f), std::forward<Args>(args)...);
    }

    auto wait_for_all() {
      return scoring_policy_->wait_for_all();
    }

    template<typename SP>
    friend std::ostream& operator<<(std::ostream &os, const policy<SP>& p);
  };

  template<typename SP>
  std::ostream& operator<<(std::ostream &os, const oneapi::dpl::experimental::policy<SP>& q) {
    os << "DS policy:\n";
    os << *q.scoring_policy_;
    return os;
  }
} // namespace experimental
} // namespace dpl
} // namespace oneapi
#include "oneapi/dpl/internal/dynamic_selection_impl/sycl_scheduler.h"
#include "oneapi/dpl/internal/dynamic_selection_impl/static_policy_impl.h"
#include "oneapi/dpl/internal/dynamic_selection_impl/round_robin_policy_impl.h"

#endif /*_ONEDPL_DYNAMIC_SELECTION_DEFS_H*/
