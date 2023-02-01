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


#ifndef _ONEDPL_DS_POLICY_DEFS_H
#define _ONEDPL_DS_POLICY_DEFS_H
#pragma once

#include <memory>
#include <ostream>

#include "oneapi/dpl/dynamic_selection/ds_properties.h"

namespace oneapi {
namespace dpl {
namespace experimental {
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

}  //namespace experimental
}  //namespace dpl
}  //namespace oneapi
#endif  /*_ONEDPL_DS_POLICY_DEFS_H*/
