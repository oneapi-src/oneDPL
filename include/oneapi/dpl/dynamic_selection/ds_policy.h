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

#include <memory>
#include <ostream>

#include "ds_properties.h"

namespace ds {
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
  std::ostream& operator<<(std::ostream &os, const ds::policy<SP>& q) {
    os << "DS policy:\n";
    os << *q.scoring_policy_;
    return os;
  }

}

