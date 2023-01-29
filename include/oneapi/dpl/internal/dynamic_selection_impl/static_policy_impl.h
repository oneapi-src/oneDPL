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

#include <ostream>
#include "oneapi/dpl/dynamic_selection/ds_properties.h"
#include "oneapi/dpl/internal/dynamic_selection_impl/scoring_policy_defs.h"

namespace ds {

  template <typename Scheduler>
  struct static_policy_impl {
    using scheduler_t = Scheduler;
    using native_resource_t = typename scheduler_t::native_resource_t;
    using execution_resource_t = typename scheduler_t::execution_resource_t;
    using native_sync_t = typename scheduler_t::native_sync_t;
    using universe_container_t = typename scheduler_t::universe_container_t;
    using selection_handle_t = ds::nop_selection_handle_t<execution_resource_t>;

    std::shared_ptr<scheduler_t> sched_;
    universe_container_t universe_;

    static_policy_impl() : sched_{std::make_shared<scheduler_t>()} {
      universe_ = property::query(*sched_, property::universe);
    }

    static_policy_impl(universe_container_t u) : sched_{std::make_shared<scheduler_t>()} {
      property::report(*sched_, property::universe, u);
      universe_ = property::query(*sched_, property::universe);
    }

    template<typename ...Args>
    static_policy_impl(Args&&... args) : sched_{std::make_shared<scheduler_t>(std::forward<Args>(args)...)} {
      universe_ = property::query(*sched_, property::universe);
    }

    //
    // Support for property queries
    //

    auto query(property::universe_t) const noexcept {
      return property::query(*sched_, property::universe);
    }

    auto query(property::universe_size_t) const noexcept {
      return property::query(*sched_, property::universe_size);
    }

    auto query(property::dynamic_load_t, typename scheduler_t::native_resource_t e) const noexcept {
      return -1;
    }

    auto query(property::is_device_available_t, typename scheduler_t::native_resource_t e) const noexcept {
      return property::query(*sched_, property::is_device_available, e);
    }

    template<typename ...Args>
    selection_handle_t select(Args&&...) {
      for(auto& e : universe_) {
        if(property::query(*sched_, property::is_device_available, e)) {
          return selection_handle_t{e};
        }
      }
      return {};
    }

    template<typename Function, typename ...Args>
    auto invoke_async(Function&& f, Args&&... args) {
      return sched_->submit(select(f, args...), std::forward<Function>(f), std::forward<Args>(args)...);
    }

    template<typename Function, typename ...Args>
    auto invoke_async(selection_handle_t e, Function&& f, Args&&... args) {
      return sched_->submit(e, std::forward<Function>(f), std::forward<Args>(args)...);
    }

    template<typename Function, typename ...Args>
    auto invoke(Function&& f, Args&&... args) {
      return wait_for_all(sched_->submit(select(std::forward<Function>(f), std::forward<Args>(args)...), 
                                         std::forward<Function>(f), std::forward<Args>(args)...));
    }

    template<typename Function, typename ...Args>
    auto invoke(selection_handle_t e, Function&& f, Args&&... args) {
      return wait_for_all(sched_->submit(e, std::forward<Function>(f), std::forward<Args>(args)...));
    }

    auto wait_for_all() {
      sched_->wait_for_all();
    }

    template<typename S>
    friend std::ostream& operator<<(std::ostream &os, const static_policy_impl<S>& p);
  };

  template<typename S>
  std::ostream& operator<<(std::ostream &os, const static_policy_impl<S>& p) {
    os << "static_policy_impl:\n";
    int r = 0;
    for (auto e : query(p, property::universe)) {
      os << e.get_native();
    }
    return os;
  }

}

