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

#ifndef _ROUND_ROBIN_POLICY_IMPL_H
#define _ROUND_ROBIN_POLICY_IMPL_H
#pragma once

#include <atomic>
#include <ostream>
#include "oneapi/dpl/dynamic_selection/ds_properties.h"
#include "oneapi/dpl/internal/dynamic_selection/scoring_policy_defs.h"

namespace oneapi {
namespace dpl{
namespace experimental{

  template <typename Scheduler>
  struct round_robin_policy_impl {
    using scheduler_t = Scheduler;
    using native_resource_t = typename scheduler_t::native_resource_t;
    using execution_resource_t = typename scheduler_t::execution_resource_t;
    using native_sync_t = typename scheduler_t::native_sync_t;
    using universe_container_t = typename scheduler_t::universe_container_t;
    using selection_handle_t = ds::nop_selection_handle_t<execution_resource_t>;
    using universe_container_size_t = typename universe_container_t::size_type;

    std::shared_ptr<scheduler_t> sched_;
    universe_container_t universe_;
    universe_container_size_t num_contexts_;
    std::atomic<universe_container_size_t> next_context_;

    round_robin_policy_impl() : sched_{std::make_shared<scheduler_t>()}  {
      universe_ = oneapi::dpl::experimental::property::query(*sched_, property::universe);
      num_contexts_ = universe_.size();
      next_context_ = 0;
    }

    round_robin_policy_impl(universe_container_t u) : sched_{std::make_shared<scheduler_t>()}  {
      oneapi::dpl::experimental::property::report(*sched_, property::universe, u);
      universe_ = oneapi::dpl::experimental::property::query(*sched_, property::universe);
      num_contexts_ = universe_.size();
      next_context_ = 0;
    }

    template<typename ...Args>
    round_robin_policy_impl(Args&&... args) : sched_{std::make_shared<scheduler_t>(std::forward<Args>(args)...)} {
      universe_ = oneapi::dpl::experimental::property::query(*sched_, property::universe);
      num_contexts_ = universe_.size();
      next_context_ = 0;
    }

    //
    // Support for property queries
    //

    auto query(oneapi::dpl::experimental::property::universe_t) const noexcept {
      return oneapi::dpl::experimental::property::query(*sched_, property::universe);
    }

    auto query(oneapi::dpl::experimental::property::universe_size_t) const noexcept {
      return oneapi::dpl::experimental::property::query(*sched_, property::universe_size);
    }

    auto query(oneapi::dpl::experimental::property::dynamic_load_t, typename scheduler_t::native_resource_t e) const noexcept {
      return -1;
    }

    auto query(oneapi::dpl::experimental::property::is_device_available_t, typename scheduler_t::native_resource_t e) const noexcept {
      return oneapi::dpl::experimental::property::query(*sched_, property::is_device_available, e);
    }

    template<typename ...Args>
    selection_handle_t select(Args&&...) {
      auto i = next_context_++ % num_contexts_;
      auto &e = universe_[i];
      if(oneapi::dpl::experimental::property::query(*sched_, property::is_device_available, e)) {
        return selection_handle_t{e};
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
    friend std::ostream& operator<<(std::ostream &os, const round_robin_policy_impl<S>& p);
  };

  template<typename S>
  std::ostream& operator<<(std::ostream &os, const round_robin_policy_impl<S>& p) {
    os << "round_robin_policy_impl:\n";
    int r = 0;
    for (auto e : query(p, oneapi::dpl::experimental::property::universe)) {
      os << e.get_native();
    }
    return os;
  }

} // namespace experimental

} // namespace dpl

} // namespace oneapi


#endif //_ROUND_ROBIN_POLICY_IMPL_H
