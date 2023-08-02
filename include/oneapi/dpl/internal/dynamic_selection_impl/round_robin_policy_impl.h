// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _ONEDPL_ROUND_ROBIN_POLICY_IMPL_H
#define _ONEDPL_ROUND_ROBIN_POLICY_IMPL_H

#include <atomic>
#include "oneapi/dpl/internal/dynamic_selection_impl/scoring_policy_defs.h"

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
    using selection_handle_t = oneapi::dpl::experimental::basic_selection_handle_t<execution_resource_t>;
    using universe_container_size_t = typename universe_container_t::size_type;

    std::shared_ptr<scheduler_t> sched_;

    struct unit_t{
        universe_container_t universe_;
        universe_container_size_t num_contexts_;
        std::atomic<universe_container_size_t> next_context_;
    };

    std::shared_ptr<unit_t> unit_;

    round_robin_policy_impl() : sched_{std::make_shared<scheduler_t>()}, unit_{std::make_shared<unit_t>()}  {
      unit_->universe_ = oneapi::dpl::experimental::property::query(*sched_, property::universe);
      unit_->num_contexts_ = unit_->universe_.size();
      unit_->next_context_ = 0;
    }

    round_robin_policy_impl(universe_container_t u) : sched_{std::make_shared<scheduler_t>()}, unit_{std::make_shared<unit_t>()}  {
      oneapi::dpl::experimental::property::report(*sched_, property::universe, u);
      unit_->universe_ = oneapi::dpl::experimental::property::query(*sched_, property::universe);
      unit_->num_contexts_ = unit_->universe_.size();
      unit_->next_context_ = 0;
    }

    template<typename ...Args>
    round_robin_policy_impl(Args&&... args) : sched_{std::make_shared<scheduler_t>(std::forward<Args>(args)...)}, unit_{std::make_shared<unit_t>()} {
      unit_->universe_ = oneapi::dpl::experimental::property::query(*sched_, property::universe);
      unit_->num_contexts_ = unit_->universe_.size();
      unit_->next_context_ = 0;
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

    template<typename ...Args>
    selection_handle_t select(Args&&...) {
      auto i = unit_->next_context_++ % unit_->num_contexts_;
      auto &e = unit_->universe_[i];
      return selection_handle_t{e};
    }

    template<typename Function, typename ...Args>
    auto invoke_async(selection_handle_t e, Function&& f, Args&&... args) {
      return sched_->submit(e, std::forward<Function>(f), std::forward<Args>(args)...);
    }

    auto get_wait_list() {
      return sched_->get_wait_list();
    }

    auto wait() {
      sched_->wait();
    }
  };
} // namespace experimental

} // namespace dpl

} // namespace oneapi


#endif //_ONEDPL_ROUND_ROBIN_POLICY_IMPL_H
