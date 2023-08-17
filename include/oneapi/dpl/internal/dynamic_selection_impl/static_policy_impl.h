// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _ONEDPL_STATIC_POLICY_IMPL_H
#define _ONEDPL_STATIC_POLICY_IMPL_H

#include "oneapi/dpl/internal/dynamic_selection_impl/scoring_policy_defs.h"

namespace oneapi {
namespace dpl {
namespace experimental {

  template <typename Scheduler>
  struct static_policy_impl {
    using scheduler_t = Scheduler;
    using universe_container_t = typename scheduler_t::universe_container_t;
    using execution_resource_t = typename scheduler_t::execution_resource_t;

    //policy traits
    using resource_type = typename scheduler_t::resource_type;
    using selection_type = oneapi::dpl::experimental::basic_selection_handle_t<execution_resource_t>;
    using wait_type = typename scheduler_t::wait_type;
    std::shared_ptr<scheduler_t> sched_;


    struct unit_t{
        universe_container_t universe_;
    };

    std::shared_ptr<unit_t> unit_;

    static_policy_impl() : sched_{std::make_shared<scheduler_t>()}, unit_{std::make_shared<unit_t>()}  {
      unit_->universe_ = get_universe();
    }

    static_policy_impl(universe_container_t u) : sched_{std::make_shared<scheduler_t>()}, unit_{std::make_shared<unit_t>()} {
      sched_->set_universe(u);
      unit_->universe_ = get_universe();
    }

    template<typename ...Args>
    static_policy_impl(Args&&... args) : sched_{std::make_shared<scheduler_t>(std::forward<Args>(args)...)}, unit_{std::make_shared<unit_t>()} {
      unit_->universe_ = get_universe();
    }

    //
    // Support for property queries
    //

    auto get_universe()  const noexcept {
      return sched_->get_universe();
    }

    template<typename ...Args>
    auto set_universe(Args&&... args) {
        return sched_->set_universe(std::forward<Args>(args)...);
    }

    template<typename ...Args>
    selection_type select(Args&&...) {
      if(!unit_->universe_.empty()) {
          return selection_type{unit_->universe_[0]};
      }
      return selection_type{};
    }

    template<typename Function, typename ...Args>
    auto submit(selection_type e, Function&& f, Args&&... args) {
      return sched_->submit(e, std::forward<Function>(f), std::forward<Args>(args)...);
    }

    auto get_wait_list() {
      return sched_->get_wait_list();
    }

    auto wait() {
      sched_->wait();
    }
  };
} //namespace experimental
} //namespace dpl
} //namespace oneapi

#endif /*_ONEDPL_STATIC_POLICY_IMPL_H*/
