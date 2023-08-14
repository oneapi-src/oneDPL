// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _ONEDPL_DYNAMIC_LOAD_POLICY_IMPL_H
#define _ONEDPL_DYNAMIC_LOAD_POLICY_IMPL_H

#include <atomic>
#include <memory>
#include <ostream>
#include <vector>
#include "oneapi/dpl/internal/dynamic_selection_impl/scoring_policy_defs.h"

namespace oneapi {
namespace dpl{
namespace experimental{

  template <typename Scheduler>
  struct dynamic_load_policy_impl {
    using scheduler_t = Scheduler;
    using native_resource_t = typename scheduler_t::native_resource_t;
    using execution_resource_t = typename scheduler_t::execution_resource_t;
    using native_sync_t = typename scheduler_t::native_sync_t;
    using load_t = int;

    struct resource_t {
      execution_resource_t e_;
      std::atomic<load_t> load_;
      resource_t(execution_resource_t e) : e_(e) { load_ = 0; }
    };

    using universe_container_t = std::vector<std::shared_ptr<resource_t>>;
    using universe_container_size_t = typename universe_container_t::size_type;

    struct dl_property_handle_t {
      std::shared_ptr<resource_t> resource_ptr_;
      static constexpr bool should_report_task_submission = true;
      static constexpr bool should_report_task_completion = true;
      static constexpr bool should_report_task_execution_time = false;

      void report(const oneapi::dpl::experimental::property::task_submission_t&) const noexcept {
        auto v = resource_ptr_->load_.fetch_add(1);
      }
      void report(const oneapi::dpl::experimental::property::task_completion_t&)  const noexcept {
        auto v = resource_ptr_->load_.fetch_sub(1);
      }
      dl_property_handle_t(std::shared_ptr<resource_t> r) : resource_ptr_(r) {}
    };

    struct selection_handle_t {
      using property_handle_t = dl_property_handle_t;
      using native_resource_t = typename execution_resource_t::native_resource_t;

      property_handle_t h_;

      selection_handle_t(property_handle_t h) : h_(h) {}
      native_resource_t get_native() { return h_.resource_ptr_->e_.get_native(); }
      auto get_property_handle() { return h_; }
    };

    std::shared_ptr<scheduler_t> sched_;

    struct unit_t{
        universe_container_t universe_;
    };

    std::shared_ptr<unit_t> unit_;

    dynamic_load_policy_impl() : sched_{std::make_shared<scheduler_t>()}, unit_{std::make_shared<unit_t>()}  {
      //auto u = property::query(*sched_, property::universe);
      auto u = get_universe();
      for (auto e : u) {
        unit_->universe_.push_back(std::make_shared<resource_t>(e));
      }
    }

    dynamic_load_policy_impl(universe_container_t u) : sched_{std::make_shared<scheduler_t>()} , unit_{std::make_shared<unit_t>()}  {
      sched_->set_universe(u);
      auto u2 = get_universe();
      for (auto e : u2) {
        unit_->universe_.push_back(std::make_shared<resource_t>(e));
      }
    }

    template<typename ...Args>
    dynamic_load_policy_impl(Args&&... args) : sched_{std::make_shared<scheduler_t>(std::forward<Args>(args)...)}, unit_{std::make_shared<unit_t>()} {
      auto u = get_universe();
      for (auto e : u) {
        unit_->universe_.push_back(std::make_shared<resource_t>(e));
      }
    }

    //
    // Support for property queries
    //

    auto get_universe() const noexcept {
      return sched_->get_universe();
    }

    auto get_universe_size() const noexcept {
      return sched_->get_universe_size();
    }

    template<typename ...Args>
    auto set_universe(Args&&... args) {
        return sched_->set_universe(std::forward<Args>(args)...);
    }

    auto query(oneapi::dpl::experimental::property::dynamic_load_t, typename scheduler_t::native_resource_t e) const noexcept {
      for (const auto& r : unit_->universe_) {
        if (r->e_ == e) {
          return r->load_.load();
        }
      }
    }

    template<typename ...Args>
    selection_handle_t select(Args&&...) {
      std::shared_ptr<resource_t> least_loaded = nullptr;
      int least_load = std::numeric_limits<load_t>::max();
      int i=0;
      int least=0;
      for (auto& r : unit_->universe_) {
          load_t v = r->load_.load();
          if (least_loaded == nullptr || v < least_load) {
            least_load = v;
            least_loaded = r;
            least=i;

          }
          i++;
      }
      std::cout<<"Selected CPU : "<<least<<"\n";
      return selection_handle_t{dl_property_handle_t{least_loaded}};
    }

    template<typename Function, typename ...Args>
    auto invoke_async(Function&& f, Args&&... args) {
      return sched_->submit(select(f, args...), std::forward<Function>(f), std::forward<Args>(args)...);
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


#endif //_ONEDPL_DYNAMIC_LOAD_POLICY_IMPL_H
