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
  struct dynamic_load_policy {

    using scheduler_t = Scheduler;
    using resource_type = typename scheduler_t::resource_type;
    using wait_type = typename scheduler_t::wait_type;
    using load_t = int;

    private:
    struct resource_t {
      resource_type e_;
      std::atomic<load_t> load_;
      resource_t(resource_type e) : e_(e) { load_ = 0; }
    };

    using resource_container_t = std::vector<std::shared_ptr<resource_t>>;
    using resource_container_size_t = typename resource_container_t::size_type;

    public :

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

    struct dl_selection_handle_t {
      using property_handle_t = dl_property_handle_t;
     // using resource_type = typename resource_t::resource_type;

      property_handle_t h_;

      dl_selection_handle_t(property_handle_t h) : h_(h) {}
      resource_type get_native() { return h_.resource_ptr_->e_; }
      auto get_property_handle() { return h_; }
    };

    using selection_type = dl_selection_handle_t;
    std::shared_ptr<scheduler_t> sched_;

    struct state_t{
        resource_container_t universe_;
        int offset;
    };

    std::shared_ptr<state_t> state_;

    dynamic_load_policy(int offset=0) : sched_{std::make_shared<scheduler_t>()}, state_{std::make_shared<state_t>()}  {
      auto u = get_universe();
      for (auto e : u) {
        state_->universe_.push_back(std::make_shared<resource_t>(e));
      }
      state_->offset=offset;
    }

    dynamic_load_policy(resource_container_t u, int offset=0) : sched_{std::make_shared<scheduler_t>()} , state_{std::make_shared<state_t>()}  {
      sched_->set_universe(u);
      auto u2 = get_universe();
      for (auto e : u2) {
        state_->universe_.push_back(std::make_shared<resource_t>(e));
      }
      state_->offset=offset;
    }

    template<typename ...Args>
    dynamic_load_policy(Args&&... args) : sched_{std::make_shared<scheduler_t>(std::forward<Args>(args)...)}, state_{std::make_shared<state_t>()} {
      auto u = get_universe();
      for (auto e : u) {
        state_->universe_.push_back(std::make_shared<resource_t>(e));
      }
      state_->offset=0;
    }

    void initialize(int offset=0) {
      state_->offset_ = offset;
      sched_->initialize();
      state_->universe_=get_universe();
    }

    void initialize(resource_container_t u, int offset=0) {
      state_->offset_ = offset;
      sched_->initialize(u);
      state_->universe_=get_universe();
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

    auto query(oneapi::dpl::experimental::property::dynamic_load_t, typename scheduler_t::resource_type e) const noexcept {
      for (const auto& r : state_->universe_) {
        if (r->e_ == e) {
          return r->load_.load();
        }
      }
    }

    template<typename ...Args>
    dl_selection_handle_t select(Args&&...) {
      std::shared_ptr<resource_t> least_loaded = nullptr;
      int least_load = std::numeric_limits<load_t>::max();
      int i=0;
      int least=0;
      for (auto& r : state_->universe_) {
          load_t v = r->load_.load();
          if (least_loaded == nullptr || v < least_load) {
            least_load = v;
            least_loaded = r;
            least=i;

          }
          i++;
      }
      return dl_selection_handle_t{dl_property_handle_t{least_loaded}};
    }

    template<typename Function, typename ...Args>
    auto invoke_async(dl_selection_handle_t e, Function&& f, Args&&... args) {
      return sched_->submit(e, std::forward<Function>(f), std::forward<Args>(args)...);
    }


    auto get_submission_group() {
      return sched_->get_submission_group();
    }

    auto wait() {
      sched_->wait();
    }

  };

} // namespace experimental

} // namespace dpl

} // namespace oneapi


#endif //_ONEDPL_DYNAMIC_LOAD_POLICY_IMPL_H
