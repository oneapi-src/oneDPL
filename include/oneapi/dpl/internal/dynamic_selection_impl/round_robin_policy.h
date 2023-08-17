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

  template <typename Backend>
  struct round_robin_policy {
  private:
    using resource_container_t = typename Backend::resource_container_t;
    using resource_container_size_t = typename resource_container_t::size_type;
  public:
    using resource_t = typename Backend::resource_t; 
    using selection_t = oneapi::dpl::experimental::basic_selection_t<resource_t>; 
    using wait_t = typename Backend::wait_t; 
    using submission_group_t = Backend::submission_group_t; 

  private:
    std::shared_ptr<Backend> backend_;

    struct state_t{
        resource_container_t resource_;
        resource_container_size_t num_contexts_;
        std::atomic<resource_container_size_t> next_context_;
	int offset_;
    };

    std::shared_ptr<state_t> state_;
  public:
    round_robin_policy(int offset=0) : backend_{std::make_shared<Backend>()}, state_{std::make_shared<state_t>()}  {
      state_->resource_ = get_resources();
      state_->num_contexts_ = state_->resource_.size();
      state_->offset_ = offset;
      state_->next_context_ = state_->offset_;
  }

    round_robin_policy(resource_container_t u, int offset=0) : backend_{std::make_shared<Backend>()}, state_{std::make_shared<state_t>()}  {
      backend_->initialize(u);
      state_->resource_ = get_resources();
      state_->num_contexts_ = state_->resource_.size();
      state_->offset_ = offset;
      state_->next_context_ = state_->offset_;
    }

    template<typename ...Args>
    round_robin_policy_impl(Args&&... args) : backend_{std::make_shared<Backend>(std::forward<Args>(args)...)}, state_{std::make_shared<state_t>()} {
      state_->resource_ = backend_->get_resources();
      state_->num_contexts_ = state_->resource_.size();
      state_->next_context_ = 0;
    }

    auto get_resources() const noexcept {
      return backend_->get_resources();
    }

    void initialize(int offset=0) {
      if (offset == -1) return;
      state_->offset_ = offset;
      backend_->initialize();
    }

    void initialize(resource_container_t u, int offset=0) {
      state_->offset_ = offset;
      backend_->initialize(u);
    }

    template<typename ...Args>
    selection_t select(Args&&...) {
      size_t i=state_->offset_;
      while(true){
          resource_container_size_t current_context_ = state_->next_context_.load();
          resource_container_size_t new_context_;
          if(current_context_ == std::numeric_limits<resource_container_size_t>::max()){
              new_context_ = (current_context_%state_->num_contexts_)+1;
          }
          else{
              new_context_ = (current_context_+1)%state_->num_contexts_;
          }

          if(state_->next_context_.compare_exchange_weak(current_context_, new_context_)){
              i = current_context_;
              break;
          }
      }
      auto &e = state_->resource_[i];
      return selection_t{e};
    }

    template<typename Function, typename ...Args>
    auto submit(selection_t e, Function&& f, Args&&... args) {
      return backend_->submit(e, std::forward<Function>(f), std::forward<Args>(args)...);
    }

    auto get_submission_group() {
      return backend_->get_submission_group();
    }

  };
} // namespace experimental

} // namespace dpl

} // namespace oneapi


#endif //_ONEDPL_ROUND_ROBIN_POLICY_IMPL_H
