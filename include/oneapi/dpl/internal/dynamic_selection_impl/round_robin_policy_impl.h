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

  template <typename Backend = sycl_backend>
  struct round_robin_policy_impl {
    using backend_t = Backend;
    using resource_container_t = typename backend_t::resource_container_t;
    using resource_container_size_t = typename resource_container_t::size_type;
    using waiter_container_t = typename backend_t::waiter_container_t;

    using execution_resource_t = typename backend_t::execution_resource_t;

    //Policy Traits
    using selection_type = oneapi::dpl::experimental::basic_selection_handle_t<execution_resource_t>;
    using resource_type = typename backend_t::resource_type;
    using wait_type = typename backend_t::wait_type;


    std::shared_ptr<backend_t> backend_;


    struct unit_t{
        resource_container_t resources_;
        resource_container_size_t num_contexts_;
        std::atomic<resource_container_size_t> next_context_;
    };

    std::shared_ptr<unit_t> unit_;

    round_robin_policy_impl() : backend_{std::make_shared<backend_t>()}, unit_{std::make_shared<unit_t>()}  {
      unit_->resources_ = get_resources();
      unit_->num_contexts_ = unit_->resources_.size();
      unit_->next_context_ = 0;
    }

    round_robin_policy_impl(resource_container_t u) : backend_{std::make_shared<backend_t>()}, unit_{std::make_shared<unit_t>()}  {
      backend_->initialize(u);
      unit_->resources_ = get_resources();
      unit_->num_contexts_ = unit_->resources_.size();
      unit_->next_context_ = 0;
    }

    template<typename ...Args>
    round_robin_policy_impl(Args&&... args) : backend_{std::make_shared<backend_t>(std::forward<Args>(args)...)}, unit_{std::make_shared<unit_t>()} {
      unit_->resources_ = backend_->get_resources();
      unit_->num_contexts_ = unit_->resources_.size();
      unit_->next_context_ = 0;
    }

    //
    // Support for property queries
    //

    auto get_resources() const noexcept {
      return backend_->get_resources();
    }

    template<typename ...Args>
    auto initialize(Args&&... args) {
        return backend_->initialize(std::forward<Args>(args)...);
    }

    template<typename ...Args>
    selection_type select(Args&&...) {
      size_t i=0;
      while(true){
          resource_container_size_t current_context_ = unit_->next_context_.load();
          resource_container_size_t new_context_;
          if(current_context_ == std::numeric_limits<resource_container_size_t>::max()){
              new_context_ = (current_context_%unit_->num_contexts_)+1;
          }
          else{
              new_context_ = (current_context_+1)%unit_->num_contexts_;
          }

          if(unit_->next_context_.compare_exchange_weak(current_context_, new_context_)){
              i = current_context_;
              break;
          }
      }
      auto &e = unit_->resources_[i];
      return selection_type{e};
    }

    template<typename Function, typename ...Args>
    auto submit(selection_type e, Function&& f, Args&&... args) {
      return backend_->submit(e, std::forward<Function>(f), std::forward<Args>(args)...);
    }

    auto get_submission_group() {
      return backend_->get_submission_group();
    }

    auto wait() {
      backend_->wait();
    }
  };
} // namespace experimental

} // namespace dpl

} // namespace oneapi


#endif //_ONEDPL_ROUND_ROBIN_POLICY_IMPL_H
