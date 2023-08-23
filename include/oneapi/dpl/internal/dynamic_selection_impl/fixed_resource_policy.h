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
#if _DS_BACKEND_SYCL != 0
    #include "oneapi/dpl/internal/dynamic_selection_impl/sycl_backend.h"
#endif
namespace oneapi {
namespace dpl {
namespace experimental {

#if _DS_BACKEND_SYCL != 0
  template <typename Backend=sycl_backend>
#else
  template <typename Backend>
#endif
  struct fixed_resouce_policy {
    using backend_t = Backend;
    using resource_container_t = typename backend_t::resource_container_t;
    using execution_resource_t = typename backend_t::execution_resource_t;

    //policy traits
    using resource_type = typename backend_t::resource_type;
    using selection_type = oneapi::dpl::experimental::basic_selection_handle_t<fixed_resouce_policy<Backend>, execution_resource_t>;
    using wait_type = typename backend_t::wait_type;
    std::shared_ptr<backend_t> backend_;

    struct state_t {
        resource_container_t resources_;
        int offset_;
    };

    std::shared_ptr<state_t> state_;

    fixed_resouce_policy(int offset) : backend_{std::make_shared<backend_t>()}, state_{std::make_shared<state_t>()}  {
      state_->resources_ = get_resources();
      state_->offset_ = offset;
    }

    fixed_resouce_policy(resource_container_t u, int offset) : backend_{std::make_shared<backend_t>()}, state_{std::make_shared<state_t>()} {
      backend_->initialize(u);
      state_->resources_ = get_resources();
      state_->offset_ = offset;
    }

    template<typename ...Args>
    fixed_resouce_policy(Args&&... args) : backend_{std::make_shared<backend_t>(std::forward<Args>(args)...)}, state_{std::make_shared<state_t>()} {
      state_->resources_ = get_resources();
    }

    auto get_resources()  const {
      return backend_->get_resources();
    }

    void initialize(int offset=0) {
      if (offset == deferred_initialization) return;
      state_->offset_ = offset;
      backend_->initialize();
    }

    void initialize(resource_container_t u, int offset=0) {
      state_->offset_ = offset;
      backend_->initialize(u);
    }

    template<typename ...Args>
    selection_type select(Args&&...) {
      if(!state_->resources_.empty()) {
          return selection_type{*this, state_->resources_[state_->offset_]};
      }
      return selection_type{*this};
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
} //namespace experimental
} //namespace dpl
} //namespace oneapi

#endif /*_ONEDPL_STATIC_POLICY_IMPL_H*/
