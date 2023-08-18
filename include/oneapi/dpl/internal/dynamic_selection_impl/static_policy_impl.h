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

  template <typename Backend>
  struct static_policy_impl {
    using backend_t = Backend;
    using resource_container_t = typename backend_t::resource_container_t;
    using execution_resource_t = typename backend_t::execution_resource_t;

    //policy traits
    using resource_type = typename backend_t::resource_type;
    using selection_type = oneapi::dpl::experimental::basic_selection_handle_t<execution_resource_t>;
    using wait_type = typename backend_t::wait_type;
    std::shared_ptr<backend_t> backend_;


    struct unit_t{
        resource_container_t resources_;
    };

    std::shared_ptr<unit_t> unit_;

    static_policy_impl() : backend_{std::make_shared<backend_t>()}, unit_{std::make_shared<unit_t>()}  {
      unit_->resources_ = get_resources();
    }

    static_policy_impl(resource_container_t u) : backend_{std::make_shared<backend_t>()}, unit_{std::make_shared<unit_t>()} {
      backend_->set_universe(u);
      unit_->resources_ = get_resources();
    }

    template<typename ...Args>
    static_policy_impl(Args&&... args) : backend_{std::make_shared<backend_t>(std::forward<Args>(args)...)}, unit_{std::make_shared<unit_t>()} {
      unit_->resources_ = get_resources();
    }

    //
    // Support for property queries
    //

    auto get_resources()  const noexcept {
      return backend_->get_resources();
    }

    template<typename ...Args>
    auto set_universe(Args&&... args) {
        return backend_->set_universe(std::forward<Args>(args)...);
    }

    template<typename ...Args>
    selection_type select(Args&&...) {
      if(!unit_->resources_.empty()) {
          return selection_type{unit_->resources_[0]};
      }
      return selection_type{};
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
