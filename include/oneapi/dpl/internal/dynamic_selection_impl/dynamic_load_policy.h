// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _ONEDPL_DYNAMIC_LOAD_POLICY_H
#define _ONEDPL_DYNAMIC_LOAD_POLICY_H

#include <atomic>
#include <memory>
#include <limits>
#include <mutex>
#include <vector>
#include <exception>
#include <type_traits>
#if _DS_BACKEND_SYCL != 0
    #include "oneapi/dpl/internal/dynamic_selection_impl/sycl_backend.h"
#endif

namespace oneapi {
namespace dpl{
namespace experimental{

#if _DS_BACKEND_SYCL != 0
  template <typename Backend=sycl_backend>
#else
  template <typename Backend>
#endif
  struct dynamic_load_policy {
    private:
    using backend_t = Backend;
    using load_t = int;
    using execution_resource_t = typename backend_t::execution_resource_t;

    public:
    //Policy Traits
    using resource_type = typename backend_t::resource_type;
    using wait_type = typename backend_t::wait_type;

    //private:
    struct resource_t {
      execution_resource_t e_;
      std::atomic<load_t> load_;
      resource_t(execution_resource_t e) : e_(e) {load_=0;}
    };
    using resource_container_t = std::vector<std::shared_ptr<resource_t>>;

    template<typename Policy>
    class dl_selection_handle_t {
      Policy policy_;
      std::shared_ptr<resource_t> resource_;

    public:
      dl_selection_handle_t(const Policy& p,std::shared_ptr<resource_t> r )
        : policy_(p), resource_(r) {}

      auto unwrap() { return ::oneapi::dpl::experimental::unwrap(resource_->e_); }

      Policy get_policy() { return policy_; };

      void report(const execution_info::task_submission_t&) {
        resource_->load_.fetch_add(1);
      }

      void report(const execution_info::task_completion_t&) const  {
        resource_->load_.fetch_sub(1);
      }
    };

    std::shared_ptr<backend_t> backend_;

    using selection_type = dl_selection_handle_t<dynamic_load_policy<Backend>>;

    struct state_t{
        resource_container_t resources_;
        int offset;
        std::mutex m_;
    };

    std::shared_ptr<state_t> state_;

    void initialize(int offset=0){
        if(!state_){
            backend_ = std::make_shared<backend_t>();
            state_= std::make_shared<state_t>();
            state_->offset=offset;
            auto u =  get_resources();
            for(auto x : u){
              state_->resources_.push_back(std::make_shared<resource_t>(x));
            }
        }

    }
    void initialize(const std::vector<resource_type> &u, int offset=0) {
        if(!state_){
            backend_ = std::make_shared<backend_t>(u);
            state_= std::make_shared<state_t>();
            state_->offset=offset;
            for(auto x : u){
              state_->resources_.push_back(std::make_shared<resource_t>(x));
            }
        }
    }

    dynamic_load_policy(int offset=0) {
        if(offset != deferred_initialization){
            initialize(offset);
        }
    }

    dynamic_load_policy(const std::vector<resource_type>& u,int offset=0) {
        if(offset != deferred_initialization){
            initialize(u, offset);
        }
    }

    auto get_resources() {
        if(backend_)
            return backend_->get_resources();
        else
           throw std::runtime_error("Called get_resources before initialization\n");
    }

    template<typename ...Args>
    selection_type select(Args&&...) {
      if(state_){
          std::unique_lock<std::mutex> l(state_->m_);
          std::shared_ptr<resource_t> least_loaded = nullptr;
          int least_load = std::numeric_limits<load_t>::max();
          for(int i = 0;i<state_->resources_.size();i++){
            auto r = state_->resources_[(i+state_->offset)%state_->resources_.size()];
            load_t v = r->load_.load();
              if (least_loaded == nullptr || v < least_load) {
                least_load = v;
                least_loaded = r;
              }
          }
          return selection_type{dynamic_load_policy<Backend>(*this), least_loaded};
      }else{
        throw std::runtime_error("Called select before initialization\n");
      }
    }

    template<typename Function, typename ...Args>
    auto submit(selection_type e, Function&& f, Args&&... args) {
      if(backend_)
        return backend_->submit(e, std::forward<Function>(f), std::forward<Args>(args)...);
      else
          throw std::runtime_error("Called submit before initialization\n");
    }

    auto get_submission_group() {
      return backend_->get_submission_group();
    }

  };
} // namespace experimental

} // namespace dpl

} // namespace oneapi


#endif //_ONEDPL_ROUND_ROBIN_POLICY_IMPL_H
