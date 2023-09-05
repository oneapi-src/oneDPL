// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _ONEDPL_AUTO_TUNE_POLICY_H
#define _ONEDPL_AUTO_TUNE_POLICY_H

#include <stdexcept>
#include <map>
#include <memory>
#include <mutex>
#include <utility>
#include <chrono>
#include <limits>
#include <vector>
#include <type_traits>
#include <tuple>
#include "oneapi/dpl/internal/dynamic_selection_traits.h"
#if _DS_BACKEND_SYCL != 0
    #include "oneapi/dpl/internal/dynamic_selection_impl/sycl_backend.h"
#endif
namespace oneapi {
namespace dpl {
namespace experimental {
#if _DS_BACKEND_SYCL != 0
  template <typename Backend=sycl_backend, typename... KeyArgs>
#else
  template <typename Backend, typename... KeyArgs>
#endif
  class auto_tune_policy {

    static constexpr double never_resample = 0.0;
    static constexpr int use_best_resource = -1;

    using wrapped_resource_t = typename std::decay_t<Backend>::execution_resource_t;
    using size_type = typename std::vector<typename Backend::resource_type>::size_type;

    using timing_t = uint64_t;

    struct time_data_t {
      uint64_t num_timings_ = 0;
      timing_t value_ = 0;
    };

    struct tuner_t {
      std::mutex m_;

      std::chrono::steady_clock::time_point t0_;

      timing_t best_timing_ = std::numeric_limits<timing_t>::max();
      wrapped_resource_t best_resource_;

      const size_type max_resource_to_profile_;
      uint64_t next_resource_to_profile_ = 0;

      using time_t = std::map<size_type, time_data_t>;
      time_t time_;

      double resample_time_ = 0.0;

      tuner_t(wrapped_resource_t br, size_type resources_size, double rt)
        : t0_(std::chrono::steady_clock::now()),
          best_resource_(br),
          max_resource_to_profile_(resources_size),
          resample_time_(rt) {}

      size_type get_resource_to_profile() {
        std::lock_guard<std::mutex> l(m_);
        if (next_resource_to_profile_ < 2*max_resource_to_profile_) {
          // do everything twice
          return next_resource_to_profile_++ % max_resource_to_profile_;
        } else if (resample_time_ == never_resample) {
          return use_best_resource;
        } else {
          auto now = std::chrono::steady_clock::now();
          auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now-t0_).count();
          if (ms < resample_time_) {
            return use_best_resource;
          } else {
            t0_ = now;
            return next_resource_to_profile_ = 0;
          }
        }
      }

      // called to add new profile info
      void add_new_timing(wrapped_resource_t r, timing_t t) {
        std::unique_lock<std::mutex> l(m_);
        timing_t new_value = t;
        if (time_.count(0) == 0) {
          // ignore the 1st timing to cover for JIT compilation
          time_[0] = time_data_t{0, std::numeric_limits<timing_t>::max()};
        } else {
          auto &td = time_[0];
          auto n = td.num_timings_;
          new_value = (n*td.value_+t)/(n+1);
          td.num_timings_ = n+1;
          td.value_ = new_value;
        }
        if (new_value < best_timing_) {
          best_timing_ = new_value;
          best_resource_ = r;
        }
      }

    };

    class auto_tune_selection_type {
      using policy_t = auto_tune_policy<Backend, KeyArgs...>;
      policy_t policy_;
      wrapped_resource_t resource_;
      std::shared_ptr<tuner_t> tuner_;

    public:
      auto_tune_selection_type() : policy_(deferred_initialization) {}

      auto_tune_selection_type(const policy_t& p, wrapped_resource_t r, std::shared_ptr<tuner_t> t)
        : policy_(p), resource_(r), tuner_(::std::move(t)) {}

      auto unwrap() { return ::oneapi::dpl::experimental::unwrap(resource_); }

      policy_t get_policy() { return policy_; };

      void report(const execution_info::task_time_t&, const typename execution_info::task_time_t::value_type& v) const {
        tuner_->add_new_timing(resource_, v);
      }
    };

   public:

    // Needed by Policy Traits
    using resource_type = decltype(unwrap(std::declval<wrapped_resource_t>()));
    using wait_type = typename Backend::wait_type;
    using selection_type = auto_tune_selection_type;

    auto_tune_policy(double resample_time=never_resample) {
        initialize(resample_time);
    }


    auto_tune_policy(deferred_initialization_t) {}

    auto_tune_policy(const std::vector<resource_type>& u, double resample_time=never_resample) {
        initialize(u, resample_time);
    }

    void initialize(double resample_time=never_resample) {
      if (!state_) {
        state_ = std::make_shared<state_t>();
        backend_ = std::make_shared<Backend>();
        initialize_impl(resample_time);
      }
    }

    void initialize(const std::vector<resource_type>& u, double resample_time=never_resample) {
      if (!state_) {
        state_ = std::make_shared<state_t>();
        backend_ = std::make_shared<Backend>(u);
        initialize_impl(resample_time);
      }
    }

    template<typename Function, typename ...Args>
    selection_type select(Function&& f, Args&&...args) {
      if (state_) {
        std::unique_lock<std::mutex> l(state_->m_);
        auto k =  make_task_key(std::forward<Function>(f), std::forward<Args>(args)...);
        auto t  = state_->tuner_by_key_[k];
        auto i = t->get_resource_to_profile();
        if (i == use_best_resource) {
          return selection_type{*this, t->best_resource_, t};
        } else {
          auto r = state_->resources_[i];
          return selection_type{*this, r, t};
        }
      } else {
         throw std::logic_error("select called before initialization");
      }
    }

    template<typename Function, typename ...Args>
    auto submit(selection_type e, Function&& f, Args&&... args) {
      if (backend_) {
        return backend_->submit(e, std::forward<Function>(f), std::forward<Args>(args)...);
      } else {
         throw std::logic_error("submit called before initialization");
      }
    }

    auto get_resources() {
       if (backend_) {
         return backend_->get_resources();
       } else {
         throw std::logic_error("get_resources called before initialization");
       }
    }

    auto get_submission_group() {
      if (backend_) {
        return backend_->get_submission_group();
       } else {
         throw std::logic_error("get_submission_group called before initialization");
       }
    }

  private:

    //
    // types
    //

    using task_key_t = std::tuple<void *, KeyArgs...>;
    using tuner_by_key_t = std::map<task_key_t, std::shared_ptr<tuner_t>>;

    //
    // member variables
    //

    double resample_time_ = 0.0;

    struct state_t {
      std::mutex m_;
      std::vector<wrapped_resource_t> resources_;
      tuner_by_key_t tuner_by_key_;
    };

    std::shared_ptr<Backend> backend_;
    std::shared_ptr<state_t> state_;

    //
    // private member functions
    //

    void initialize_impl(double resample_time=never_resample) {
      resample_time_ = resample_time;
      auto u = get_resources();
      for (size_type i = 0; i < u.size(); ++i) {
        state_->resources_.push_back(u[i]);
      }
    }

    template<typename Function, typename... Args>
    task_key_t make_task_key(Function&& f, Args&&... args) {
      // called under lock
      task_key_t k = std::tuple_cat(std::tuple<void *>(&f), std::make_tuple(std::forward<Args>(args)...));
      if (state_->tuner_by_key_.count(k) == 0) {
        state_->tuner_by_key_[k] = std::make_shared<tuner_t>(state_->resources_[0], state_->resources_.size(), resample_time_);
      }
      return k;
    }


  };

}
}
}

#endif //_ONEDPL_AUTO_TUNE_POLICY_H


