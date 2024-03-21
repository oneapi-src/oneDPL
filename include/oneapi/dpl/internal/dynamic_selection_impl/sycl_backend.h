// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _ONEDPL_SYCL_BACKEND_IMPL_H
#define _ONEDPL_SYCL_BACKEND_IMPL_H

#include <sycl/sycl.hpp>
#include "oneapi/dpl/internal/dynamic_selection_traits.h"

#include "oneapi/dpl/internal/dynamic_selection_impl/scoring_policy_defs.h"

#include <chrono>
#include <vector>
#include <memory>
#include <utility>
#include <atomic>

namespace oneapi
{
namespace dpl
{
namespace experimental
{

class sycl_backend
{
  public:
    using resource_type = sycl::queue;
    using wait_type = sycl::event;
    using execution_resource_t = resource_type;
    using resource_container_t = std::vector<execution_resource_t>;

  private:
    class storage_base{
        public:
            virtual void wait() = 0;
            virtual void report() = 0;
            virtual bool is_complete() = 0;
    };

    std::vector<storage_base*> storage_arr;

    template<typename T>
    void addStorage(T *t){
        storage_arr.push_back(t);
    }

    template<typename Selection>
    class async_waiter : public storage_base
    {
        sycl::event e_;
        Selection* s;
        std::optional<std::chrono::steady_clock::time_point> timing;

      public:
        async_waiter(sycl::event e) : e_(e){}
        async_waiter(sycl::event e, Selection* selection, std::optional<std::chrono::steady_clock::time_point> t=std::nullopt) : e_(e), s(selection), timing(t) {}
        sycl::event
        unwrap()
        {
            return e_;
        }

        void
        wait() override
        {
            e_.wait();
        }

        void
        report() override{
            if constexpr (report_value_v<Selection, execution_info::task_time_t>){
                if (!timing.has_value())
                {
                    cl_ulong time_start = e_.template get_profiling_info<sycl::info::event_profiling::command_start>();
                    cl_ulong time_end = e_.template get_profiling_info<sycl::info::event_profiling::command_end>();
                    s->report(execution_info::task_time, time_end-time_start);
                }else{
                    auto t = timing.value();
                    s->report(execution_info::task_time, (std::chrono::steady_clock::now() - t).count());
                }
            }

        }

        bool
        is_complete() override{
            return e_.get_info<sycl::info::event::command_execution_status>() == sycl::info::event_command_status::complete;
        }

    };


    class submission_group
    {
        resource_container_t resources_;

      public:
        submission_group(const resource_container_t& v) : resources_(v) {}

        void
        wait()
        {
            for (auto& r : resources_)
            {
                unwrap(r).wait();
            }
        }
    };

  public:
    sycl_backend(const sycl_backend& v) = delete;
    sycl_backend&
    operator=(const sycl_backend&) = delete;

    sycl_backend()
    {
        initialize_default_resources();
        sgroup_ptr_ = std::make_unique<submission_group>(global_rank_);
        number_of_resources=global_rank_.size();
    }

    template <typename NativeUniverseVector>
    sycl_backend(const NativeUniverseVector& v)
    {
        global_rank_.reserve(v.size());
        for (auto e : v)
        {
            global_rank_.push_back(e);
        }
        sgroup_ptr_ = std::make_unique<submission_group>(global_rank_);
    }

    template <typename SelectionHandle, typename Function, typename... Args>
    auto
    submit(SelectionHandle s, Function&& f, Args&&... args)
    {
        auto q = unwrap(s);
        if constexpr (report_info_v<SelectionHandle, execution_info::task_submission_t>)
        {
            report(s, execution_info::task_submission);
        }
        if constexpr (report_info_v<SelectionHandle, execution_info::task_completion_t> ||
                      report_value_v<SelectionHandle, execution_info::task_time_t>)
        {
            std::chrono::steady_clock::time_point t0;
            bool use_event_profiling = q.template has_property<sycl::property::queue::enable_profiling>();
            if constexpr (report_value_v<SelectionHandle, execution_info::task_time_t>)
            {
                if (!use_event_profiling)
                {
                    t0 = std::chrono::steady_clock::now();
                }
            }
            auto e1 = f(q, std::forward<Args>(args)...);
            if constexpr(report_info_v<SelectionHandle, execution_info::task_completion_t>){
                auto e2 = q.submit([=](sycl::handler& h){
                    h.depends_on(e1);
                    h.host_task([=](){
                        s.report(execution_info::task_completion);
                    });
                });
                return async_waiter{e2, new SelectionHandle(s)};
            }
            if constexpr(report_value_v<SelectionHandle, execution_info::task_time_t>){
                if (use_event_profiling)
                {
                    auto waiter = async_waiter{e1, new SelectionHandle(s)};
                    addStorage(new async_waiter(waiter));
                    return waiter;
                }
                else{
                    auto waiter = async_waiter{e1, new SelectionHandle(s), t0};
                    addStorage(new async_waiter(waiter));
                    return waiter;
                }
            }
        }
        else
        {
            return async_waiter{f(unwrap(s), std::forward<Args>(args)...), new SelectionHandle(s)};
        }
    }

    auto
    get_submission_group()
    {
        return *sgroup_ptr_;
    }

    auto
    get_resources()
    {
        return global_rank_;
    }

    void lazy_report(){
        int size = storage_arr.size();
        for(auto i = storage_arr.begin(); i!=storage_arr.begin()+size; i++){
            if((*i)->is_complete()){
                (*i)->report();
                storage_arr.erase(i);
            }
        }
    }
  private:
    std::atomic<int> number_of_resources;;
    resource_container_t global_rank_;
    std::unique_ptr<submission_group> sgroup_ptr_;

    void
    initialize_default_resources()
    {
        auto devices = sycl::device::get_devices();
        for (auto x : devices)
        {
            global_rank_.push_back(sycl::queue{x});
        }
    }
};

} //namespace experimental
} //namespace dpl
} //namespace oneapi

#endif /*_ONEDPL_SYCL_BACKEND_IMPL_H*/
