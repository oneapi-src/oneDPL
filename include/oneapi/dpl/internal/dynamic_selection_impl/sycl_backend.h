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

    bool has_enable_profiling = false;
  private:
    class async_waiter_base{
        public:
            virtual void wait() = 0;
            virtual void report() = 0;
            virtual bool is_complete() = 0;
    };

    template<typename Selection>
    class async_waiter : public async_waiter_base
    {
        sycl::event e_;
        Selection* s;
      public:
        async_waiter(sycl::event e) : e_(e){}
        async_waiter(sycl::event e, Selection* selection) : e_(e), s(selection) {}

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
                cl_ulong time_start = e_.template get_profiling_info<sycl::info::event_profiling::command_start>();
                cl_ulong time_end = e_.template get_profiling_info<sycl::info::event_profiling::command_end>();
                s->report(execution_info::task_time, time_end-time_start);
            }

        }

        bool
        is_complete() override{
            return e_.get_info<sycl::info::event::command_execution_status>() == sycl::info::event_command_status::complete;
        }

    };

    struct async_waiter_list_t{

        std::mutex m_;
        std::vector<async_waiter_base*> async_waiters;

        template<typename T>
        void add_waiter(T *t){
            std::lock_guard<std::mutex> l(m_);
            async_waiters.push_back(t);
        }

        void lazy_report(){
            std::lock_guard<std::mutex> l(m_);
            int size = async_waiters.size();
            for(auto i = async_waiters.begin(); i!=async_waiters.begin()+size; i++){
                if((*i)->is_complete()){
                    (*i)->report();
                    async_waiters.erase(i);
                }
            }
        }
    };

    async_waiter_list_t async_waiter_list;


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
    }

    template <typename NativeUniverseVector>
    sycl_backend(const NativeUniverseVector& v)
    {
        global_rank_.reserve(v.size());
        for (auto e : v)
        {
            global_rank_.push_back(e);
            if(e.template has_property<sycl::property::queue::enable_profiling>()){
                has_enable_profiling = true;
            }
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
            else if constexpr(report_value_v<SelectionHandle, execution_info::task_time_t>){
                if (use_event_profiling)
                {
                    auto waiter = async_waiter{e1, new SelectionHandle(s)};
                    async_waiter_list.add_waiter(new async_waiter(waiter));
                    return waiter;
                }
                else{
                    auto e2 = q.submit([=](sycl::handler& h){
                        h.depends_on(e1);
                        h.host_task([=](){
                            s.report(execution_info::task_time, (std::chrono::steady_clock::now() - t0).count());
                        });
                    });
                    return async_waiter{e2, new SelectionHandle(s)};
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
        async_waiter_list.lazy_report();
    }

  private:
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
