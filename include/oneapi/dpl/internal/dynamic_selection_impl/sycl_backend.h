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
#include <ratio>
#include <vector>
#include <memory>
#include <utility>
#include <algorithm>

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

    using report_clock_type = std::chrono::steady_clock;
    using report_duration = std::chrono::milliseconds;

    static inline bool is_profiling_enabled = false;
  private:
    class async_waiter_base{
        public:
            virtual void wait() = 0;
            virtual void report() = 0;
            virtual bool is_complete() = 0;
            virtual ~async_waiter_base() = default;
    };

    template<typename Selection>
    class async_waiter : public async_waiter_base
    {
        sycl::event e_;
        std::shared_ptr<Selection> s;
      public:
        async_waiter(sycl::event e) : e_(e) {}
        async_waiter(sycl::event e, std::shared_ptr<Selection> selection) : e_(e), s(selection) {}

        async_waiter(async_waiter &w) : e_(w.e_), s(w.s) {}
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
                if(s!=nullptr){
                    const auto duration_in_ns = std::chrono::nanoseconds(time_end-time_start);
                    s->report(execution_info::task_time, std::chrono::duration_cast<report_duration>(duration_in_ns));
                }
            }


        }

        bool
        is_complete() override{
            return e_.get_info<sycl::info::event::command_execution_status>() == sycl::info::event_command_status::complete;
        }

    };

    struct async_waiter_list_t{

        std::mutex m_;
        std::vector<std::unique_ptr<async_waiter_base>> async_waiters;

        template<typename T>
        void add_waiter(T *t){
            std::lock_guard<std::mutex> l(m_);
            async_waiters.push_back(std::unique_ptr<T>(t));
        }

        void lazy_report(){
            if(is_profiling_enabled){
                std::lock_guard<std::mutex> l(m_);
                async_waiters.erase(std::remove_if(async_waiters.begin(), async_waiters.end(), [](std::unique_ptr<async_waiter_base>& async_waiter){
                        if(async_waiter->is_complete()){
                            async_waiter->report();
                            return true;
                        }
                        return false;
                    }), async_waiters.end());
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
        bool profiling = true;
        global_rank_.reserve(v.size());
        for (auto e : v)
        {
            global_rank_.push_back(e);
            if(!e.template has_property<sycl::property::queue::enable_profiling>()){
                profiling = false;
            }
        }
        is_profiling_enabled = profiling;
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
            report_clock_type::time_point t0;
            if constexpr (report_value_v<SelectionHandle, execution_info::task_time_t>)
            {
                if (!is_profiling_enabled)
                {
                    t0 = report_clock_type::now();
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
                return async_waiter{e2, std::make_shared<SelectionHandle>(s)};
            }
            if constexpr(report_value_v<SelectionHandle, execution_info::task_time_t>){
                if (is_profiling_enabled)
                {
                    auto waiter = async_waiter{e1,std::make_shared<SelectionHandle>(s)};
                    async_waiter_list.add_waiter(new async_waiter(waiter));
                    return waiter;
                }
                else{
                    auto e2 = q.submit([=](sycl::handler& h){
                        h.depends_on(e1);
                        h.host_task([=](){
                            const auto tp_now = report_clock_type::now();
                            s.report(execution_info::task_time, std::chrono::duration_cast<report_duration>(tp_now - t0));
                        });
                    });
                    return async_waiter{e2, std::make_shared<SelectionHandle>(s)};
                }
            }
        }
        else
        {
            return async_waiter{f(unwrap(s), std::forward<Args>(args)...), std::make_shared<SelectionHandle>(s)};
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
        bool profiling = true;
        auto prop_list = sycl::property_list{};
        auto devices = sycl::device::get_devices();
        for (auto x : devices)
        {
            global_rank_.push_back(sycl::queue{x});
            if(!x.has(sycl::aspect::queue_profiling)){
                profiling = false;
            }
        }
        is_profiling_enabled = profiling;
        if(is_profiling_enabled){
            prop_list = sycl::property_list{sycl::property::queue::enable_profiling()};
        }
        for (auto x : devices)
        {
            global_rank_.push_back(sycl::queue{x, prop_list});
        }
    }
};

} //namespace experimental
} //namespace dpl
} //namespace oneapi

#endif /*_ONEDPL_SYCL_BACKEND_IMPL_H*/
