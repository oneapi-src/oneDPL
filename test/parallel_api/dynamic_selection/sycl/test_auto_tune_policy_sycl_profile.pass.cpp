// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <iostream>
#include <thread>
#include "oneapi/dpl/dynamic_selection"
#include "support/test_dynamic_selection_utils.h"
#include "support/test_config.h"
#if TEST_DYNAMIC_SELECTION_AVAILABLE
#    include "support/sycl_sanity.h"

template <bool call_select_before_submit, typename Policy, typename UniverseContainer>
int
test_auto_submit_wait_on_event(UniverseContainer u, int best_resource)
{
    using my_policy_t = Policy;

    int j;
    std::vector<double> v(1000000, 0.0);

    my_policy_t p{u};
    auto n_samples = u.size();

    const int N = 10;
    std::atomic<int> ecount = 0;
    bool pass = true;

    for (int i = 1; i <= N; ++i)
    {
        if (i <= 2 * n_samples && (i - 1) % n_samples != best_resource)
        {
            j = 100;
        }
        else
        {
            j = 0;
        }
        const size_t bytes = 1000000 * sizeof(double);
        if constexpr (call_select_before_submit)
        {
            auto f = [&](typename oneapi::dpl::experimental::policy_traits<Policy>::resource_type q) {
                if (i <= 2 * n_samples)
                {
                    // we should be round-robining through the resources
                    if (q != u[(i - 1) % n_samples])
                    {
                        std::cout << i << ": mismatch during rr phase\n" << std::flush;
                        pass = false;
                    }
                }
                else
                {
                    if (q != u[best_resource])
                    {
                        std::cout << i << ": mismatch during prod phase " << best_resource << "\n" << std::flush;
                        pass = false;
                    }
                }
                ecount += i;
                if (j == 0)
                {
                     return q.submit([&](sycl::handler& h)
                     {
                             h.single_task([=](){});
                         });
                }
                else
                {
                    return q.submit([&](sycl::handler& h) {
                        double *d_v = sycl::malloc_device<double>(1000000, q);
                        q.memcpy(d_v, v.data(), bytes).wait();
                        q.parallel_for(
                            1000000, [=](sycl::id<1> idx) {
                                for (int j0 = 0; j0 < j; ++j0)
                                {
                                    d_v[idx] += idx;
                                }
                            });
                        q.memcpy(v.data(), d_v, bytes).wait();
                    });
                }
            };
            auto s = oneapi::dpl::experimental::select(p, f);
            auto e = oneapi::dpl::experimental::submit(s, f);
            oneapi::dpl::experimental::wait(e);
        }
        else
        {
            // it's ok to capture by reference since we are waiting on each call
            auto s = oneapi::dpl::experimental::submit(
                p, [&](typename oneapi::dpl::experimental::policy_traits<Policy>::resource_type q) {
                    if (i <= 2 * n_samples)
                    {
                        // we should be round-robining through the resources
                        if (q != u[(i - 1) % n_samples])
                        {
                            std::cout << i << ": mismatch during rr phase\n" << std::flush;
                            pass = false;
                        }
                    }
                    else
                    {
                        if (q != u[best_resource])
                        {
                            std::cout << i << ": mismatch during prod phase " << best_resource << "\n" << std::flush;
                            pass = false;
                        }
                    }
                    ecount += i;
                    if (j == 0)
                    {
                     std::cout<<"Device 0\n";
                     return q.submit([&](sycl::handler& h){
                             h.single_task([=](){});
                         });
                    }
                    else
                    {
                        std::cout<<"Device 1\n";
                        return q.submit([&](sycl::handler& h) {
                            double *d_v = sycl::malloc_device<double>(1000000, q);
                            q.memcpy(d_v, v.data(), bytes).wait();
                            q.parallel_for(
                                1000000, [=](sycl::id<1> idx) {
                                    for (int j0 = 0; j0 < j; ++j0)
                                    {
                                        d_v[idx] += idx;
                                    }
                                });
                            q.memcpy(v.data(), d_v, bytes).wait();
                        });
                    }
                });
            oneapi::dpl::experimental::wait(s);
        }

        int count = ecount.load();
        if (count != i * (i + 1) / 2)
        {
            std::cout << "ERROR: scheduler did not execute all tasks exactly once\n";
            return 1;
        }
    }
    if (!pass)
    {
        std::cout << "ERROR: did not select expected resources\n";
        return 1;
    }
    if constexpr (call_select_before_submit)
    {
        std::cout << "select then submit and wait on event: OK\n";
    }
    else
    {
        std::cout << "submit and wait on event: OK\n";
    }
    return 0;
}

template <bool call_select_before_submit, typename Policy, typename UniverseContainer>
int
test_auto_submit_wait_on_group(UniverseContainer u, int best_resource)
{
    using my_policy_t = Policy;

    int j;
    std::vector<double> v(1000000, 0.0);

    my_policy_t p{u};
    auto n_samples = u.size();

    const int N = 10;
    std::atomic<int> ecount = 0;
    bool pass = true;

    for (int i = 1; i <= N; ++i)
    {
        if (i <= 2 * n_samples && (i - 1) % n_samples != best_resource)
        {
            j = 100;
        }
        else
        {
            j = 0;
        }
        const size_t bytes = 1000000 * sizeof(double);
        if constexpr (call_select_before_submit)
        {
            auto f = [&](typename oneapi::dpl::experimental::policy_traits<Policy>::resource_type q) {
                if (i <= 2 * n_samples)
                {
                    // we should be round-robining through the resources
                    if (q != u[(i - 1) % n_samples])
                    {
                        std::cout << i << ": mismatch during rr phase\n" << std::flush;
                        pass = false;
                    }
                }
                else
                {
                    if (q != u[best_resource])
                    {
                        std::cout << i << ": mismatch during prod phase " << best_resource << "\n" << std::flush;
                        pass = false;
                    }
                }
                ecount += i;
                if (j == 0)
                {
                     return q.submit([=](sycl::handler& h){
                            h.single_task([=](){});
                         });
                }
                else
                {
                    return q.submit([&](sycl::handler& h) {
                            double *d_v = sycl::malloc_device<double>(1000000, q);
                            q.memcpy(d_v, v.data(), bytes).wait();
                            h.parallel_for(
                            1000000, [=](sycl::id<1> idx) {
                                for (int j0 = 0; j0 < j; ++j0)
                                {
                                    d_v[idx] += idx;
                                }
                            });
                            q.memcpy(v.data(), d_v, bytes).wait();
                    });
                }
            };
            auto s = oneapi::dpl::experimental::select(p, f);
            auto e = oneapi::dpl::experimental::submit(s, f);
            oneapi::dpl::experimental::wait(p.get_submission_group());
        }
        else
        {
            // it's ok to capture by reference since we are waiting on each call
            auto s = oneapi::dpl::experimental::submit(
                p, [&](typename oneapi::dpl::experimental::policy_traits<Policy>::resource_type q) {
                    if (i <= 2 * n_samples)
                    {
                        // we should be round-robining through the resources
                        if (q != u[(i - 1) % n_samples])
                        {
                            std::cout << i << ": mismatch during rr phase\n" << std::flush;
                            pass = false;
                        }
                    }
                    else
                    {
                        if (q != u[best_resource])
                        {
                            std::cout << i << ": mismatch during prod phase " << best_resource << "\n" << std::flush;
                            pass = false;
                        }
                    }
                    ecount += i;
                    if (j == 0)
                    {
                     return q.submit([=](sycl::handler& h){
                             h.single_task([=](){});
                         });
                    }
                    else
                    {
                        return q.submit([&](sycl::handler& h) {
                            double *d_v = sycl::malloc_device<double>(1000000, q);
                            q.memcpy(d_v, v.data(), bytes).wait();
                            h.parallel_for(
                                1000000, [=](sycl::id<1> idx) {
                                    for (int j0 = 0; j0 < j; ++j0)
                                    {
                                        d_v[idx] += idx;
                                    }
                                });
                            q.memcpy(v.data(), d_v, bytes).wait();
                        });
                    }
                });
            oneapi::dpl::experimental::wait(p.get_submission_group());
        }

        int count = ecount.load();
        if (count != i * (i + 1) / 2)
        {
            std::cout << "ERROR: scheduler did not execute all tasks exactly once\n";
            return 1;
        }
    }
    if (!pass)
    {
        std::cout << "ERROR: did not select expected resources\n";
        return 1;
    }
    if constexpr (call_select_before_submit)
    {
        std::cout << "select then submit and wait on group: OK\n";
    }
    else
    {
        std::cout << "submit and wait on group: OK\n";
    }
    return 0;
}

template <bool call_select_before_submit, typename Policy, typename UniverseContainer>
int
test_auto_submit_and_wait(UniverseContainer u, int best_resource)
{
    using my_policy_t = Policy;

    // they are cpus so this is ok
    int j;
    std::vector<double> v(1000000, 0.0);

    my_policy_t p{u};
    auto n_samples = u.size();

    const int N = 10;
    std::atomic<int> ecount = 0;
    bool pass = true;

    for (int i = 1; i <= N; ++i)
    {
        if (i <= 2 * n_samples && (i - 1) % n_samples != best_resource)
        {
            j = 100;
        }
        else
        {
            j = 0;
        }
        const size_t bytes = 1000000 * sizeof(double);
        if constexpr (call_select_before_submit)
        {
            auto f = [&](typename oneapi::dpl::experimental::policy_traits<Policy>::resource_type q) {
                if (i <= 2 * n_samples)
                {
                    // we should be round-robining through the resources
                    if (q != u[(i - 1) % n_samples])
                    {
                        std::cout << i << ": mismatch during rr phase\n" << std::flush;
                        pass = false;
                    }
                }
                else
                {
                    if (q != u[best_resource])
                    {
                        std::cout << i << ": mismatch during prod phase " << best_resource << "\n" << std::flush;
                        pass = false;
                    }
                }
                ecount += i;
                if (j == 0)
                {
                     return q.submit([=](sycl::handler& h){
                             h.single_task([=](){});
                         });
                }
                else
                {
                    return q.submit([&](sycl::handler& h) {
                        double *d_v = sycl::malloc_device<double>(1000000, q);
                        q.memcpy(d_v, v.data(), bytes).wait();
                        h.parallel_for(
                            1000000, [=](sycl::id<1> idx) {
                                for (int j0 = 0; j0 < j; ++j0)
                                {
                                    d_v[idx] += idx;
                                }
                            });
                            q.memcpy(v.data(), d_v, bytes).wait();
                    });
                }
            };
            auto s = oneapi::dpl::experimental::select(p, f);
            oneapi::dpl::experimental::submit_and_wait(s, f);
        }
        else
        {
            // it's ok to capture by reference since we are waiting on each call
            oneapi::dpl::experimental::submit_and_wait(
                p, [&](typename oneapi::dpl::experimental::policy_traits<Policy>::resource_type q) {
                    if (i <= 2 * n_samples)
                    {
                        // we should be round-robining through the resources
                        if (q != u[(i - 1) % n_samples])
                        {
                            std::cout << i << ": mismatch during rr phase\n" << std::flush;
                            pass = false;
                        }
                    }
                    else
                    {
                        if (q != u[best_resource])
                        {
                            std::cout << i << ": mismatch during prod phase " << best_resource << "\n" << std::flush;
                            pass = false;
                        }
                    }
                    ecount += i;
                    if (j == 0)
                    {
                     return q.submit([=](sycl::handler& h){
                             h.single_task([=](){});
                         });
                    }
                    else
                    {
                        return q.submit([&](sycl::handler& h) {
                            double *d_v = sycl::malloc_device<double>(1000000, q);
                            q.memcpy(d_v, v.data(), bytes).wait();
                            h.parallel_for(
                                1000000, [=](sycl::id<1> idx) {
                                    for (int j0 = 0; j0 < j; ++j0)
                                    {
                                        d_v[idx] += idx;
                                    }
                                });
                            q.memcpy(v.data(), d_v, bytes).wait();
                        });
                    }
                });
        }

        int count = ecount.load();
        if (count != i * (i + 1) / 2)
        {
            std::cout << "ERROR: scheduler did not execute all tasks exactly once\n";
            return 1;
        }
    }
    if (!pass)
    {
        std::cout << "ERROR: did not select expected resources\n";
        return 1;
    }
    if constexpr (call_select_before_submit)
    {
        std::cout << "select then submit_and_wait: OK\n";
    }
    else
    {
        std::cout << "submit_and_wait: OK\n";
    }
    return 0;
}

static inline void
build_auto_tune_universe(std::vector<sycl::queue>& u)
{
    auto prop_list = sycl::property_list{sycl::property::queue::enable_profiling()};
    try
    {
        auto device_cpu = sycl::device(sycl::cpu_selector_v);
        sycl::queue cpu_queue{device_cpu, prop_list};
        run_sycl_sanity_test(cpu_queue);
        u.push_back(cpu_queue);
    }
    catch (const sycl::exception&)
    {
        std::cout << "SKIPPED: Unable to run with cpu_selector\n";
    }
    try
    {
        auto device_gpu = sycl::device(sycl::gpu_selector_v);
        sycl::queue gpu_queue{device_gpu, prop_list};
        run_sycl_sanity_test(gpu_queue);
        u.push_back(gpu_queue);
    }
    catch (const sycl::exception&)
    {
        std::cout << "SKIPPED: Unable to run with gpu_selector\n";
    }
}

#endif

int
main()
{
#if TEST_DYNAMIC_SELECTION_AVAILABLE
    using policy_t = oneapi::dpl::experimental::auto_tune_policy<oneapi::dpl::experimental::sycl_backend>;
    std::vector<sycl::queue> u;
    build_auto_tune_universe(u);

    //If building the universe is not a success, return
    if (u.size() == 0 || u.size()==0)
        return 0;

    constexpr bool just_call_submit = false;
    constexpr bool call_select_before_submit = true;

    if (test_auto_submit_wait_on_event<just_call_submit, policy_t>(u, 0) /*||
        test_auto_submit_wait_on_event<just_call_submit, policy_t>(u, 1) ||
        test_auto_submit_wait_on_event<just_call_submit, policy_t>(u, 0) ||
        test_auto_submit_wait_on_event<just_call_submit, policy_t>(u, 1) ||
        test_auto_submit_wait_on_group<just_call_submit, policy_t>(u, 0) ||
        test_auto_submit_wait_on_group<just_call_submit, policy_t>(u, 1) ||
        test_auto_submit_wait_on_group<just_call_submit, policy_t>(u, 0) ||
        test_auto_submit_wait_on_group<just_call_submit, policy_t>(u, 1) ||
         test_auto_submit_and_wait<just_call_submit, policy_t>(u, 0) ||
        test_auto_submit_and_wait<just_call_submit, policy_t>(u, 1)||
        test_auto_submit_and_wait<just_call_submit, policy_t>(u, 0) ||
        test_auto_submit_and_wait<just_call_submit, policy_t>(u, 1) ||
        // now select then submits
        test_auto_submit_wait_on_event<call_select_before_submit, policy_t>(u, 0) ||
        test_auto_submit_wait_on_event<call_select_before_submit, policy_t>(u, 1) ||
        test_auto_submit_wait_on_event<call_select_before_submit, policy_t>(u, 0) ||
        test_auto_submit_wait_on_event<call_select_before_submit, policy_t>(u, 1) ||
        test_auto_submit_wait_on_group<call_select_before_submit, policy_t>(u, 0) ||
        test_auto_submit_wait_on_group<call_select_before_submit, policy_t>(u, 1) ||
        test_auto_submit_wait_on_group<call_select_before_submit, policy_t>(u, 0) ||
        test_auto_submit_wait_on_group<call_select_before_submit, policy_t>(u, 1) ||
        test_auto_submit_and_wait<call_select_before_submit, policy_t>(u, 0) ||
        test_auto_submit_and_wait<call_select_before_submit, policy_t>(u, 1) ||
        test_auto_submit_and_wait<call_select_before_submit, policy_t>(u, 0) ||
        test_auto_submit_and_wait<call_select_before_submit, policy_t>(u, 1)*/)
    {
        std::cout << "FAIL\n";
        return 1;
    }
    else
    {
        std::cout << "PASS\n";
        return 0;
    }
#else
    std::cout << "SKIPPED\n";
    return 0;
#endif // TEST_DYNAMIC_SELECTION_AVAILABLE
}
