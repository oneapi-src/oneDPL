// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "support/test_config.h"


#include "oneapi/dpl/dynamic_selection"
#include "support/utils.h"
#include <unordered_map>
#include <thread>

template <typename Policy, typename UniverseContainer, typename UniverseMapping>
int
test_submit_and_wait(UniverseContainer u, UniverseMapping map, std::vector<int> actual, int offset = 0)
{
    using my_policy_t = Policy;
    my_policy_t p(u, offset);
    std::vector<int> result(u.size(), 0);
    const int N = 5;
    std::atomic<int> ecount=0;
    bool pass = true;

    std::mutex m;
    auto func = [&result,&map, &m, &ecount](typename oneapi::dpl::experimental::policy_traits<Policy>::resource_type e) {
        int x = map[e];
        result[x]++;
        ecount++;

        if constexpr (std::is_same_v<typename oneapi::dpl::experimental::policy_traits<Policy>::resource_type,
                                     int>)
            return e;
        else
            return typename oneapi::dpl::experimental::policy_traits<Policy>::wait_type{};
    };
    auto thread_func = [&p, &func, &m, &ecount](){
        for(int i=0;i<10;i++){
            //auto s = oneapi::dpl::experimental::select(p);

            std::lock_guard<std::mutex> lg(m);
            oneapi::dpl::experimental::submit_and_wait(p, func);
        }
    };

    std::vector<std::thread> threads;
    for(int i=0;i<5;i++){
        threads.push_back(std::thread(thread_func));
    }

    for(auto& thread : threads){
        thread.join();
    }

    oneapi::dpl::experimental::wait(p.get_submission_group());
    for(auto x : result){
        std::cout<<x<<"\t";
    }

    std::cout<<"\n"<<ecount<<"\n";
    return actual == result;
}
template <typename Policy, typename UniverseContainer, typename UniverseMapping>
int
test_submit_and_wait_on_group(UniverseContainer u, UniverseMapping map, std::vector<int> actual, int offset = 0)
{
    using my_policy_t = Policy;
    my_policy_t p(u, offset);
    std::vector<int> result(u.size(), 0);
    const int N = 5;
    std::atomic<int> ecount=0;
    bool pass = true;

    std::mutex m;
    auto func = [&result,&map, &m, &ecount](typename oneapi::dpl::experimental::policy_traits<Policy>::resource_type e) {
        int x = map[e];
        result[x]++;
        ecount++;

        if constexpr (std::is_same_v<typename oneapi::dpl::experimental::policy_traits<Policy>::resource_type,
                                     int>)
            return e;
        else
            return typename oneapi::dpl::experimental::policy_traits<Policy>::wait_type{};
    };
    auto thread_func = [&p, &func, &m, &ecount](){
        for(int i=0;i<10;i++){
            auto s = oneapi::dpl::experimental::select(p);

            std::lock_guard<std::mutex> lg(m);
            auto w = oneapi::dpl::experimental::submit(s, func);
        }
    };

    std::vector<std::thread> threads;
    for(int i=0;i<5;i++){
        threads.push_back(std::thread(thread_func));
    }

    for(auto& thread : threads){
        thread.join();
    }

    oneapi::dpl::experimental::wait(p.get_submission_group());
    for(auto x : result){
        std::cout<<x<<"\t";
    }

    std::cout<<"\n"<<ecount<<"\n";
    return actual == result;
}
template <typename Policy, typename UniverseContainer, typename UniverseMapping>
int
test_submit_and_wait_on_event(UniverseContainer u, UniverseMapping map, std::vector<int> actual, int offset = 0)
{
    using my_policy_t = Policy;
    my_policy_t p(u, offset);
    std::vector<int> result(u.size(), 0);
    const int N = 5;
    std::atomic<int> ecount=0;
    bool pass = true;

    std::mutex m;
    auto func = [&result,&map, &m, &ecount](typename oneapi::dpl::experimental::policy_traits<Policy>::resource_type e) {
        int x = map[e];
        result[x]++;
        ecount++;

        if constexpr (std::is_same_v<typename oneapi::dpl::experimental::policy_traits<Policy>::resource_type,
                                     int>)
            return e;
        else
            return typename oneapi::dpl::experimental::policy_traits<Policy>::wait_type{};
    };
    auto thread_func = [&p, &func, &m, &ecount](){
        for(int i=0;i<10;i++){
            auto s = oneapi::dpl::experimental::select(p);

            std::lock_guard<std::mutex> lg(m);
            auto w = oneapi::dpl::experimental::submit(s, func);
            oneapi::dpl::experimental::wait(w);
        }
    };

    std::vector<std::thread> threads;
    for(int i=0;i<5;i++){
        threads.push_back(std::thread(thread_func));
    }

    for(auto& thread : threads){
        thread.join();
    }

    for(auto x : result){
        std::cout<<x<<"\t";
    }

    std::cout<<"\n"<<ecount<<"\n";
    return actual==result;
}
static inline void
build_universe(std::vector<sycl::queue>& u, std::unordered_map<sycl::queue, int>& map)
{
    try
    {
        auto device_default = sycl::device(sycl::default_selector_v);
        sycl::queue default_queue(device_default);
        u.push_back(default_queue);
        map[default_queue] = 0;
    }
    catch (const sycl::exception&)
    {
        std::cout << "SKIPPED: Unable to run with default_selector\n";
    }

    try
    {
        auto device_gpu = sycl::device(sycl::cpu_selector_v);
        //auto policy_a = oneapi::dpl::execution::device_policy{device_gpu};
        sycl::queue gpu_queue(device_gpu);
        u.push_back(gpu_queue);
        map[gpu_queue] = 1;
    }
    catch (const sycl::exception&)
    {
        std::cout << "SKIPPED: Unable to run with gpu_selector\n";
    }

    try
    {
        auto device_cpu = sycl::device(sycl::cpu_selector_v);
        //auto policy_b = oneapi::dpl::execution::device_policy{device_cpu};
        sycl::queue cpu_queue(device_cpu);
        u.push_back(cpu_queue);
        map[cpu_queue] = 2;
    }
    catch (const sycl::exception&)
    {
        std::cout << "SKIPPED: Unable to run with cpu_selector\n";
    }
}

static inline auto
build_result(int universe_size, int count, int offset=0){
    std::vector<int> result(universe_size, 0);
    result[offset]=count;
    return result;
}
int
main()
{
    bool bProcessed = false;

    using policy_t = oneapi::dpl::experimental::fixed_resource_policy<oneapi::dpl::experimental::sycl_backend>;
    std::vector<sycl::queue> u;
    std::unordered_map<sycl::queue, int> map;
    build_universe(u, map);
    int constexpr count = 50;
    int actual;
    if (!u.empty())
    {
        actual = test_submit_and_wait_on_event<policy_t>(u, map, build_result(u.size(), 50, 0), 0);
        actual = test_submit_and_wait_on_event<policy_t>(u, map, build_result(u.size(), 50, 1), 1);
        actual = test_submit_and_wait_on_event<policy_t>(u, map, build_result(u.size(), 50, 2), 2);
        actual = test_submit_and_wait_on_group<policy_t>(u, map, build_result(u.size(), 50, 0), 0);
        actual = test_submit_and_wait_on_group<policy_t>(u, map, build_result(u.size(), 50, 0), 1);
        actual = test_submit_and_wait_on_group<policy_t>(u, map, build_result(u.size(), 50, 1), 2);
        actual = test_submit_and_wait<policy_t>(u, map, build_result(u.size(), 50, 2), 0);
        actual = test_submit_and_wait<policy_t>(u, map, build_result(u.size(), 50, 1), 1);
        actual = test_submit_and_wait<policy_t>(u, map, build_result(u.size(), 50, 2), 2);
    }
}
