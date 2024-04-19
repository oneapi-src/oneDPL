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
test_submit_and_wait_on_event(UniverseContainer u, UniverseMapping map, std::vector<int> actual, int best_resource)
{

    using my_policy_t = Policy;
    std::vector<int> result(u.size(), 0);

    // they are cpus so this is ok
    double* v = sycl::malloc_shared<double>(1000000, u[0]);

    my_policy_t p{u};
    auto n_samples = u.size();

    const int N = 10;
    bool pass = false;

    std::mutex m;
    std::mutex m1;
    std::atomic<int> count=-1;
    auto func = [&](typename oneapi::dpl::experimental::policy_traits<Policy>::resource_type e) {
        int x = map[e];
        int j=0;
        result[x]++;
        if (count < 2 * n_samples && count % n_samples != best_resource)
        {
            std::cout<<"Chosen device\n";
            return e.submit([=](sycl::handler& h){
                            h.single_task([](){});
                         });
        }
        else
        {
            std::cout<<"Real device\n";
            return e.submit([=](sycl::handler& h) {
                h.parallel_for<TestUtils::unique_kernel_name<class tune1, 0>>(
                    1000000, [=](sycl::id<1> idx) {
                        for (int j0 = 0; j0 < 100; ++j0)
                        {
                            v[idx] += idx;
                        }
                    });
            });
        }
    };

    auto thread_func = [&](){
        //for(int i=0;i<5;i++){
            count.fetch_add(1, std::memory_order_release);
            auto s = oneapi::dpl::experimental::select(p, func);
            std::lock_guard<std::mutex> lg(m);
            {
                auto w = oneapi::dpl::experimental::submit(s, func);
                oneapi::dpl::experimental::wait(w);
            }
       // }
    };

    std::vector<std::thread> threads;
    for(int i=0;i<10;i++){
        threads.push_back(std::thread(thread_func));
    }

    for(auto& thread : threads){
        thread.join();
    }

    for(auto x : result){
        std::cout<<x<<"\t";
    }

    EXPECT_TRUE(actual==result, "ERROR : did not select expected resources\n");
    std::cout<<"Submit and wait on event : OK\n";
    return 0;
}

static inline void
build_auto_tune_universe(std::vector<sycl::queue>& u, std::unordered_map<sycl::queue, int>& map)
{

    try
    {
        auto device_cpu1 = sycl::device(sycl::cpu_selector_v);
        sycl::queue cpu1_queue{device_cpu1};
        u.push_back(cpu1_queue);
        map[cpu1_queue] = 0;
    }
    catch (const sycl::exception&)
    {
        std::cout << "SKIPPED: Unable to run with cpu_selector\n";
    }
    try
    {
        auto device_cpu2 = sycl::device(sycl::cpu_selector_v);
        sycl::queue cpu2_queue{device_cpu2};
        u.push_back(cpu2_queue);
        map[cpu2_queue] = 1;
    }
    catch (const sycl::exception&)
    {
        std::cout << "SKIPPED: Unable to run with cpu_selector\n";
    }
    try
    {
        auto device_cpu3 = sycl::device(sycl::cpu_selector_v);
        sycl::queue cpu3_queue{device_cpu3};
        u.push_back(cpu3_queue);
        map[cpu3_queue] = 2;
    }
    catch (const sycl::exception&)
    {
        std::cout << "SKIPPED: Unable to run with cpu_selector\n";
    }
    try
    {
        auto device_cpu4 = sycl::device(sycl::cpu_selector_v);
        sycl::queue cpu4_queue{device_cpu4};
        u.push_back(cpu4_queue);
        map[cpu4_queue] = 3;
    }
    catch (const sycl::exception&)
    {
        std::cout << "SKIPPED: Unable to run with cpu_selector\n";
    }
}

static inline auto
build_result(int universe_size, int count, int offset=0){
    std::vector<int> result(universe_size, 0);
    for(int i=0;i<universe_size*2;i++){
        result[i%universe_size]++;
    }
    result[offset]+=count-universe_size*2;
    std::cout<<"Printing result\n";
    for(auto x : result){
        std::cout<<x<<"\t";
    }
    std::cout<<"\n";
    return result;
}
int
main()
{
    bool bProcessed = false;

    using policy_t = oneapi::dpl::experimental::auto_tune_policy<oneapi::dpl::experimental::sycl_backend>;
    std::vector<sycl::queue> u;
    std::unordered_map<sycl::queue, int> map;
    build_auto_tune_universe(u, map);
    int constexpr count = 10;
    int actual;
    if (!u.empty())
    {
        actual = test_submit_and_wait_on_event<policy_t>(u, map, build_result(u.size(), count, 0), 0);
        /*actual = test_submit_and_wait_on_event<policy_t>(u, map, build_result(u.size(), count, 1), 1);
        actual = test_submit_and_wait_on_event<policy_t>(u, map, build_result(u.size(), count, 2), 2);
        actual = test_submit_and_wait_on_event<policy_t>(u, map, build_result(u.size(), count, 3), 3);
*/
        bProcessed = true;
    }
        return TestUtils::done(bProcessed);
       /* actual = test_submit_and_wait_on_event<policy_t>(u, map, build_result(u.size(), 50, 1), 1);
        actual = test_submit_and_wait_on_event<policy_t>(u, map, build_result(u.size(), 50, 2), 2);
        actual = test_submit_and_wait_on_group<policy_t>(u, map, build_result(u.size(), 50, 0), 0);
        actual = test_submit_and_wait_on_group<policy_t>(u, map, build_result(u.size(), 50, 0), 1);
        actual = test_submit_and_wait_on_group<policy_t>(u, map, build_result(u.size(), 50, 1), 2);
        actual = test_submit_and_wait<policy_t>(u, map, build_result(u.size(), 50, 2), 0);
        actual = test_submit_and_wait<policy_t>(u, map, build_result(u.size(), 50, 1), 1);
        actual = test_submit_and_wait<policy_t>(u, map, build_result(u.size(), 50, 2), 2);
    }*/
}
