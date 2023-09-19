// -*- C++ -*-
//===---------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _ONEDPL_TEST_DYNAMIC_SELECTION_UTILS_H
#define _ONEDPL_TEST_DYNAMIC_SELECTION_UTILS_H

#include <thread>
#include <chrono>
#include <random>
#include <algorithm>

template <typename Policy, typename T>
int
test_initialization(const std::vector<T>& u)
{
    // initialize
    using my_policy_t = Policy;
    my_policy_t p{u};
    auto u2 = oneapi::dpl::experimental::get_resources(p);
    auto u2s = u2.size();
    if (!std::equal(std::begin(u2), std::end(u2), std::begin(u)))
    {
        std::cout << "ERROR: provided resources and queried resources are not equal\n";
        return 1;
    }

    // deferred initialization
    my_policy_t p2{oneapi::dpl::experimental::deferred_initialization};
    try
    {
        auto u3 = oneapi::dpl::experimental::get_resources(p2);
        if (!u3.empty())
        {
            std::cout << "ERROR: deferred initialization not respected\n";
            return 1;
        }
    }
    catch (...)
    {
    }
    p2.initialize(u);
    auto u3 = oneapi::dpl::experimental::get_resources(p);
    auto u3s = u3.size();
    if (!std::equal(std::begin(u3), std::end(u3), std::begin(u)))
    {
        std::cout << "ERROR: reported resources and queried resources are not equal after deferred initialization\n";
        return 1;
    }

    std::cout << "initialization: OK\n" << std::flush;
    return 0;
}

template <typename Policy, typename UniverseContainer>
int
test_unwrap(UniverseContainer u)
{
    //Checking if unwrap returns the same type when passed a policy
    using my_policy_t = Policy;
    my_policy_t p{u};
    bool pass = true;
    auto policy_unwrap = oneapi::dpl::experimental::unwrap(p);

    if(!std::is_same_v<decltype(p),decltype(policy_unwrap)>){
        pass=false;
        std::cout << "ERROR: Unwrapped policy type is not equal to the actual policy type\n";
    }


    //Checking if unwrap returns a different type when passed a const submission
    const auto const_submission = oneapi::dpl::experimental::submit(
        p, [](typename oneapi::dpl::experimental::policy_traits<Policy>::resource_type e) {
            if constexpr (std::is_same_v<
                              typename oneapi::dpl::experimental::policy_traits<Policy>::resource_type, int>)
                return e;
            else
                return typename oneapi::dpl::experimental::policy_traits<Policy>::wait_type{};
        });
    const auto const_submission_unwrap = oneapi::dpl::experimental::unwrap(const_submission);

    if(std::is_same_v<decltype(const_submission),decltype(const_submission_unwrap)>){
        pass=false;
        std::cout << "ERROR: Unwrapped const submission type is equal to the actual const submission type\n";
    }

    //Checking if the const unwrapped submission type is the same as const policy::wait_type
    if(!std::is_same_v<decltype(const_submission_unwrap),const typename oneapi::dpl::experimental::policy_traits<Policy>::wait_type>){
        pass=false;
        std::cout << "ERROR: Unwrapped const submission type is not equal to the policy's wait type\n";
    }

    //Checking if unwrap returns a different type when passed a submission
    auto submission = oneapi::dpl::experimental::submit(
        p, [](typename oneapi::dpl::experimental::policy_traits<Policy>::resource_type e) {
            if constexpr (std::is_same_v<
                              typename oneapi::dpl::experimental::policy_traits<Policy>::resource_type, int>)
                return e;
            else
                return typename oneapi::dpl::experimental::policy_traits<Policy>::wait_type{};
        });
    auto submission_unwrap = oneapi::dpl::experimental::unwrap(submission);

    if(std::is_same_v<decltype(submission),decltype(submission_unwrap)>)
    {
        pass=false;
        std::cout << "ERROR: Unwrapped submission type is equal to the actual submission type\n";
    }


    //Checking if the unwrapped submission type is the same as policy::wait_type
    if(!std::is_same_v<decltype(submission_unwrap),typename oneapi::dpl::experimental::policy_traits<Policy>::wait_type>)
    {
        pass=false;
        std::cout << "ERROR: Unwrapped submission type is not equal to the policy's wait type\n";
    }

    auto func = [](){};
    //Checking if unwrap returns a different type when passed a const selection
    const auto const_selection = oneapi::dpl::experimental::select(p, func);
    const auto const_selection_unwrap = oneapi::dpl::experimental::unwrap(const_selection);

    if(std::is_same_v<decltype(const_selection),decltype(const_selection_unwrap)>){
        pass=false;
        std::cout << "ERROR: Unwrapped const selection type is equal to the actual const selection type\n";
    }

    //Checking if the const unwrapped selection type is the same as const policy::resource_type
    if(!std::is_same_v<decltype(const_selection_unwrap),const typename oneapi::dpl::experimental::policy_traits<Policy>::resource_type >){
        pass=false;
        std::cout << "ERROR: Unwrapped const selection type is not equal to the policy's resource type\n";
    }

    //Checking if unwrap returns a different type when passed a selection
    auto selection = oneapi::dpl::experimental::select(p, func);
    auto selection_unwrap = oneapi::dpl::experimental::unwrap(selection);

    if(std::is_same_v<decltype(selection),decltype(selection_unwrap)>){
        pass=false;
        std::cout << "ERROR: Unwrapped selection type is equal to the actual selection type\n";
    }

    //Checking if the unwrapped selection type is the same as policy::resource_type
    if(!std::is_same_v<decltype(selection_unwrap),typename oneapi::dpl::experimental::policy_traits<Policy>::resource_type>)
    {
        pass=false;
        std::cout << "ERROR: Unwrapped selection type is not equal to the policy's resource type\n";
    }


    if(pass==false) return 1;
    std::cout << "Unwrap: OK\n" << std::flush;
    return 0;
}

template <typename Policy, typename UniverseContainer, typename ResourceFunction, bool AutoTune = false>
int
test_select(UniverseContainer u, ResourceFunction&& f)
{
    using my_policy_t = Policy;
    my_policy_t p{u};

    const int N = 100;
    std::atomic<int> ecount = 0;
    bool pass = true;

    auto function_key = []() {};

    for (int i = 1; i <= N; ++i)
    {
        auto test_resource = f(i);
        if constexpr (AutoTune)
        {
            auto h = select(p, function_key);
            if (oneapi::dpl::experimental::unwrap(h) != test_resource)
            {
                pass = false;
            }
        }
        else
        {
            auto h = select(p);
            if (oneapi::dpl::experimental::unwrap(h) != test_resource)
            {
                pass = false;
            }
        }
        ecount += i;
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
    std::cout << "select: OK\n";
    return 0;
}

template <bool call_select_before_submit, typename Policy, typename UniverseContainer, typename ResourceFunction>
int
test_submit_and_wait_on_group(UniverseContainer u, ResourceFunction&& f, int offset = 0)
{
    using my_policy_t = Policy;
    my_policy_t p{u};

    int N = 100;
    std::atomic<int> ecount = 0;
    bool pass = true;
    if constexpr (call_select_before_submit)
    {
        for (int i = 1; i <= N; i++)
        {
            auto test_resource = f(i);
            auto func = [&](typename Policy::resource_type e) {
                if (e != test_resource)
                {
                    pass = false;
                }
                ecount += i;
                if constexpr (std::is_same_v<typename oneapi::dpl::experimental::policy_traits<Policy>::resource_type,
                                             int>)
                    return e;
                else
                    return typename oneapi::dpl::experimental::policy_traits<Policy>::wait_type{};
            };
            auto s = oneapi::dpl::experimental::select(p);
            auto e = oneapi::dpl::experimental::submit(s, func);
        }
        oneapi::dpl::experimental::wait(p.get_submission_group());
    }
    else
    {
        for (int i = 1; i <= N; ++i)
        {
            auto test_resource = f(i);
            oneapi::dpl::experimental::submit(
                p, [&pass, &ecount, test_resource,
                    i](typename oneapi::dpl::experimental::policy_traits<Policy>::resource_type e) {
                    if (e != test_resource)
                    {
                        pass = false;
                    }
                    ecount += i;
                    if constexpr (std::is_same_v<
                                      typename oneapi::dpl::experimental::policy_traits<Policy>::resource_type, int>)
                        return e;
                    else
                        return typename oneapi::dpl::experimental::policy_traits<Policy>::wait_type{};
                });
        }
        oneapi::dpl::experimental::wait(p.get_submission_group());
    }
    if (!pass)
    {
        std::cout << "ERROR: did not select expected resources\n";
        return 1;
    }
    std::cout << "submit_and_wait_on_group: OK\n";
    return 0;
}

template <bool call_select_before_submit, typename Policy, typename UniverseContainer, typename ResourceFunction>
int
test_submit_and_wait_on_event(UniverseContainer u, ResourceFunction&& f, int offset = 0)
{
    using my_policy_t = Policy;
    my_policy_t p{u};

    const int N = 100;
    bool pass = true;

    std::atomic<int> ecount = 0;

    if constexpr (call_select_before_submit)
    {
        for (int i = 1; i <= N; ++i)
        {
            auto test_resource = f(i);
            auto func = [&pass, test_resource, &ecount,
                         i](typename oneapi::dpl::experimental::policy_traits<Policy>::resource_type e) {
                if (e != test_resource)
                {
                    pass = false;
                }
                ecount += i;
                if constexpr (std::is_same_v<typename oneapi::dpl::experimental::policy_traits<Policy>::resource_type,
                                             int>)
                    return e;
                else
                    return typename oneapi::dpl::experimental::policy_traits<Policy>::wait_type{};
            };
            auto s = oneapi::dpl::experimental::select(p, func);
            auto w = oneapi::dpl::experimental::submit(s, func);
            oneapi::dpl::experimental::wait(w);
            int count = ecount.load();
            if (count != i * (i + 1) / 2)
            {
                std::cout << "ERROR: scheduler did not execute all tasks exactly once\n";
                return 1;
            }
        }
    }
    else
    {
        for (int i = 1; i <= N; ++i)
        {
            auto test_resource = f(i);
            auto w = oneapi::dpl::experimental::submit(
                p, [&pass, test_resource, &ecount,
                    i](typename oneapi::dpl::experimental::policy_traits<Policy>::resource_type e) {
                    if (e != test_resource)
                    {
                        pass = false;
                    }
                    ecount += i;
                    if constexpr (std::is_same_v<
                                      typename oneapi::dpl::experimental::policy_traits<Policy>::resource_type, int>)
                        return e;
                    else
                        return typename oneapi::dpl::experimental::policy_traits<Policy>::wait_type{};
                });
            oneapi::dpl::experimental::wait(w);
            int count = ecount.load();
            if (count != i * (i + 1) / 2)
            {
                std::cout << "ERROR: scheduler did not execute all tasks exactly once\n";
                return 1;
            }
        }
    }
    if (!pass)
    {
        std::cout << "ERROR: did not select expected resources\n";
        return 1;
    }
    std::cout << "submit_and_wait_on_sync: OK\n";
    return 0;
}

template <bool call_select_before_submit, typename Policy, typename UniverseContainer, typename ResourceFunction>
int
test_submit_and_wait(UniverseContainer u, ResourceFunction&& f, int offset = 0)
{
    using my_policy_t = Policy;
    my_policy_t p{u};

    const int N = 100;
    std::atomic<int> ecount = 0;
    bool pass = true;

    if constexpr (call_select_before_submit)
    {
        for (int i = 1; i <= N; ++i)
        {
            auto test_resource = f(i);
            auto func = [&pass, test_resource, &ecount,
                         i](typename oneapi::dpl::experimental::policy_traits<Policy>::resource_type e) {
                if (e != test_resource)
                {
                    pass = false;
                }
                ecount += i;
                return typename oneapi::dpl::experimental::policy_traits<Policy>::wait_type{};
            };
            auto s = oneapi::dpl::experimental::select(p, func);
            oneapi::dpl::experimental::submit_and_wait(s, func);
            int count = ecount.load();
            if (count != i * (i + 1) / 2)
            {
                std::cout << "ERROR: scheduler did not execute all tasks exactly once\n";
                return 1;
            }
        }
    }
    else
    {
        for (int i = 1; i <= N; ++i)
        {
            auto test_resource = f(i);
            oneapi::dpl::experimental::submit_and_wait(
                p, [&pass, &ecount, test_resource,
                    i](typename oneapi::dpl::experimental::policy_traits<Policy>::resource_type e) {
                    if (e != test_resource)
                    {
                        pass = false;
                    }
                    ecount += i;
                    if constexpr (std::is_same_v<
                                      typename oneapi::dpl::experimental::policy_traits<Policy>::resource_type, int>)
                        return e;
                    else
                        return typename oneapi::dpl::experimental::policy_traits<Policy>::wait_type{};
                });
            int count = ecount.load();
            if (count != i * (i + 1) / 2)
            {
                std::cout << "ERROR: scheduler did not execute all tasks exactly once\n";
                return 1;
            }
        }
    }
    if (!pass)
    {
        std::cout << "ERROR: did not select expected resources\n";
        return 1;
    }
    std::cout << "submit_and_wait: OK\n";
    return 0;
}

#endif /* _ONEDPL_TEST_DYNAMIC_SELECTION_UTILS_H */
