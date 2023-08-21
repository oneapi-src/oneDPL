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
#include "support/test_ds_utils.h"

int test_auto_initialization(const std::vector<sycl::queue>& u) {
  // initialize
  oneapi::dpl::experimental::auto_tune_policy p{u};
  auto u2 = oneapi::dpl::experimental::get_resources(p);
  auto u2s = u2.size();
  if (!std::equal(std::begin(u2), std::end(u2), std::begin(u))) {
    std::cout << "ERROR: provided resources and queried resources are not equal\n";
    return 1;
  }

  // deferred initialization
  oneapi::dpl::experimental::auto_tune_policy p2{oneapi::dpl::experimental::deferred_initialization};
  try {
    auto u3 = oneapi::dpl::experimental::get_resources(p2);
    if (!u3.empty()) {
      std::cout << "ERROR: deferred initialization not respected\n";
      return 1;
    }
  } catch (...)  { }
  p2.initialize(u); 
  auto u3 = oneapi::dpl::experimental::get_resources(p);
  auto u3s = u3.size();
  if (!std::equal(std::begin(u3), std::end(u3), std::begin(u))) {
    std::cout << "ERROR: reported resources and queried resources are not equal after deferred initialization\n";
    return 1;
  }

  std::cout << "initialization: OK\n" << std::flush;
  return 0;
}

template<bool call_select_before_submit, typename Policy, typename UniverseContainer>
int test_auto_submit_wait_on_event(UniverseContainer u, int best_resource) {
  using my_policy_t = Policy;

  // they are cpus so this is ok
  double *v = sycl::malloc_shared<double>(1000000, u[0]);
  int *j = sycl::malloc_shared<int>(1, u[0]);

  my_policy_t p{u};
  auto n_samples = u.size();

  const int N = 10;
  std::atomic<int> ecount = 0;
  bool pass = true;

  for (int i = 1; i <= N; ++i) {
    if (i <= n_samples && i-1 != best_resource) {
        *j = 10000;
    } else {
        *j = 1;
    }
    // we can capture all by reference
    // the inline_scheduler reports timings in submit
    // We wait but it should return immediately, since inline 
    // scheduler does the work "inline". 
    // The unwrapped wait type should be equal to the resource 
    if constexpr (call_select_before_submit) {
      auto f = [&](typename oneapi::dpl::experimental::policy_traits<Policy>::resource_type q) {
                   if (i <= n_samples) {
                     // we should be round-robining through the resources
                     if (q != u[i-1]) {
                       std::cout << i << ": mismatch during rr phase\n" << std::flush;
                       pass = false;
                     }
                   } else {
                     if (q != u[best_resource]) {
                       std::cout << i << ": mismatch during prod phase " << best_resource << "\n" << std::flush;
                       pass = false;
                     }
                   }
                   ecount += i;
                   return q.submit([=](sycl::handler& h) {
                     h.parallel_for(1000000, [=](sycl::id<1> idx) {
                       for (int j0 = 0; j0 < *j; ++j0) {
                         v[idx] += idx;
                       }
                     });
                   });
                 };
      auto s = oneapi::dpl::experimental::select(p, f);
      auto e = oneapi::dpl::experimental::submit(s, f);
      oneapi::dpl::experimental::wait(e);
    } else {
      // it's ok to capture by reference since we are waiting on each call
      auto s = oneapi::dpl::experimental::submit(p,
                 [&](typename oneapi::dpl::experimental::policy_traits<Policy>::resource_type q) {
                   if (i <= n_samples) {
                     // we should be round-robining through the resources
                     if (q != u[i-1]) {
                       std::cout << i << ": mismatch during rr phase\n" << std::flush;
                       pass = false;
                     }
                   } else {
                     if (q != u[best_resource]) {
                       std::cout << i << ": mismatch during prod phase " << best_resource << "\n" << std::flush;
                       pass = false;
                     }
                   }
                   ecount += i;
                   return q.submit([=](sycl::handler& h) {
                     h.parallel_for(1000000, [=](sycl::id<1> idx) {
                       for (int j0 = 0; j0 < *j; ++j0) {
                         v[idx] += idx;
                       }
                     });
                   });
                 });
      oneapi::dpl::experimental::wait(s);
    }

    int count = ecount.load();
    if (count != i*(i+1)/2) {
      std::cout << "ERROR: scheduler did not execute all tasks exactly once\n";
      return 1;
    }
  }
  if (!pass) {
    std::cout << "ERROR: did not select expected resources\n";
    return 1;
  }
  if constexpr (call_select_before_submit) {
    std::cout << "select then submit and wait on event: OK\n";
  } else {
    std::cout << "submit and wait on event: OK\n";
  }
  return 0;
}

template<bool call_select_before_submit, typename Policy, typename UniverseContainer>
int test_auto_submit_wait_on_group(UniverseContainer u, int best_resource) {
  using my_policy_t = Policy;

  // they are cpus so this is ok
  double *v = sycl::malloc_shared<double>(1000000, u[0]);
  int *j = sycl::malloc_shared<int>(1, u[0]);

  my_policy_t p{u};
  auto n_samples = u.size();

  const int N = 10;
  std::atomic<int> ecount = 0;
  bool pass = true;

  for (int i = 1; i <= N; ++i) {
    if (i <= n_samples && i-1 != best_resource) {
        *j = 10000;
    } else {
        *j = 1;
    }
    // we can capture all by reference
    // the inline_scheduler reports timings in submit
    // We wait but it should return immediately, since inline 
    // scheduler does the work "inline". 
    // The unwrapped wait type should be equal to the resource 
    if constexpr (call_select_before_submit) {
      auto f = [&](typename oneapi::dpl::experimental::policy_traits<Policy>::resource_type q) {
                   if (i <= n_samples) {
                     // we should be round-robining through the resources
                     if (q != u[i-1]) {
                       std::cout << i << ": mismatch during rr phase\n" << std::flush;
                       pass = false;
                     }
                   } else {
                     if (q != u[best_resource]) {
                       std::cout << i << ": mismatch during prod phase " << best_resource << "\n" << std::flush;
                       pass = false;
                     }
                   }
                   ecount += i;
                   return q.submit([=](sycl::handler& h) {
                     h.parallel_for(1000000, [=](sycl::id<1> idx) {
                       for (int j0 = 0; j0 < *j; ++j0) {
                         v[idx] += idx;
                       }
                     });
                   });
                 };
      auto s = oneapi::dpl::experimental::select(p, f);
      auto e = oneapi::dpl::experimental::submit(s, f);
      oneapi::dpl::experimental::wait(p.get_submission_group());
    } else {
      // it's ok to capture by reference since we are waiting on each call
      auto s = oneapi::dpl::experimental::submit(p,
                 [&](typename oneapi::dpl::experimental::policy_traits<Policy>::resource_type q) {
                   if (i <= n_samples) {
                     // we should be round-robining through the resources
                     if (q != u[i-1]) {
                       std::cout << i << ": mismatch during rr phase\n" << std::flush;
                       pass = false;
                     }
                   } else {
                     if (q != u[best_resource]) {
                       std::cout << i << ": mismatch during prod phase " << best_resource << "\n" << std::flush;
                       pass = false;
                     }
                   }
                   ecount += i;
                   return q.submit([=](sycl::handler& h) {
                     h.parallel_for(1000000, [=](sycl::id<1> idx) {
                       for (int j0 = 0; j0 < *j; ++j0) {
                         v[idx] += idx;
                       }
                     });
                   });
                 });
      oneapi::dpl::experimental::wait(p.get_submission_group());
    }

    int count = ecount.load();
    if (count != i*(i+1)/2) {
      std::cout << "ERROR: scheduler did not execute all tasks exactly once\n";
      return 1;
    }
  }
  if (!pass) {
    std::cout << "ERROR: did not select expected resources\n";
    return 1;
  }
  if constexpr (call_select_before_submit) {
    std::cout << "select then submit and wait on group: OK\n";
  } else {
    std::cout << "submit and wait on group: OK\n";
  }
  return 0;
}

template<bool call_select_before_submit, typename Policy, typename UniverseContainer>
int test_auto_submit_and_wait(UniverseContainer u, int best_resource) {
  using my_policy_t = Policy;

  // they are cpus so this is ok
  double *v = sycl::malloc_shared<double>(1000000, u[0]);
  int *j = sycl::malloc_shared<int>(1, u[0]);

  my_policy_t p{u};
  auto n_samples = u.size();

  const int N = 10;
  std::atomic<int> ecount = 0;
  bool pass = true;

  for (int i = 1; i <= N; ++i) {
    if (i <= n_samples && i-1 != best_resource) {
        *j = 10000;
    } else {
        *j = 1;
    }
    // we can capture all by reference
    // the inline_scheduler reports timings in submit
    // We wait but it should return immediately, since inline 
    // scheduler does the work "inline". 
    // The unwrapped wait type should be equal to the resource 
    if constexpr (call_select_before_submit) {
      auto f = [&](typename oneapi::dpl::experimental::policy_traits<Policy>::resource_type q) {
                   if (i <= n_samples) {
                     // we should be round-robining through the resources
                     if (q != u[i-1]) {
                       std::cout << i << ": mismatch during rr phase\n" << std::flush;
                       pass = false;
                     }
                   } else {
                     if (q != u[best_resource]) {
                       std::cout << i << ": mismatch during prod phase " << best_resource << "\n" << std::flush;
                       pass = false;
                     }
                   }
                   ecount += i;
                   return q.submit([=](sycl::handler& h) {
                     h.parallel_for(1000000, [=](sycl::id<1> idx) {
                       for (int j0 = 0; j0 < *j; ++j0) {
                         v[idx] += idx;
                       }
                     });
                   });
                 };
      auto s = oneapi::dpl::experimental::select(p, f);
      oneapi::dpl::experimental::submit_and_wait(s, f);
    } else {
      // it's ok to capture by reference since we are waiting on each call
      oneapi::dpl::experimental::submit_and_wait(p,
                 [&](typename oneapi::dpl::experimental::policy_traits<Policy>::resource_type q) {
                   if (i <= n_samples) {
                     // we should be round-robining through the resources
                     if (q != u[i-1]) {
                       std::cout << i << ": mismatch during rr phase\n" << std::flush;
                       pass = false;
                     }
                   } else {
                     if (q != u[best_resource]) {
                       std::cout << i << ": mismatch during prod phase " << best_resource << "\n" << std::flush;
                       pass = false;
                     }
                   }
                   ecount += i;
                   return q.submit([=](sycl::handler& h) {
                     h.parallel_for(1000000, [=](sycl::id<1> idx) {
                       for (int j0 = 0; j0 < *j; ++j0) {
                         v[idx] += idx;
                       }
                     });
                   });
                 });
    }

    int count = ecount.load();
    if (count != i*(i+1)/2) {
      std::cout << "ERROR: scheduler did not execute all tasks exactly once\n";
      return 1;
    }
  }
  if (!pass) {
    std::cout << "ERROR: did not select expected resources\n";
    return 1;
  }
  if constexpr (call_select_before_submit) {
    std::cout << "select then submit_and_wait: OK\n";
  } else {
    std::cout << "submit_and_wait: OK\n";
  }
  return 0;
}

/*

template<typename Policy, typename UniverseContainer, bool do_select=false>
int test_auto_submit_wait_on_group(UniverseContainer u, int best_resource) {
  using my_policy_t = Policy;

  my_policy_t p{u};
  auto n_samples = u.size();

  const int N = 100;
  std::atomic<int> ecount = 0;
  bool pass = true;

  for (int i = 1; i <= N; ++i) {
    // we can capture all by reference, since it should wait, no concurrency
    if constexpr (do_select) {
      auto f = [&](typename oneapi::dpl::experimental::policy_traits<Policy>::resource_type e) {
                   std::this_thread::sleep_for(std::chrono::milliseconds(e));
                   if (i <= n_samples) {
                     // we should be round-robining through the resources
                     if (e != u[i-1]) {
                       pass = false;
                     }
                   } else {
                     if (e != best_resource) {
                       pass = false;
                     }
                   }
                   ecount += i;
                   return typename oneapi::dpl::experimental::policy_traits<Policy>::wait_type{};
                 };
      auto s = oneapi::dpl::experimental::select(p, f);
      auto e = oneapi::dpl::experimental::submit(s, f);
    } else {
      oneapi::dpl::experimental::submit(p,
                 [&](typename oneapi::dpl::experimental::policy_traits<Policy>::resource_type e) {
                   std::this_thread::sleep_for(std::chrono::milliseconds(e));
                   if (i <= n_samples) {
                     // we should be round-robining through the resources
                     if (e != u[i-1]) {
                       pass = false;
                     }
                   } else {
                     if (e != best_resource) {
                       pass = false;
                     }
                   }
                   ecount += i;
                   return typename oneapi::dpl::experimental::policy_traits<Policy>::wait_type{};
                 });
    }
    int count = ecount.load();
    if (count != i*(i+1)/2) {
      std::cout << "ERROR: scheduler did not execute all tasks exactly once\n";
      return 1;
    }
  }
  // this has no effect for inline_scheduler, so nothing to test other than the call
  // doesn't fail
  oneapi::dpl::experimental::wait(p.get_submission_group());
  
  if (!pass) {
    std::cout << "ERROR: did not select expected resources\n";
    return 1;
  }
  std::cout << "submit_wait_on_group: OK\n";
  return 0;
}

template<typename Policy, typename UniverseContainer, bool do_select=false>
int test_auto_submit_and_wait(UniverseContainer u, int best_resource) {
  using my_policy_t = Policy;

  my_policy_t p{u};
  auto n_samples = u.size();

  const int N = 100;
  std::atomic<int> ecount = 0;
  bool pass = true;

  for (int i = 1; i <= N; ++i) {
    // we can capture all by reference, since it should wait, no concurrency
    if constexpr (do_select) {
      auto f =  [&](typename oneapi::dpl::experimental::policy_traits<Policy>::resource_type e) {
                   std::this_thread::sleep_for(std::chrono::milliseconds(e));
                   if (i <= n_samples) {
                     // we should be round-robining through the resources
                     if (e != u[i-1]) {
                       pass = false;
                     }
                   } else {
                     if (e != best_resource) {
                       pass = false;
                     }
                   }
                   ecount += i;
                   return typename oneapi::dpl::experimental::policy_traits<Policy>::wait_type{};
                 };
      auto s = oneapi::dpl::experimental::select(p, f);
      oneapi::dpl::experimental::submit_and_wait(s, f);
    } else {
      oneapi::dpl::experimental::submit_and_wait(p,
                 [&](typename oneapi::dpl::experimental::policy_traits<Policy>::resource_type e) {
                   std::this_thread::sleep_for(std::chrono::milliseconds(e));
                   if (i <= n_samples) {
                     // we should be round-robining through the resources
                     if (e != u[i-1]) {
                       pass = false;
                     }
                   } else {
                     if (e != best_resource) {
                       pass = false;
                     }
                   }
                   ecount += i;
                   return typename oneapi::dpl::experimental::policy_traits<Policy>::wait_type{};
                 });
    }
    int count = ecount.load();
    if (count != i*(i+1)/2) {
      std::cout << "ERROR: scheduler did not execute all tasks exactly once\n";
      return 1;
    }
  }
  if (!pass) {
    std::cout << "ERROR: did not select expected resources\n";
    return 1;
  }
  std::cout << "submit_and_wait: OK\n";
  return 0;
}
*/

int main() { 
  using policy_t = oneapi::dpl::experimental::auto_tune_policy<oneapi::dpl::experimental::sycl_backend>; 

  sycl::queue q1 = sycl::queue{sycl::cpu_selector_v};
  sycl::queue q2 = sycl::queue{sycl::cpu_selector_v};
  sycl::queue q3 = sycl::queue{sycl::cpu_selector_v};
  sycl::queue q4 = sycl::queue{sycl::cpu_selector_v};
  std::vector<sycl::queue> u = {q1, q2, q3, q4};

  auto f = [u](int i) { 
             if (i <= 4)
               return u[i-1]; 
             else
               return u[0]; 
           };
//         || test_auto_submit_and_wait<policy_t>(u, best_resource)
//         // now select then submits
//         || test_auto_submit_wait_on_event<policy_t,  decltype(u), true>(u, best_resource)
//         || test_auto_submit_and_wait<policy_t, decltype(u), true>(u, best_resource);

  constexpr bool just_call_submit = false;
  constexpr bool call_select_before_submit = true;

  if (test_auto_initialization(u)
      || test_select<policy_t, decltype(u), const decltype(f)&, true>(u, f) 
      || test_auto_submit_wait_on_event<just_call_submit, policy_t>(u, 0)
      || test_auto_submit_wait_on_event<just_call_submit, policy_t>(u, 1)
      || test_auto_submit_wait_on_event<just_call_submit, policy_t>(u, 2)
      || test_auto_submit_wait_on_event<just_call_submit, policy_t>(u, 3)
      || test_auto_submit_wait_on_group<just_call_submit, policy_t>(u, 0)
      || test_auto_submit_wait_on_group<just_call_submit, policy_t>(u, 1)
      || test_auto_submit_wait_on_group<just_call_submit, policy_t>(u, 2)
      || test_auto_submit_wait_on_group<just_call_submit, policy_t>(u, 3)
      || test_auto_submit_and_wait<just_call_submit, policy_t>(u, 0)
      || test_auto_submit_and_wait<just_call_submit, policy_t>(u, 1)
      || test_auto_submit_and_wait<just_call_submit, policy_t>(u, 2)
      || test_auto_submit_and_wait<just_call_submit, policy_t>(u, 3)
      // now select then submits
      || test_auto_submit_wait_on_event<call_select_before_submit, policy_t>(u, 0)
      || test_auto_submit_wait_on_event<call_select_before_submit, policy_t>(u, 1)
      || test_auto_submit_wait_on_event<call_select_before_submit, policy_t>(u, 2)
      || test_auto_submit_wait_on_event<call_select_before_submit, policy_t>(u, 3)
      || test_auto_submit_wait_on_group<call_select_before_submit, policy_t>(u, 0)
      || test_auto_submit_wait_on_group<call_select_before_submit, policy_t>(u, 1)
      || test_auto_submit_wait_on_group<call_select_before_submit, policy_t>(u, 2)
      || test_auto_submit_wait_on_group<call_select_before_submit, policy_t>(u, 3)
      || test_auto_submit_and_wait<call_select_before_submit, policy_t>(u, 0)
      || test_auto_submit_and_wait<call_select_before_submit, policy_t>(u, 1)
      || test_auto_submit_and_wait<call_select_before_submit, policy_t>(u, 2)
      || test_auto_submit_and_wait<call_select_before_submit, policy_t>(u, 3)
     )
  {
    std::cout << "FAIL\n";
    return 1;
  } else {
    std::cout << "PASS\n";
    return 0;
  }

}


