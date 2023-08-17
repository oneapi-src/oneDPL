// -*- C++ -*-
//===---------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//


#ifndef _ONEDPL_TEST_DS_UTILS_H
#define _ONEDPL_TEST_DS_UTILS_H

template<typename Policy>
int test_cout() {
  Policy p;
  //std::cout << p << "\n";
  return 0;
}

template<typename Policy, typename UniverseContainer>
int test_properties(UniverseContainer u, typename UniverseContainer::value_type test_resource) {
  using my_policy_t = Policy;

  my_policy_t p{u};

  auto u2 = oneapi::dpl::experimental::get_universe(p);
  auto u2s = u2.size();
  if (!std::equal(std::begin(u2), std::end(u2), std::begin(u))) {
    std::cout << "ERROR: reported universe and queried universe are not equal\n";
    return 1;
  }
  auto us = oneapi::dpl::experimental::get_universe_size(p);
  if (u2s != us) {
    std::cout << "ERROR: queried universe size inconsistent with queried universe\n";
    return 1;
  }
  std::cout << "properties: OK\n";
  return 0;
}

template<typename Policy, typename UniverseContainer, typename ResourceFunction>
int test_submit_and_wait_on_policy(UniverseContainer u, ResourceFunction&& f) {
  using my_policy_t = Policy;
  my_policy_t p{u};

  const int N = 100;
  std::atomic<int> ecount = 0;
  bool pass = true;

  for (int i = 1; i <= N; ++i) {
    auto test_resource = f(i);
    oneapi::dpl::experimental::submit(p,
                     [&pass,&ecount,test_resource, i](typename oneapi::dpl::experimental::policy_traits<Policy>::resource_type e) {
                       if (e != test_resource) {
                         pass = false;
                       }
                       ecount += i;
                       if constexpr (std::is_same_v<typename oneapi::dpl::experimental::policy_traits<Policy>::resource_type, int>)
                         return e;
                       else
                         return typename oneapi::dpl::experimental::policy_traits<Policy>::wait_type{};
                     });
  }
  oneapi::dpl::experimental::wait(p);
  int count = ecount.load();
  if (count != N*(N+1)/2) {
    std::cout << "ERROR: scheduler did not execute all tasks exactly once: " << count << "\n";
    return 1;
  }
  if (!pass) {
    std::cout << "ERROR: did not select expected resources\n";
    return 1;
  }
  std::cout << "submit_and_wait_on_policy: OK\n";
  return 0;
}

template<typename Policy, typename UniverseContainer, typename ResourceFunction>
int test_submit_and_get_wait_list(UniverseContainer u, ResourceFunction&& f) {
  using my_policy_t = Policy;
  my_policy_t p{u};

  const int N = 100;
  std::atomic<int> ecount = 0;
  bool pass = true;

  for (int i = 1; i <= N; ++i) {
    auto test_resource = f(i);
    oneapi::dpl::experimental::submit(p,
                     [&pass,&ecount,test_resource, i](typename oneapi::dpl::experimental::policy_traits<Policy>::resource_type e) {
                       if (e != test_resource) {
                         pass = false;
                       }
                       ecount += i;
                       if constexpr (std::is_same_v<typename oneapi::dpl::experimental::policy_traits<Policy>::resource_type, int>)
                         return e;
                       else
                         return typename oneapi::dpl::experimental::policy_traits<Policy>::wait_type{};
                     });
  }
  auto wlist=oneapi::dpl::experimental::get_wait_list(p);
  oneapi::dpl::experimental::wait(wlist);
  int count = ecount.load();
  if (count != N*(N+1)/2) {
    std::cout << "ERROR: scheduler did not execute all tasks exactly once: " << count << "\n";
    return 1;
  }
  if (!pass) {
    std::cout << "ERROR: did not select expected resources\n";
    return 1;
  }
  std::cout << "submit_and_get_wait_list: OK\n";
  return 0;
}

template<typename Policy, typename UniverseContainer, typename ResourceFunction>
int test_submit_and_get_wait_list_single_element(UniverseContainer u, ResourceFunction&& f) {
  using my_policy_t = Policy;
  my_policy_t p{u};

  const int N = 1;
  std::atomic<int> ecount = 0;
  bool pass = true;

  for (int i = 1; i <= N; ++i) {
    auto test_resource = f(i);
    oneapi::dpl::experimental::submit(p,
                     [&pass,&ecount,test_resource, i](typename oneapi::dpl::experimental::policy_traits<Policy>::resource_type e) {
                       if (e != test_resource) {
                         pass = false;
                       }
                       ecount += i;
                       if constexpr (std::is_same_v<typename oneapi::dpl::experimental::policy_traits<Policy>::resource_type, int>)
                         return e;
                       else
                         return typename oneapi::dpl::experimental::policy_traits<Policy>::wait_type{};
                     });
  }
  auto wlist=oneapi::dpl::experimental::get_wait_list(p);
  oneapi::dpl::experimental::wait(wlist);
  int count = ecount.load();
  if (count != 1) {
    std::cout << "ERROR: scheduler did not execute all tasks exactly once: " << count << "\n";
    return 1;
  }
  if (!pass) {
    std::cout << "ERROR: did not select expected resources\n";
    return 1;
  }
  std::cout << "submit_and_get_wait_list single element: OK\n";
  return 0;
}

template<typename Policy, typename UniverseContainer, typename ResourceFunction>
int test_submit_and_get_wait_list_empty(UniverseContainer u, ResourceFunction&& f) {
  using my_policy_t = Policy;
  my_policy_t p{u};

  const int N = 0;
  std::atomic<int> ecount = 0;
  bool pass = true;

  for (int i = 1; i <= N; ++i) {
    auto test_resource = f(i);
    oneapi::dpl::experimental::submit(p,
                     [&pass,&ecount,test_resource, i](typename oneapi::dpl::experimental::policy_traits<Policy>::resource_type e) {
                       if (e != test_resource) {
                         pass = false;
                       }
                       ecount += i;
                       if constexpr (std::is_same_v<typename oneapi::dpl::experimental::policy_traits<Policy>::resource_type, int>)
                         return e;
                       else
                         return typename oneapi::dpl::experimental::policy_traits<Policy>::wait_type{};
                     });
  }
  auto wlist=oneapi::dpl::experimental::get_wait_list(p);
  oneapi::dpl::experimental::wait(wlist);
  int count = ecount.load();
  if (count != 0) {
    std::cout << "ERROR: scheduler did not execute all tasks exactly once: " << count << "\n";
    return 1;
  }
  if (!pass) {
    std::cout << "ERROR: did not select expected resources\n";
    return 1;
  }
  std::cout << "submit_and_get_wait_list empty list: OK\n";
  return 0;
}

template<typename Policy, typename UniverseContainer, typename ResourceFunction>
int test_submit_and_wait_on_sync(UniverseContainer u, ResourceFunction&& f) {
  using my_policy_t = Policy;
  my_policy_t p{u};

  const int N = 100;
  std::atomic<int> ecount = 0;
  bool pass = true;

  for (int i = 1; i <= N; ++i) {
    auto test_resource = f(i);
    auto w = oneapi::dpl::experimental::submit(p,
                              [&pass,&ecount,test_resource, i](typename oneapi::dpl::experimental::policy_traits<Policy>::resource_type e) {
                                if (e != test_resource) {
                                  pass = false;
                                }
                                ecount += i;
                                if constexpr (std::is_same_v<typename oneapi::dpl::experimental::policy_traits<Policy>::resource_type, int>)
                                  return e;
                                else
                                  return typename oneapi::dpl::experimental::policy_traits<Policy>::wait_type{};
                              });
    oneapi::dpl::experimental::wait(w);
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
  std::cout << "submit_and_wait_on_sync: OK\n";
  return 0;
}

template<typename Policy, typename UniverseContainer, typename ResourceFunction>
int test_submit_and_wait(UniverseContainer u, ResourceFunction&& f) {
  using my_policy_t = Policy;
  my_policy_t p{u};

  const int N = 100;
  std::atomic<int> ecount = 0;
  bool pass = true;

  for (int i = 1; i <= N; ++i) {
    auto test_resource = f(i);
    oneapi::dpl::experimental::submit_and_wait(p,
               [&pass,&ecount,test_resource, i](typename oneapi::dpl::experimental::policy_traits<Policy>::resource_type e) {
                 if (e != test_resource) {
                   pass = false;
                 }
                 ecount += i;
                 if constexpr (std::is_same_v<typename oneapi::dpl::experimental::policy_traits<Policy>::resource_type, int>)
                   return e;
                 else
                   return typename oneapi::dpl::experimental::policy_traits<Policy>::wait_type{};
               });
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

template<typename Policy, typename UniverseContainer, typename ResourceFunction>
int test_select_and_wait_on_policy(UniverseContainer u, ResourceFunction&& f) {
  using my_policy_t = Policy;
  my_policy_t p{u};

  const int N = 100;
  std::atomic<int> ecount = 0;
  bool pass = true;

  for (int i = 1; i <= N; ++i) {
    auto test_resource = f(i);
    auto h = select(p);
    oneapi::dpl::experimental::submit(p, h,
                     [&pass,&ecount,test_resource,i](typename oneapi::dpl::experimental::policy_traits<Policy>::resource_type e) {
                       if (e != test_resource) {
                         pass = false;
                       }
                       ecount += i;
                       if constexpr (std::is_same_v<typename oneapi::dpl::experimental::policy_traits<Policy>::resource_type, int>)
                         return e;
                       else
                         return typename oneapi::dpl::experimental::policy_traits<Policy>::wait_type{};
                     });
  }
  oneapi::dpl::experimental::wait(p);
  int count = ecount.load();
  if (count != N*(N+1)/2) {
    std::cout << "ERROR: scheduler did not execute all tasks exactly once: " << count << "\n";
    return 1;
  }
  if (!pass) {
    std::cout << "ERROR: did not select expected resources\n";
    return 1;
  }
  std::cout << "select_submit_and_wait_on_policy: OK\n";
  return 0;
}

template<typename Policy, typename UniverseContainer, typename ResourceFunction>
int test_select_and_wait_on_sync(UniverseContainer u, ResourceFunction&& f) {
  using my_policy_t = Policy;
  my_policy_t p{u};

  const int N = 100;
  std::atomic<int> ecount = 0;
  bool pass = true;

  for (int i = 1; i <= N; ++i) {
    auto test_resource = f(i);
    auto h = select(p);
    auto w = oneapi::dpl::experimental::submit(p, h,
                     [&pass,&ecount,test_resource,i](typename oneapi::dpl::experimental::policy_traits<Policy>::resource_type e) {
                       if (e != test_resource) {
                         pass = false;
                       }
                       ecount += i;
                       if constexpr (std::is_same_v<typename oneapi::dpl::experimental::policy_traits<Policy>::resource_type, int>)
                         return e;
                       else
                         return typename oneapi::dpl::experimental::policy_traits<Policy>::wait_type{};
                     });
    oneapi::dpl::experimental::wait(w);
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
  std::cout << "select_submit_and_wait_on_sync: OK\n";
  return 0;
}

template<typename Policy, typename UniverseContainer, typename ResourceFunction>
int test_select_submit_and_wait(UniverseContainer u, ResourceFunction&& f) {
  using my_policy_t = Policy;
  my_policy_t p{u};

  const int N = 100;
  std::atomic<int> ecount = 0;
  bool pass = true;

  for (int i = 1; i <= N; ++i) {
    auto test_resource = f(i);
    auto h = select(p);
    oneapi::dpl::experimental::submit_and_wait(p, h,
               [&pass,&ecount,test_resource,i](typename oneapi::dpl::experimental::policy_traits<Policy>::resource_type e) {
                 if (e != test_resource) {
                   pass = false;
                 }
                 ecount += i;
                 if constexpr (std::is_same_v<typename oneapi::dpl::experimental::policy_traits<Policy>::resource_type, int>)
                   return e;
                 else
                   return typename oneapi::dpl::experimental::policy_traits<Policy>::wait_type{};
               });
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
  std::cout << "select_submit_and_wait: OK\n";
  return 0;
}

template<typename Policy, typename UniverseContainer, typename ResourceFunction>
int test_select(UniverseContainer u, ResourceFunction&& f) {
  using my_policy_t = Policy;
  my_policy_t p{u};

  const int N = 100;
  std::atomic<int> ecount = 0;
  bool pass = true;

  for (int i = 1; i <= N; ++i) {
    auto test_resource = f(i);
    auto h = select(p);
    if (h.get_native() != test_resource) {
         pass = false;
    }
    ecount += i;
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
  std::cout << "select: OK\n";
  return 0;
}

#endif /* _ONEDPL_TEST_DS_UTILS_H */
