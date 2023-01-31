// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This file incorporates work covered by the following copyright and permission
// notice:
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//


template<typename Policy>
int test_cout() {
  Policy p;
  std::cout << p << "\n";
  return 0;
}

template<typename Policy, typename UniverseContainer>
int test_properties(UniverseContainer u, typename UniverseContainer::value_type test_resource) {
  using my_policy_t = Policy;

  my_policy_t p{u};

  auto u2 = oneapi::dpl::experimental::property::query(p, oneapi::dpl::experimental::property::universe);
  auto u2s = u2.size();
  if (!std::equal(std::begin(u2), std::end(u2), std::begin(u))) {
    std::cout << "ERROR: reported universe and queried universe are not equal\n";
    return 1;
  }
  auto us = oneapi::dpl::experimental::property::query(p, oneapi::dpl::experimental::property::universe_size);
  if (u2s != us) {
    std::cout << "ERROR: queried universe size inconsistent with queried universe\n";
    return 1;
  }
  if (!oneapi::dpl::experimental::property::query(p, oneapi::dpl::experimental::property::is_device_available, test_resource)) {
    std::cout << "ERROR: cpu queried as not available\n";
    return 1;
  }
  std::cout << "properties: OK\n";
  return 0;
}

template<typename Policy, typename UniverseContainer, typename ResourceFunction>
int test_invoke_async_and_wait_on_policy(UniverseContainer u, ResourceFunction&& f) {
  using my_policy_t = Policy;
  my_policy_t p{u};

  const int N = 100;
  std::atomic<int> ecount = 0;
  bool pass = true;

  for (int i = 1; i <= N; ++i) {
    auto test_resource = f(i);
    oneapi::dpl::experimental::invoke_async(p,
                     [&pass,&ecount,test_resource, i](typename Policy::native_resource_t e) {
                       if (e != test_resource) {
                         pass = false;
                       }
                       ecount += i;
                       if constexpr (std::is_same_v<typename Policy::native_resource_t, int>)
                         return e;
                       else
                         return typename Policy::native_sync_t{};
                     });
  }
  oneapi::dpl::experimental::wait_for_all(p);
  int count = ecount.load();
  if (count != N*(N+1)/2) {
    std::cout << "ERROR: scheduler did not execute all tasks exactly once: " << count << "\n";
    return 1;
  }
  if (!pass) {
    std::cout << "ERROR: did not select expected resources\n";
    return 1;
  }
  std::cout << "async_invoke_wait_on_policy: OK\n";
  return 0;
}

template<typename Policy, typename UniverseContainer, typename ResourceFunction>
int test_invoke_async_and_wait_on_sync(UniverseContainer u, ResourceFunction&& f) {
  using my_policy_t = Policy;
  my_policy_t p{u};

  const int N = 100;
  std::atomic<int> ecount = 0;
  bool pass = true;

  for (int i = 1; i <= N; ++i) {
    auto test_resource = f(i);
    auto w = oneapi::dpl::experimental::invoke_async(p,
                              [&pass,&ecount,test_resource, i](typename Policy::native_resource_t e) {
                                if (e != test_resource) {
                                  pass = false;
                                }
                                ecount += i;
                                if constexpr (std::is_same_v<typename Policy::native_resource_t, int>)
                                  return e;
                                else
                                  return typename Policy::native_sync_t{};
                              });
    oneapi::dpl::experimental::wait_for_all(w);
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
  std::cout << "async_invoke_wait_on_sync: OK\n";
  return 0;
}

template<typename Policy, typename UniverseContainer, typename ResourceFunction>
int test_invoke(UniverseContainer u, ResourceFunction&& f) {
  using my_policy_t = Policy;
  my_policy_t p{u};

  const int N = 100;
  std::atomic<int> ecount = 0;
  bool pass = true;

  for (int i = 1; i <= N; ++i) {
    auto test_resource = f(i);
    oneapi::dpl::experimental::invoke(p,
               [&pass,&ecount,test_resource, i](typename Policy::native_resource_t e) {
                 if (e != test_resource) {
                   pass = false;
                 }
                 ecount += i;
                 if constexpr (std::is_same_v<typename Policy::native_resource_t, int>)
                   return e;
                 else
                   return typename Policy::native_sync_t{};
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
  std::cout << "invoke: OK\n";
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
    oneapi::dpl::experimental::invoke_async(p, h,
                     [&pass,&ecount,test_resource,i](typename Policy::native_resource_t e) {
                       if (e != test_resource) {
                         pass = false;
                       }
                       ecount += i;
                       if constexpr (std::is_same_v<typename Policy::native_resource_t, int>)
                         return e;
                       else
                         return typename Policy::native_sync_t{};
                     });
  }
  oneapi::dpl::experimental::wait_for_all(p);
  int count = ecount.load();
  if (count != N*(N+1)/2) {
    std::cout << "ERROR: scheduler did not execute all tasks exactly once: " << count << "\n";
    return 1;
  }
  if (!pass) {
    std::cout << "ERROR: did not select expected resources\n";
    return 1;
  }
  std::cout << "select_invoke_async_and_wait_on_policy: OK\n";
  return 0;
}

template<typename Policy, typename UniverseContainer, typename ResourceFunction>
int test_auto_tune_select_and_wait_on_policy(UniverseContainer u, ResourceFunction&& f) {
  using my_policy_t = Policy;
  my_policy_t p{u};

  const int N = 100;
  std::atomic<int> ecount = 0;
  bool pass = true;

  auto function_key = [](){};

  for (int i = 1; i <= N; ++i) {
    auto test_resource = f(i);
    auto h = select(p, function_key);
    oneapi::dpl::experimental::invoke_async(p, h,
                     [&pass,&ecount,test_resource,i](typename Policy::native_resource_t e) {
                       if (e != test_resource) {
                         pass = false;
                       }
                       ecount += i;
                       if constexpr (std::is_same_v<typename Policy::native_resource_t, int>)
                         return e;
                       else
                         return typename Policy::native_sync_t{};
                     });
  }
  oneapi::dpl::experimental::wait_for_all(p);
  int count = ecount.load();
  if (count != N*(N+1)/2) {
    std::cout << "ERROR: scheduler did not execute all tasks exactly once: " << count << "\n";
    return 1;
  }
  if (!pass) {
    std::cout << "ERROR: did not select expected resources\n";
    return 1;
  }
  std::cout << "select_invoke_async_and_wait_on_policy: OK\n";
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
    auto w = oneapi::dpl::experimental::invoke_async(p, h,
                     [&pass,&ecount,test_resource,i](typename Policy::native_resource_t e) {
                       if (e != test_resource) {
                         pass = false;
                       }
                       ecount += i;
                       if constexpr (std::is_same_v<typename Policy::native_resource_t, int>)
                         return e;
                       else
                         return typename Policy::native_sync_t{};
                     });
    oneapi::dpl::experimental::wait_for_all(w);
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
  std::cout << "select_invoke_async_and_wait_on_sync: OK\n";
  return 0;
}

template<typename Policy, typename UniverseContainer, typename ResourceFunction>
int test_auto_tune_select_and_wait_on_sync(UniverseContainer u, ResourceFunction&& f) {
  using my_policy_t = Policy;
  my_policy_t p{u};

  const int N = 100;
  std::atomic<int> ecount = 0;
  bool pass = true;

  auto function_key = [](){};

  for (int i = 1; i <= N; ++i) {
    auto test_resource = f(i);
    auto h = select(p, function_key);
    auto w = oneapi::dpl::experimental::invoke_async(p, h,
                     [&pass,&ecount,test_resource,i](typename Policy::native_resource_t e) {
                       if (e != test_resource) {
                         pass = false;
                       }
                       ecount += i;
                       if constexpr (std::is_same_v<typename Policy::native_resource_t, int>)
                         return e;
                       else
                         return typename Policy::native_sync_t{};
                     });
    oneapi::dpl::experimental::wait_for_all(w);
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
  std::cout << "select_invoke_async_and_wait_on_sync: OK\n";
  return 0;
}

template<typename Policy, typename UniverseContainer, typename ResourceFunction>
int test_select_invoke(UniverseContainer u, ResourceFunction&& f) {
  using my_policy_t = Policy;
  my_policy_t p{u};

  const int N = 100;
  std::atomic<int> ecount = 0;
  bool pass = true;

  for (int i = 1; i <= N; ++i) {
    auto test_resource = f(i);
    auto h = select(p);
    oneapi::dpl::experimental::invoke(p, h,
               [&pass,&ecount,test_resource,i](typename Policy::native_resource_t e) {
                 if (e != test_resource) {
                   pass = false;
                 }
                 ecount += i;
                 if constexpr (std::is_same_v<typename Policy::native_resource_t, int>)
                   return e;
                 else
                   return typename Policy::native_sync_t{};
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
  std::cout << "select_invoke_async_and_wait_on_sync: OK\n";
  return 0;
}

template<typename Policy, typename UniverseContainer, typename ResourceFunction>
int test_auto_tune_select_invoke(UniverseContainer u, ResourceFunction&& f) {
  using my_policy_t = Policy;
  my_policy_t p{u};

  const int N = 100;
  std::atomic<int> ecount = 0;
  bool pass = true;

  auto function_key = [](){};

  for (int i = 1; i <= N; ++i) {
    auto test_resource = f(i);
    auto h = select(p, function_key);
    oneapi::dpl::experimental::invoke(p, h,
               [&pass,&ecount,test_resource,i](typename Policy::native_resource_t e) {
                 if (e != test_resource) {
                   pass = false;
                 }
                 ecount += i;
                 if constexpr (std::is_same_v<typename Policy::native_resource_t, int>)
                   return e;
                 else
                   return typename Policy::native_sync_t{};
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
  std::cout << "select_invoke_async_and_wait_on_sync: OK\n";
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

template<typename Policy, typename UniverseContainer, typename ResourceFunction>
int test_auto_tune_select(UniverseContainer u, ResourceFunction&& f) {
  using my_policy_t = Policy;
  my_policy_t p{u};

  const int N = 100;
  std::atomic<int> ecount = 0;
  bool pass = true;

  auto function_key = [](){};

  for (int i = 1; i <= N; ++i) {
    auto test_resource = f(i);
    auto h = select(p, function_key);
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

