/*
    Copyright 2021 Intel Corporation.  All Rights Reserved.

    The source code contained or described herein and all documents related
    to the source code ("Material") are owned by Intel Corporation or its
    suppliers or licensors.  Title to the Material remains with Intel
    Corporation or its suppliers and licensors.  The Material is protected
    by worldwide copyright laws and treaty provisions.  No part of the
    Material may be used, copied, reproduced, modified, published, uploaded,
    posted, transmitted, distributed, or disclosed in any way without
    Intel's prior express written permission.

    No license under any patent, copyright, trade secret or other
    intellectual property right is granted to or conferred upon you by
    disclosure or delivery of the Materials, either expressly, by
    implication, inducement, estoppel or otherwise.  Any license under such
    intellectual property rights must be express and approved by Intel in
    writing.
*/

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

  auto u2 = ds::property::query(p, ds::property::universe);
  auto u2s = u2.size();
  if (!std::equal(std::begin(u2), std::end(u2), std::begin(u))) {
    std::cout << "ERROR: reported universe and queried universe are not equal\n";
    return 1;
  } 
  auto us = ds::property::query(p, ds::property::universe_size);
  if (u2s != us) {
    std::cout << "ERROR: queried universe size inconsistent with queried universe\n";
    return 1;
  } 
  if (!ds::property::query(p, ds::property::is_device_available, test_resource)) {
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
    ds::invoke_async(p, 
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
  ds::wait_for_all(p);
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
    auto w = ds::invoke_async(p, 
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
    ds::wait_for_all(w);
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
    ds::invoke(p, 
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
    ds::invoke_async(p, h, 
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
  ds::wait_for_all(p);
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
    ds::invoke_async(p, h, 
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
  ds::wait_for_all(p);
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
    auto w = ds::invoke_async(p, h, 
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
    ds::wait_for_all(w);
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
    auto w = ds::invoke_async(p, h, 
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
    ds::wait_for_all(w);
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
    ds::invoke(p, h, 
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
    ds::invoke(p, h, 
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

