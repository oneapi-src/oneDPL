// -*- C++ -*-
//===---------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//


#ifndef _ONEDPL_TEST_DYNAMIC_LOAD_UTILS_H
#define _ONEDPL_TEST_DYNAMIC_LOAD_UTILS_H

#include<thread>
#include<chrono>
#include<random>
#include<algorithm>

template<typename Policy>
int test_cout() {
  Policy p;
  //std::cout << p << "\n";
  return 0;
}

/*template<typename Policy, typename UniverseContainer>
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
  std::cout << "properties: OK\n";
  return 0;
}*/

template<typename Policy, typename UniverseContainer, typename ResourceFunction>
int test_invoke_async_and_wait_on_policy(UniverseContainer u, ResourceFunction&& f) {
  using my_policy_t = Policy;
  my_policy_t p{u};

  const int loops = 6;
  std::atomic<int> ecount = 0;
  bool pass = true;


  std::random_device rnd_device;
  // Specify the engine and distribution.
  std::mt19937 mersenne_engine {rnd_device()};  // Generates random integers
  std::uniform_int_distribution<int> dist {1, 52};

  auto gen = [&dist, &mersenne_engine](){
               return dist(mersenne_engine);
           };
  std::vector<double> a_host(1000000, 0);
  std::vector<double> b_host(1000000, 0);
  std::vector<double> c_host(100000000, 0);
  std::generate(begin(a_host), end(a_host), gen);
  std::generate(begin(b_host), end(b_host), gen);


  size_t M = 10000;
  size_t N = 100;
  size_t P = 10000;

  sycl::buffer a(a_host.data(), sycl::range<1>{a_host.size()});
  sycl::buffer b(b_host.data(), sycl::range<1>{b_host.size()});
  sycl::buffer c(c_host.data(), sycl::range<2>{M, P});

  for (int i = 0; i < loops; ++i) {
    auto test_resource = f(i);
    oneapi::dpl::experimental::invoke_async(p,
                     [&pass,&ecount,test_resource, i, &a, &b, &c, M, N, P](typename Policy::native_resource_t e) {
                       if (e != test_resource) {
                         pass = false;
                       }
                       ecount += i+1;
                        double *v = sycl::malloc_shared<double>(1000000, e);
                        auto e2 = e.submit([&](sycl::handler& h){
                           auto A = a.get_access<sycl::access::mode::read>(h);
                           auto B = b.get_access<sycl::access::mode::read>(h);
                           auto C = c.get_access<sycl::access::mode::write>(h);

                           if(i==0){
                                    std::cout<<"Printing first function"<<std::endl;
                                h.parallel_for(sycl::range<2>{M, P}, [=](sycl::id<2> idx) {
                                     int row = idx[0];
                                     int col = idx[1];
                                     double sum = 0.0;
                                     for (int i = 0; i < N; i++){
                                        sum += A[row*M+i]*B[i*N+col];
                                     }
                                     C[idx] = sum;
                                });
                           }else{
                                    std::cout<<"Printing second function"<<std::endl;
                                    for(int i=0;i<100;i++);
                           }

                        });
                       return e2;
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
  std::cout << "async_invoke_wait_on_policy: OK\n";
  return 0;
}

/*template<typename Policy, typename UniverseContainer, typename ResourceFunction>
int test_invoke_async_and_get_wait_list(UniverseContainer u, ResourceFunction&& f) {
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
  std::cout << "async_invoke_and_get_wait_list: OK\n";
  return 0;
}

template<typename Policy, typename UniverseContainer, typename ResourceFunction>
int test_invoke_async_and_get_wait_list_single_element(UniverseContainer u, ResourceFunction&& f) {
  using my_policy_t = Policy;
  my_policy_t p{u};

  const int N = 1;
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
  std::cout << "async_invoke_and_get_wait_list single element: OK\n";
  return 0;
}

template<typename Policy, typename UniverseContainer, typename ResourceFunction>
int test_invoke_async_and_get_wait_list_empty(UniverseContainer u, ResourceFunction&& f) {
  using my_policy_t = Policy;
  my_policy_t p{u};

  const int N = 0;
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
  std::cout << "async_invoke_and_get_wait_list empty list: OK\n";
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
  std::cout << "select_invoke: OK\n";
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
*/
#endif /* _ONEDPL_TEST_DYNAMIC_LOAD_UTILS_H */
