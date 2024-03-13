// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cstring>

namespace dr {

template <typename T> struct default_memory {
  using value_type = T;
  std::allocator<T> std_allocator;

  T *allocate(std::size_t size) {
    auto p = std_allocator.allocate(size);
    assert(p != nullptr);
    memset(p, 0, sizeof(T) * size);
    return p;
  }

  template <typename F> F *allocate(std::size_t size) {
    std::allocator<F> allocator;
    auto p = allocator.allocate(size);
    assert(p != nullptr);
    memset(p, 0, sizeof(F) * size);
    return p;
  }

  constexpr void deallocate(T *p, std::size_t n) {
    std_allocator.deallocate(p, n);
  }

  template <typename F> void deallocate(F *p, std::size_t n) {
    std::allocator<F> allocator;
    allocator.deallocate(p, n);
    p = nullptr;
  }

  void memcpy(void *dst, const void *src, std::size_t numBytes) {
    std::memcpy(dst, src, numBytes);
  }

  template <typename F> void offload(F lambda) { lambda(); }
};

#ifdef SYCL_LANGUAGE_VERSION
template <typename T> struct sycl_memory {
  using value_type = T;
  using device_type = sycl::device;

  sycl::device device_;
  sycl::context context_;
  sycl::usm::alloc kind_;
  std::size_t alignment_;
  sycl::queue offload_queue_;

  sycl_memory(sycl::queue queue,
              sycl::usm::alloc kind = sycl::usm::alloc::shared,
              std::size_t alignment = 1)
      : device_(queue.get_device()), context_(queue.get_context()), kind_(kind),
        alignment_(alignment), offload_queue_(queue) {}

  T *allocate(std::size_t n) {
    auto p = sycl::aligned_alloc<T>(alignment_, n, device_, context_, kind_);
    assert(p != nullptr);
    return p;
  }

  template <typename F> F *allocate(std::size_t n) {
    auto p = sycl::aligned_alloc<F>(alignment_, n, device_, context_, kind_);
    assert(p != nullptr);
    return p;
  }

  void deallocate(T *p, std::size_t n) {
    assert(p != nullptr);
    sycl::free(p, context_);
    p = nullptr;
  }

  template <typename F> void deallocate(F *p, std::size_t n) {
    assert(p != nullptr);
    sycl::free(p, context_);
    p = nullptr;
  }

  void memcpy(void *dst, const void *src, std::size_t numBytes) {
    assert(dst != nullptr);
    assert(src != nullptr);
    offload_queue_.memcpy(dst, src, numBytes).wait();
  }

  template <typename F> void offload(F lambda) {
    if (kind_ == sycl::usm::alloc::device) {
      offload_queue_.single_task(lambda).wait();
    } else {
      lambda();
    }
  }
};
#endif

} // namespace dr
