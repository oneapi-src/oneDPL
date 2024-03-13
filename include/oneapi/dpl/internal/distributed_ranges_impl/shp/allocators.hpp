// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <type_traits>

#include <sycl/sycl.hpp>

#include <dr/shp/device_ptr.hpp>

namespace dr::shp {

template <typename T>
using shared_allocator = sycl::usm_allocator<T, sycl::usm::alloc::shared>;

template <typename T, std::size_t Alignment = 0>
  requires(std::is_trivially_copyable_v<T>)
class device_allocator {
public:
  using value_type = T;
  using pointer = device_ptr<T>;
  using const_pointer = device_ptr<T>;
  using reference = device_ref<T>;
  using const_reference = device_ref<const T>;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;

  template <typename U>
  device_allocator(const device_allocator<U, Alignment> &other) noexcept
      : device_(other.get_device()), context_(other.get_context()) {}

  device_allocator(const sycl::queue &q) noexcept
      : device_(q.get_device()), context_(q.get_context()) {}
  device_allocator(const sycl::context &ctxt, const sycl::device &dev) noexcept
      : device_(dev), context_(ctxt) {}

  device_allocator(const device_allocator &) = default;
  device_allocator &operator=(const device_allocator &) = default;
  ~device_allocator() = default;

  using is_always_equal = std::false_type;

  pointer allocate(std::size_t size) {
    if constexpr (Alignment == 0) {
      return pointer(sycl::malloc_device<T>(size, device_, context_));
    } else {
      return pointer(
          sycl::aligned_alloc_device<T>(Alignment, size, device_, context_));
    }
  }

  void deallocate(pointer ptr, std::size_t n) {
    sycl::free(ptr.get_raw_pointer(), context_);
  }

  bool operator==(const device_allocator &) const = default;
  bool operator!=(const device_allocator &) const = default;

  template <typename U> struct rebind {
    using other = device_allocator<U, Alignment>;
  };

  sycl::device get_device() const noexcept { return device_; }

  sycl::context get_context() const noexcept { return context_; }

private:
  sycl::device device_;
  sycl::context context_;
};

template <typename Allocator> class buffered_allocator {
public:
  using value_type = typename std::allocator_traits<Allocator>::value_type;
  using pointer = typename std::allocator_traits<Allocator>::pointer;
  using const_pointer =
      typename std::allocator_traits<Allocator>::const_pointer;
  using size_type = typename std::allocator_traits<Allocator>::size_type;
  using difference_type =
      typename std::allocator_traits<Allocator>::difference_type;

  buffered_allocator(const Allocator &alloc, std::size_t buffer_size,
                     std::size_t n_buffers)
      : alloc_(alloc), buffer_size_(buffer_size),
        free_buffers_(new std::vector<pointer>()),
        buffers_(new std::vector<pointer>()) {
    for (std::size_t i = 0; i < n_buffers; i++) {
      buffers_->push_back(alloc_.allocate(buffer_size_));
    }
    free_buffers_->assign(buffers_->begin(), buffers_->end());
  }

  ~buffered_allocator() {
    if (buffers_.use_count() == 1) {
      for (auto &&buffer : *buffers_) {
        alloc_.deallocate(buffer, buffer_size_);
      }
    }
  }

  using is_always_equal = std::false_type;

  pointer allocate(std::size_t size) {
    if (size > buffer_size_ || free_buffers_->empty()) {
      throw std::bad_alloc();
    } else {
      pointer buffer = free_buffers_->back();
      free_buffers_->pop_back();
      return buffer;
    }
  }

  void deallocate(pointer ptr, std::size_t n) { free_buffers_->push_back(ptr); }

  bool operator==(const buffered_allocator &) const = default;
  bool operator!=(const buffered_allocator &) const = default;

private:
  Allocator alloc_;
  std::size_t buffer_size_;
  std::shared_ptr<std::vector<pointer>> free_buffers_;
  std::shared_ptr<std::vector<pointer>> buffers_;
};

} // namespace dr::shp
