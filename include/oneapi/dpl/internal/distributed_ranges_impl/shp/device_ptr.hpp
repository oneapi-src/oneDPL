// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <type_traits>

#include <sycl/sycl.hpp>

#include <dr/shp/device_ref.hpp>

namespace dr::shp {

template <typename T>
  requires(std::is_trivially_copyable_v<T> || std::is_void_v<T>)
class device_ptr {
public:
  using value_type = T;
  using difference_type = std::ptrdiff_t;
  using pointer = device_ptr<T>;
  using const_pointer = device_ptr<std::add_const_t<T>>;
  using nonconst_pointer = device_ptr<std::remove_const_t<T>>;
  using iterator_category = std::random_access_iterator_tag;
  using reference = device_ref<T>;

  device_ptr(T *pointer) noexcept : pointer_(pointer) {}
  device_ptr() noexcept = default;
  ~device_ptr() noexcept = default;
  device_ptr(const device_ptr &) noexcept = default;
  device_ptr &operator=(const device_ptr &) noexcept = default;

  device_ptr(std::nullptr_t) noexcept : pointer_(nullptr) {}

  device_ptr &operator=(std::nullptr_t) noexcept {
    pointer_ = nullptr;
    return *this;
  }

  operator device_ptr<void>() const noexcept
    requires(!std::is_void_v<T>)
  {
    return device_ptr<void>(reinterpret_cast<void *>(pointer_));
  }

  operator device_ptr<const void>() const noexcept
    requires(!std::is_void_v<T>)
  {
    return device_ptr<const void>(reinterpret_cast<const void *>(pointer_));
  }

  operator const_pointer() const noexcept
    requires(!std::is_const_v<T>)
  {
    return const_pointer(pointer_);
  }

  bool operator==(std::nullptr_t) const noexcept { return pointer_ == nullptr; }
  bool operator!=(std::nullptr_t) const noexcept { return pointer_ != nullptr; }

  bool operator==(const device_ptr &) const noexcept = default;
  bool operator!=(const device_ptr &) const noexcept = default;

  pointer operator+(difference_type offset) const noexcept {
    return pointer(pointer_ + offset);
  }
  pointer operator-(difference_type offset) const noexcept {
    return pointer(pointer_ - offset);
  }

  difference_type operator-(const_pointer other) const noexcept
    requires(!std::is_const_v<T>)
  {
    return pointer_ - other.pointer_;
  }
  difference_type operator-(pointer other) const noexcept {
    return pointer_ - other.pointer_;
  }

  bool operator<(const_pointer other) const noexcept {
    return pointer_ < other.pointer_;
  }
  bool operator>(const_pointer other) const noexcept {
    return pointer_ > other.pointer_;
  }
  bool operator<=(const_pointer other) const noexcept {
    return pointer_ <= other.pointer_;
  }
  bool operator>=(const_pointer other) const noexcept {
    return pointer_ >= other.pointer_;
  }

  pointer &operator++() noexcept {
    ++pointer_;
    return *this;
  }

  pointer operator++(int) noexcept {
    pointer other = *this;
    ++(*this);
    return other;
  }

  pointer &operator--() noexcept {
    --pointer_;
    return *this;
  }

  pointer operator--(int) noexcept {
    pointer other = *this;
    --(*this);
    return other;
  }

  pointer &operator+=(difference_type offset) noexcept {
    pointer_ += offset;
    return *this;
  }

  pointer &operator-=(difference_type offset) noexcept {
    pointer_ -= offset;
    return *this;
  }

  reference operator*() const noexcept { return reference(pointer_); }

  reference operator[](difference_type offset) const noexcept {
    return reference(*(*this + offset));
  }

  T *get_raw_pointer() const noexcept { return pointer_; }

  friend pointer operator+(difference_type n, pointer iter) { return iter + n; }

  T *local() const noexcept { return pointer_; }

  friend const_pointer;
  friend nonconst_pointer;

private:
  T *pointer_;
};

} // namespace dr::shp
