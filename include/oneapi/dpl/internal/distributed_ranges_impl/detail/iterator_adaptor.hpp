// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <iterator>
#include <type_traits>

#include <dr/detail/ranges_shim.hpp>

namespace dr {

namespace {

template <typename R>
concept has_segments_method = requires(R r) {
  { r.segments() };
};

} // namespace

template <typename Accessor> class iterator_adaptor {
public:
  using accessor_type = Accessor;
  using const_accessor_type = typename Accessor::const_iterator_accessor;
  using nonconst_accessor_type = typename Accessor::nonconst_iterator_accessor;

  using difference_type = typename Accessor::difference_type;
  using value_type = typename Accessor::value_type;
  using iterator = iterator_adaptor<accessor_type>;
  using const_iterator = iterator_adaptor<const_accessor_type>;
  using reference = typename Accessor::reference;
  using iterator_category = typename Accessor::iterator_category;

  using nonconst_iterator = iterator_adaptor<nonconst_accessor_type>;

  static_assert(std::is_same_v<iterator, iterator_adaptor<Accessor>>);

  iterator_adaptor() = default;
  ~iterator_adaptor() = default;
  iterator_adaptor(const iterator_adaptor &) = default;
  iterator_adaptor &operator=(const iterator_adaptor &) = default;

  template <typename... Args>
    requires(
        sizeof...(Args) >= 1 &&
        !((sizeof...(Args) == 1 &&
           (std::is_same_v<nonconst_iterator, std::decay_t<Args>> || ...)) ||
          (std::is_same_v<const_iterator, std::decay_t<Args>> || ...) ||
          (std::is_same_v<nonconst_accessor_type, std::decay_t<Args>> || ...) ||
          (std::is_same_v<const_accessor_type, std::decay_t<Args>> || ...)) &&
        std::is_constructible_v<accessor_type, Args...>)
  iterator_adaptor(Args &&...args) : accessor_(std::forward<Args>(args)...) {}

  iterator_adaptor(const accessor_type &accessor) : accessor_(accessor) {}
  iterator_adaptor(const const_accessor_type &accessor)
    requires(!std::is_same_v<accessor_type, const_accessor_type>)
      : accessor_(accessor) {}

  operator const_iterator() const
    requires(!std::is_same_v<iterator, const_iterator>)
  {
    return const_iterator(accessor_);
  }

  bool operator==(const_iterator other) const {
    return accessor_ == other.accessor_;
  }

  bool operator!=(const_iterator other) const { return !(*this == other); }

  bool operator<(const_iterator other) const
    requires(std::is_same_v<iterator_category, std::random_access_iterator_tag>)
  {
    return accessor_ < other.accessor_;
  }

  bool operator<=(const_iterator other) const
    requires(std::is_same_v<iterator_category, std::random_access_iterator_tag>)
  {
    return *this < other || *this == other;
  }

  bool operator>(const_iterator other) const
    requires(std::is_same_v<iterator_category, std::random_access_iterator_tag>)
  {
    return !(*this <= other);
  }

  bool operator>=(const_iterator other) const
    requires(std::is_same_v<iterator_category, std::random_access_iterator_tag>)
  {
    return !(*this < other);
  }

  reference operator*() const { return *accessor_; }

  reference operator[](difference_type offset) const
    requires(std::is_same_v<iterator_category, std::random_access_iterator_tag>)
  {
    return *(*this + offset);
  }

  iterator &operator+=(difference_type offset) noexcept
    requires(std::is_same_v<iterator_category, std::random_access_iterator_tag>)
  {
    accessor_ += offset;
    return *this;
  }

  iterator &operator-=(difference_type offset) noexcept
    requires(std::is_same_v<iterator_category, std::random_access_iterator_tag>)
  {
    accessor_ += -offset;
    return *this;
  }

  iterator operator+(difference_type offset) const
    requires(std::is_same_v<iterator_category, std::random_access_iterator_tag>)
  {
    iterator other = *this;
    other += offset;
    return other;
  }

  iterator operator-(difference_type offset) const
    requires(std::is_same_v<iterator_category, std::random_access_iterator_tag>)
  {
    iterator other = *this;
    other += -offset;
    return other;
  }

  difference_type operator-(const_iterator other) const
    requires(std::is_same_v<iterator_category, std::random_access_iterator_tag>)
  {
    return accessor_ - other.accessor_;
  }

  iterator &operator++() noexcept
    requires(std::is_same_v<iterator_category, std::random_access_iterator_tag>)
  {
    *this += 1;
    return *this;
  }

  iterator &operator++() noexcept
    requires(
        !std::is_same_v<iterator_category, std::random_access_iterator_tag>)
  {
    ++accessor_;
    return *this;
  }

  iterator operator++(int) noexcept {
    iterator other = *this;
    ++(*this);
    return other;
  }

  iterator &operator--() noexcept
    requires(
        std::is_same_v<iterator_category, std::random_access_iterator_tag> ||
        std::is_same_v<iterator_category, std::bidirectional_iterator_tag>)
  {
    *this += -1;
    return *this;
  }

  iterator operator--(int) noexcept
    requires(
        std::is_same_v<iterator_category, std::random_access_iterator_tag> ||
        std::is_same_v<iterator_category, std::bidirectional_iterator_tag>)
  {
    iterator other = *this;
    --(*this);
    return other;
  }

  auto segments() const noexcept
    requires(has_segments_method<accessor_type>)
  {
    return accessor_.segments();
  }

  friend iterator operator+(difference_type n, iterator iter)
    requires(std::is_same_v<iterator_category, std::random_access_iterator_tag>)
  {
    return iter + n;
  }

private:
  friend const_iterator;
  friend nonconst_iterator;

  accessor_type accessor_;
};

} // namespace dr
