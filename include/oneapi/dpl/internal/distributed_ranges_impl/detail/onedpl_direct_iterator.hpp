// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <iterator>

#include <dr/detail/iterator_adaptor.hpp>

namespace dr {

namespace __detail {

template <std::random_access_iterator Iter> class direct_iterator {
public:
  using value_type = std::iter_value_t<Iter>;
  using difference_type = std::iter_difference_t<Iter>;
  using reference = std::iter_reference_t<Iter>;
  using iterator = direct_iterator<Iter>;

  using pointer = iterator;

  using iterator_category = std::random_access_iterator_tag;

  using is_passed_directly = ::std::true_type;

  direct_iterator(Iter iter) noexcept : iter_(iter) {}
  direct_iterator() noexcept = default;
  ~direct_iterator() noexcept = default;
  direct_iterator(const direct_iterator &) noexcept = default;
  direct_iterator &operator=(const direct_iterator &) noexcept = default;

  bool operator==(const direct_iterator &) const noexcept = default;
  bool operator!=(const direct_iterator &) const noexcept = default;

  iterator operator+(difference_type offset) const noexcept {
    return iterator(iter_ + offset);
  }

  iterator operator-(difference_type offset) const noexcept {
    return iterator(iter_ - offset);
  }

  difference_type operator-(iterator other) const noexcept {
    return iter_ - other.iter_;
  }

  bool operator<(iterator other) const noexcept { return iter_ < other.iter_; }

  bool operator>(iterator other) const noexcept { return iter_ > iter_; }

  bool operator<=(iterator other) const noexcept {
    return iter_ <= other.iter_;
  }

  bool operator>=(iterator other) const noexcept {
    return iter_ >= other.iter_;
  }

  iterator &operator++() noexcept {
    ++iter_;
    return *this;
  }

  iterator operator++(int) noexcept {
    iterator other = *this;
    ++(*this);
    return other;
  }

  iterator &operator--() noexcept {
    --iter_;
    return *this;
  }

  iterator operator--(int) noexcept {
    iterator other = *this;
    --(*this);
    return other;
  }

  iterator &operator+=(difference_type offset) noexcept {
    iter_ += offset;
    return *this;
  }

  iterator &operator-=(difference_type offset) noexcept {
    iter_ -= offset;
    return *this;
  }

  reference operator*() const noexcept { return *iter_; }

  reference operator[](difference_type offset) const noexcept {
    return reference(*(*this + offset));
  }

  friend iterator operator+(difference_type n, iterator iter) {
    return iter.iter_ + n;
  }

  Iter base() const noexcept { return iter_; }

private:
  Iter iter_;
};

} // namespace __detail

} // namespace dr
