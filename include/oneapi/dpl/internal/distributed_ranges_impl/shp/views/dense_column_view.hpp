// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <dr/detail/index.hpp>
#include <dr/shp/containers/matrix_entry.hpp>
#include <iterator>

namespace dr::shp {
template <typename T, typename Iter> class dense_matrix_column_accessor {
public:
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;

  using scalar_value_type = std::iter_value_t<Iter>;
  using scalar_reference = std::iter_reference_t<Iter>;

  using value_type = dr::shp::matrix_entry<scalar_value_type, std::size_t>;

  using reference = dr::shp::matrix_ref<T, std::size_t, scalar_reference>;

  using iterator_category = std::random_access_iterator_tag;

  using iterator_accessor = dense_matrix_column_accessor;
  using const_iterator_accessor = iterator_accessor;
  using nonconst_iterator_accessor = iterator_accessor;

  using key_type = dr::index<>;

  constexpr dense_matrix_column_accessor() noexcept = default;
  constexpr ~dense_matrix_column_accessor() noexcept = default;
  constexpr dense_matrix_column_accessor(
      const dense_matrix_column_accessor &) noexcept = default;
  constexpr dense_matrix_column_accessor &
  operator=(const dense_matrix_column_accessor &) noexcept = default;

  constexpr dense_matrix_column_accessor(Iter data, std::size_t i,
                                         std::size_t j, std::size_t ld) noexcept
      : data_(data), i_(i), j_(j), ld_(ld) {}

  constexpr dense_matrix_column_accessor &
  operator+=(difference_type offset) noexcept {
    i_ += offset;
    return *this;
  }

  constexpr bool operator==(const iterator_accessor &other) const noexcept {
    return i_ == other.i_;
  }

  constexpr difference_type
  operator-(const iterator_accessor &other) const noexcept {
    return difference_type(i_) - difference_type(other.i_);
  }

  constexpr bool operator<(const iterator_accessor &other) const noexcept {
    return i_ < other.i_;
  }

  constexpr reference operator*() const noexcept {
    return reference(key_type({i_, j_}), data_[i_ * ld_]);
  }

private:
  size_type i_, j_;
  size_type ld_;

  Iter data_;
};

template <typename T, typename Iter>
using dense_matrix_column_iterator =
    dr::iterator_adaptor<dense_matrix_column_accessor<T, Iter>>;

template <typename T, typename Iter> class dense_matrix_column_view {
public:
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;

  using scalar_reference = std::iter_reference_t<Iter>;

  using key_type = dr::index<>;
  using map_type = T;

  using iterator = dense_matrix_column_iterator<T, Iter>;

  dense_matrix_column_view(Iter data, size_type column_idx, size_type size,
                           size_type ld)
      : data_(data), column_idx_(column_idx), size_(size), ld_(ld) {}

  scalar_reference operator[](size_type idx) { return data_[idx * ld_]; }

  iterator begin() const { return iterator(data_, 0, column_idx_, ld_); }

  iterator end() const { return iterator(data_, size_, column_idx_, ld_); }

  size_type size() const noexcept { return size_; }

  Iter data_;
  size_type column_idx_;
  size_type size_;
  size_type ld_;
};

template <std::random_access_iterator Iter>
dense_matrix_column_view(Iter, std::size_t, std::size_t, std::size_t)
    -> dense_matrix_column_view<std::iter_value_t<Iter>, Iter>;

} // namespace dr::shp
