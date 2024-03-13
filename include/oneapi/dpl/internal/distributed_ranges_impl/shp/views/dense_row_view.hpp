// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <dr/detail/index.hpp>
#include <dr/detail/iterator_adaptor.hpp>
#include <dr/shp/containers/matrix_entry.hpp>
#include <iterator>

namespace dr::shp {
template <typename T, typename Iter> class dense_matrix_row_accessor {
public:
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;

  using scalar_value_type = std::iter_value_t<Iter>;
  using scalar_reference = std::iter_reference_t<Iter>;

  using value_type = dr::shp::matrix_entry<scalar_value_type, std::size_t>;

  using reference = dr::shp::matrix_ref<T, std::size_t, scalar_reference>;

  using iterator_category = std::random_access_iterator_tag;

  using iterator_accessor = dense_matrix_row_accessor;
  using const_iterator_accessor = iterator_accessor;
  using nonconst_iterator_accessor = iterator_accessor;

  using key_type = dr::index<>;

  constexpr dense_matrix_row_accessor() noexcept = default;
  constexpr ~dense_matrix_row_accessor() noexcept = default;
  constexpr dense_matrix_row_accessor(
      const dense_matrix_row_accessor &) noexcept = default;
  constexpr dense_matrix_row_accessor &
  operator=(const dense_matrix_row_accessor &) noexcept = default;

  constexpr dense_matrix_row_accessor(Iter data, std::size_t i,
                                      std::size_t j) noexcept
      : data_(data), i_(i), j_(j) {}

  constexpr dense_matrix_row_accessor &
  operator+=(difference_type offset) noexcept {
    j_ += offset;
    return *this;
  }

  constexpr bool operator==(const iterator_accessor &other) const noexcept {
    return j_ == other.j_;
  }

  constexpr difference_type
  operator-(const iterator_accessor &other) const noexcept {
    return difference_type(j_) - difference_type(other.j_);
  }

  constexpr bool operator<(const iterator_accessor &other) const noexcept {
    return j_ < other.j_;
  }

  constexpr reference operator*() const noexcept {
    return reference(key_type({i_, j_}), data_[j_]);
  }

private:
  size_type i_, j_;

  Iter data_;
};

template <typename T, typename Iter>
using dense_matrix_row_iterator =
    dr::iterator_adaptor<dense_matrix_row_accessor<T, Iter>>;

template <typename T, typename Iter> class dense_matrix_row_view {
public:
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;

  using scalar_reference = std::iter_reference_t<Iter>;

  using key_type = dr::index<>;
  using map_type = T;

  using iterator = dense_matrix_row_iterator<T, Iter>;

  dense_matrix_row_view(Iter data, size_type row_idx, size_type size)
      : data_(data), row_idx_(row_idx), size_(size) {}

  scalar_reference operator[](size_type idx) { return data_[idx]; }

  iterator begin() const { return iterator(data_, row_idx_, 0); }

  iterator end() const { return iterator(data_, row_idx_, size_); }

  size_type size() const noexcept { return size_; }

  Iter data_;
  size_type row_idx_;
  size_type size_;
};

template <std::random_access_iterator Iter>
dense_matrix_row_view(Iter, std::size_t, std::size_t)
    -> dense_matrix_row_view<std::iter_value_t<Iter>, Iter>;

} // namespace dr::shp
