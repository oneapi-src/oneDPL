// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <iterator>

#include <dr/detail/index.hpp>
#include <dr/detail/iterator_adaptor.hpp>
#include <dr/shp/containers/matrix_entry.hpp>
#include <dr/shp/views/dense_column_view.hpp>
#include <dr/shp/views/dense_row_view.hpp>

namespace dr::shp {

template <typename T, typename Iter> class dense_matrix_accessor {
public:
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;

  using scalar_type = std::iter_value_t<Iter>;
  using scalar_reference = std::iter_reference_t<Iter>;

  using value_type = dr::shp::matrix_entry<scalar_type, std::size_t>;

  using reference = dr::shp::matrix_ref<T, std::size_t, scalar_reference>;

  using iterator_category = std::random_access_iterator_tag;

  using iterator_accessor = dense_matrix_accessor;
  using const_iterator_accessor = iterator_accessor;
  using nonconst_iterator_accessor = iterator_accessor;

  using key_type = dr::index<>;

  constexpr dense_matrix_accessor() noexcept = default;
  constexpr ~dense_matrix_accessor() noexcept = default;
  constexpr dense_matrix_accessor(const dense_matrix_accessor &) noexcept =
      default;
  constexpr dense_matrix_accessor &
  operator=(const dense_matrix_accessor &) noexcept = default;

  constexpr dense_matrix_accessor(Iter data, key_type idx,
                                  key_type matrix_shape, size_type ld) noexcept
      : data_(data), idx_(idx), matrix_shape_(matrix_shape), ld_(ld),
        idx_offset_({0, 0}) {}

  constexpr dense_matrix_accessor(Iter data, key_type idx, key_type idx_offset,
                                  key_type matrix_shape, size_type ld) noexcept
      : data_(data), idx_(idx), matrix_shape_(matrix_shape), ld_(ld),
        idx_offset_(idx_offset) {}

  constexpr dense_matrix_accessor &operator+=(difference_type offset) noexcept {
    size_type new_idx = get_global_idx() + offset;
    idx_ = {new_idx / matrix_shape_[1], new_idx % matrix_shape_[1]};

    return *this;
  }

  constexpr bool operator==(const iterator_accessor &other) const noexcept {
    return idx_ == other.idx_;
  }

  constexpr difference_type
  operator-(const iterator_accessor &other) const noexcept {
    return difference_type(get_global_idx()) - other.get_global_idx();
  }

  constexpr bool operator<(const iterator_accessor &other) const noexcept {
    if (idx_[0] < other.idx_[0]) {
      return true;
    } else if (idx_[0] == other.idx_[0]) {
      return idx_[1] < other.idx_[1];
    } else {
      return false;
    }
  }

  constexpr reference operator*() const noexcept {
    return reference(
        key_type(idx_[0] + idx_offset_[0], idx_[1] + idx_offset_[1]),
        data_[idx_[0] * ld_ + idx_[1]]);
  }

  Iter data() const noexcept { return data_; }

private:
  size_type get_global_idx() const noexcept {
    return idx_[0] * matrix_shape_[1] + idx_[1];
  }

private:
  Iter data_;
  key_type idx_;
  key_type matrix_shape_;
  size_type ld_;

  key_type idx_offset_;
};

template <typename T, typename Iter>
using dense_matrix_iterator =
    dr::iterator_adaptor<dense_matrix_accessor<T, Iter>>;

template <typename T, typename Iter>
using dense_matrix_view_iterator = dense_matrix_iterator<T, Iter>;

} // namespace dr::shp
