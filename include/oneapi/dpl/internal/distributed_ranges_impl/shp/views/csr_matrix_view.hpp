// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <dr/detail/index.hpp>
#include <dr/shp/containers/matrix_entry.hpp>
#include <iterator>

namespace dr::shp {

template <typename T, typename I, typename TIter, typename IIter>
class csr_matrix_view_accessor {
public:
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;

  using scalar_type = std::iter_value_t<TIter>;
  using scalar_reference = std::iter_reference_t<TIter>;

  using index_type = I;

  using value_type = dr::shp::matrix_entry<scalar_type, I>;

  using reference = dr::shp::matrix_ref<T, I, scalar_reference>;

  using iterator_category = std::random_access_iterator_tag;

  using iterator_accessor = csr_matrix_view_accessor;
  using const_iterator_accessor = iterator_accessor;
  using nonconst_iterator_accessor = iterator_accessor;

  using key_type = dr::index<I>;

  constexpr csr_matrix_view_accessor() noexcept = default;
  constexpr ~csr_matrix_view_accessor() noexcept = default;
  constexpr csr_matrix_view_accessor(
      const csr_matrix_view_accessor &) noexcept = default;
  constexpr csr_matrix_view_accessor &
  operator=(const csr_matrix_view_accessor &) noexcept = default;

  constexpr csr_matrix_view_accessor(TIter values, IIter rowptr, IIter colind,
                                     size_type idx, index_type row,
                                     size_type row_dim) noexcept
      : values_(values), rowptr_(rowptr), colind_(colind), idx_(idx), row_(row),
        row_dim_(row_dim), idx_offset_(key_type{0, 0}) {
    fast_forward_row();
  }

  constexpr csr_matrix_view_accessor(TIter values, IIter rowptr, IIter colind,
                                     size_type idx, index_type row,
                                     size_type row_dim,
                                     key_type idx_offset) noexcept
      : values_(values), rowptr_(rowptr), colind_(colind), idx_(idx), row_(row),
        row_dim_(row_dim), idx_offset_(idx_offset) {
    fast_forward_row();
  }

  // Given that `idx_` has just been advanced to an element
  // possibly in a new row, advance `row_` to find the new row.
  // That is:
  // Advance `row_` until idx_ >= rowptr_[row_] && idx_ < rowptr_[row_+1]
  void fast_forward_row() noexcept {
    while (row_ < row_dim_ - 1 && idx_ >= rowptr_[row_ + 1]) {
      row_++;
    }
  }

  // Given that `idx_` has just been retreated to an element
  // possibly in a previous row, retreat `row_` to find the new row.
  // That is:
  // Retreat `row_` until idx_ >= rowptr_[row_] && idx_ < rowptr_[row_+1]
  void fast_backward_row() noexcept {
    while (idx_ < rowptr_[row_]) {
      row_--;
    }
  }

  constexpr csr_matrix_view_accessor &
  operator+=(difference_type offset) noexcept {
    idx_ += offset;
    if (offset < 0) {
      fast_backward_row();
    } else {
      fast_forward_row();
    }
    return *this;
  }

  constexpr bool operator==(const iterator_accessor &other) const noexcept {
    return idx_ == other.idx_;
  }

  constexpr difference_type
  operator-(const iterator_accessor &other) const noexcept {
    return difference_type(idx_) - difference_type(other.idx_);
  }

  constexpr bool operator<(const iterator_accessor &other) const noexcept {
    return idx_ < other.idx_;
  }

  constexpr reference operator*() const noexcept {
    return reference(
        key_type(row_ + idx_offset_[0], colind_[idx_] + idx_offset_[1]),
        values_[idx_]);
  }

private:
  TIter values_;
  IIter rowptr_;
  IIter colind_;
  size_type idx_;
  index_type row_;
  size_type row_dim_;
  key_type idx_offset_;
};

template <typename T, typename I, typename TIter, typename IIter>
using csr_matrix_view_iterator =
    dr::iterator_adaptor<csr_matrix_view_accessor<T, I, TIter, IIter>>;

template <typename T, typename I, typename TIter = T *, typename IIter = I *>
class csr_matrix_view
    : public rng::view_interface<csr_matrix_view<T, I, TIter, IIter>> {
public:
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;

  using scalar_reference = std::iter_reference_t<TIter>;
  using reference = dr::shp::matrix_ref<T, I, scalar_reference>;

  using scalar_type = T;
  using index_type = I;

  using key_type = dr::index<I>;
  using map_type = T;

  using iterator = csr_matrix_view_iterator<T, I, TIter, IIter>;

  csr_matrix_view(TIter values, IIter rowptr, IIter colind, key_type shape,
                  size_type nnz, size_type rank)
      : values_(values), rowptr_(rowptr), colind_(colind), shape_(shape),
        nnz_(nnz), rank_(rank), idx_offset_(key_type{0, 0}) {}

  csr_matrix_view(TIter values, IIter rowptr, IIter colind, key_type shape,
                  size_type nnz, size_type rank, key_type idx_offset)
      : values_(values), rowptr_(rowptr), colind_(colind), shape_(shape),
        nnz_(nnz), rank_(rank), idx_offset_(idx_offset) {}

  key_type shape() const noexcept { return shape_; }

  size_type size() const noexcept { return nnz_; }

  std::size_t rank() const { return rank_; }

  iterator begin() const {
    return iterator(values_, rowptr_, colind_, 0, 0, shape()[1], idx_offset_);
  }

  iterator end() const {
    return iterator(values_, rowptr_, colind_, nnz_, shape()[1], shape()[1],
                    idx_offset_);
  }

  auto row(I row_index) const {
    I first = rowptr_[row_index];
    I last = rowptr_[row_index + 1];

    TIter values = values_;
    IIter colind = colind_;

    auto row_elements = rng::views::iota(first, last);

    return row_elements | rng::views::transform([=](auto idx) {
             return reference(key_type(row_index, colind[idx]), values[idx]);
           });
  }

  auto submatrix(key_type rows, key_type columns) const {
    return rng::views::iota(rows[0], rows[1]) |
           rng::views::transform([=, *this](auto &&row_index) {
             return row(row_index) | rng::views::drop_while([=](auto &&e) {
                      auto &&[index, v] = e;
                      return index[1] < columns[0];
                    }) |
                    rng::views::take_while([=](auto &&e) {
                      auto &&[index, v] = e;
                      return index[1] < columns[1];
                    }) |
                    rng::views::transform([=](auto &&elem) {
                      auto &&[index, v] = elem;
                      auto &&[i, j] = index;
                      return reference(key_type(i - rows[0], j - columns[0]),
                                       v);
                    });
           }) |
           rng::views::join;
  }

  auto values_data() const { return values_; }

  auto rowptr_data() const { return rowptr_; }

  auto colind_data() const { return colind_; }

private:
  TIter values_;
  IIter rowptr_;
  IIter colind_;

  key_type shape_;
  size_type nnz_;

  size_type rank_;
  key_type idx_offset_;
};

template <typename TIter, typename IIter, typename... Args>
csr_matrix_view(TIter, IIter, IIter, Args &&...)
    -> csr_matrix_view<std::iter_value_t<TIter>, std::iter_value_t<IIter>,
                       TIter, IIter>;

} // namespace dr::shp
