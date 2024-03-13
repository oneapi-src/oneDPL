// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <iterator>

#include <dr/detail/index.hpp>
#include <dr/detail/iterator_adaptor.hpp>
#include <dr/shp/containers/matrix_entry.hpp>
#include <dr/shp/containers/sequential/dense_matrix.hpp>
#include <dr/shp/views/dense_column_view.hpp>
#include <dr/shp/views/dense_matrix_iterator.hpp>
#include <dr/shp/views/dense_row_view.hpp>

namespace dr::shp {

template <typename T, typename Iter = T *>
class dense_matrix_view
    : public rng::view_interface<dense_matrix_view<T, Iter>> {
public:
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;

  using scalar_reference = std::iter_reference_t<Iter>;
  using reference = dr::shp::matrix_ref<T, std::size_t, scalar_reference>;

  using key_type = dr::index<>;
  using map_type = T;

  using iterator = dense_matrix_view_iterator<T, Iter>;

  dense_matrix_view(Iter data, key_type shape, size_type ld, size_type rank)
      : data_(data), shape_(shape), idx_offset_(key_type{0, 0}), ld_(ld),
        rank_(rank) {}

  dense_matrix_view(Iter data, key_type shape, key_type idx_offset,
                    size_type ld, size_type rank)
      : data_(data), shape_(shape), idx_offset_(idx_offset), ld_(ld),
        rank_(rank) {}

  template <typename Allocator>
    requires(std::is_same_v<typename std::allocator_traits<Allocator>::pointer,
                            Iter>)
  dense_matrix_view(dense_matrix<T, Allocator> &m)
      : data_(m.data()), shape_(m.shape()), idx_offset_(key_type{0, 0}),
        ld_(m.ld()), rank_(0) {}

  key_type shape() const noexcept { return shape_; }

  size_type size() const noexcept { return shape()[0] * shape()[1]; }

  scalar_reference operator[](key_type idx) const {
    return data_[idx[0] * ld_ + idx[1]];
  }

  iterator begin() const {
    return iterator(data_, key_type{0, 0}, idx_offset_, shape_, ld_);
  }

  iterator end() const {
    return iterator(data_, key_type{shape_[0], 0}, idx_offset_, shape_, ld_);
  }

  auto row(size_type row_index) const {
    // return dense_matrix_row_view(data_ + row_index * ld_, row_index,
    // shape()[1]);
    auto row_elements = rng::views::iota(size_type(0), size_type(shape()[1]));
    Iter data = data_ + row_index * ld_;

    return row_elements | rng::views::transform([=](auto column_index) {
             return reference(key_type(row_index, column_index),
                              data[column_index]);
           });
  }

  auto column(size_type column_index) const {
    // return dense_matrix_column_view(data_ + column_index, column_index,
    // shape()[0], ld_);
    auto column_elements =
        rng::views::iota(size_type(0), size_type(shape()[0]));
    Iter data = data_ + column_index;
    size_type ld = ld_;

    return column_elements | rng::views::transform([=](auto row_index) {
             return reference(key_type(row_index, column_index),
                              data[row_index * ld]);
           });
  }

  Iter data() const { return data_; }

  std::size_t rank() const { return rank_; }

  size_type ld() const { return ld_; }

  auto local() const {
    auto local_data = __detail::local(data_);
    return dense_matrix_view<T, decltype(local_data)>(
        local_data, shape_, idx_offset_, ld(), rank());
  }

private:
  Iter data_;
  key_type shape_;
  key_type idx_offset_;
  size_type ld_;
  size_type rank_;
};

template <std::random_access_iterator Iter>
dense_matrix_view(Iter, dr::index<>, std::size_t)
    -> dense_matrix_view<std::iter_value_t<Iter>, Iter>;

template <std::random_access_iterator Iter>
dense_matrix_view(Iter, dr::index<>)
    -> dense_matrix_view<std::iter_value_t<Iter>, Iter>;

template <typename T, typename Allocator>
dense_matrix_view(dense_matrix<T, Allocator> &)
    -> dense_matrix_view<T, typename std::allocator_traits<Allocator>::pointer>;

} // namespace dr::shp
