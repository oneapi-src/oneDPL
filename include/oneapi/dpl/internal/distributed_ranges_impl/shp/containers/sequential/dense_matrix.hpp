// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <iterator>

#include <dr/detail/index.hpp>
#include <dr/detail/iterator_adaptor.hpp>
#include <dr/shp/containers/matrix_entry.hpp>
#include <dr/shp/views/dense_column_view.hpp>
#include <dr/shp/views/dense_matrix_iterator.hpp>
#include <dr/shp/views/dense_row_view.hpp>

namespace dr::shp {

template <typename T, typename Allocator = std::allocator<T>>
class dense_matrix {
public:
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;

  using allocator_type = Allocator;

  using scalar_pointer = typename std::allocator_traits<Allocator>::pointer;

  using scalar_reference = std::iter_reference_t<scalar_pointer>;
  using reference = dr::shp::matrix_ref<T, std::size_t, scalar_reference>;

  using key_type = dr::index<>;
  using map_type = T;

  using iterator = dense_matrix_iterator<T, scalar_pointer>;

  dense_matrix(key_type shape)
      : allocator_(Allocator()), shape_(shape), ld_(shape[1]) {
    data_ = allocator_.allocate(shape_[0] * shape_[1]);
  }

  dense_matrix(key_type shape, std::size_t ld)
    requires(std::is_default_constructible_v<Allocator>)
      : allocator_(Allocator()), shape_(shape), ld_(ld) {
    data_ = allocator_.allocate(shape_[0] * ld_);
  }

  dense_matrix(key_type shape, std::size_t ld, const Allocator &alloc)
      : allocator_(alloc), shape_(shape), ld_(ld) {
    data_ = allocator_.allocate(shape_[0] * ld_);
  }

  dense_matrix(dense_matrix &&other)
      : allocator_(other.allocator_), data_(other.data_), shape_(other.shape_),
        ld_(other.ld_) {
    other.null_data_();
  }

  dense_matrix &operator=(dense_matrix &&other) {
    deallocate_storage_();
    allocator_ = other.allocator_;
    data_ = other.data_;
    shape_ = other.shape_;
    ld_ = other.ld_;

    other.null_data_();
  }

  dense_matrix(const dense_matrix &other) = delete;
  dense_matrix &operator=(const dense_matrix &other) = delete;

  ~dense_matrix() { deallocate_storage_(); }

  key_type shape() const noexcept { return shape_; }

  size_type size() const noexcept { return shape()[0] * shape()[1]; }

  scalar_reference operator[](key_type idx) const {
    return data_[idx[0] * ld_ + idx[1]];
  }

  iterator begin() const {
    return iterator(data_, key_type{0, 0}, shape_, ld_);
  }

  iterator end() const {
    return iterator(data_, key_type{shape_[0], 0}, shape_, ld_);
  }

  auto row(size_type row_index) const {
    // return dense_matrix_row_view(data_ + row_index * ld_, row_index,
    // shape()[1]);
    auto row_elements = rng::views::iota(size_type(0), size_type(shape()[1]));
    scalar_pointer data = data_ + row_index * ld_;

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
    scalar_pointer data = data_ + column_index;
    size_type ld = ld_;

    return column_elements | rng::views::transform([=](auto row_index) {
             return reference(key_type(row_index, column_index),
                              data[row_index * ld]);
           });
  }

  scalar_pointer data() const { return data_; }

  size_type ld() const { return ld_; }

  /*
    auto local() const {
    }
    */

private:
  void deallocate_storage_() {
    if (data_ != nullptr) {
      allocator_.deallocate(data_, shape_[0] * ld_);
    }
  }

  void null_data_() {
    data_ = nullptr;
    shape_ = {0, 0};
    ld_ = 0;
  }

  allocator_type allocator_;
  scalar_pointer data_;
  key_type shape_;
  size_type ld_;
};

} // namespace dr::shp
