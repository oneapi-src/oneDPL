// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <memory>

#include <dr/detail/index.hpp>
#include <dr/detail/owning_view.hpp>
#include <dr/shp/containers/matrix_entry.hpp>
#include <dr/shp/containers/matrix_partition.hpp>
#include <dr/shp/containers/sequential/dense_matrix.hpp>
#include <dr/shp/device_vector.hpp>
#include <dr/shp/future.hpp>
#include <dr/shp/views/dense_matrix_view.hpp>

namespace dr::shp {

template <typename T, typename L> class distributed_dense_matrix_accessor {
public:
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;

  using scalar_value_type = rng::range_value_t<L>;
  using scalar_reference = rng::range_reference_t<L>;

  using value_type = dr::shp::matrix_entry<scalar_value_type, std::size_t>;

  using reference = dr::shp::matrix_ref<T, std::size_t, scalar_reference>;

  using iterator_category = std::random_access_iterator_tag;

  using iterator_accessor = distributed_dense_matrix_accessor;
  using const_iterator_accessor = iterator_accessor;
  using nonconst_iterator_accessor = iterator_accessor;

  using tile_type = L;

  using key_type = dr::index<>;

  constexpr distributed_dense_matrix_accessor() noexcept = default;
  constexpr ~distributed_dense_matrix_accessor() noexcept = default;
  constexpr distributed_dense_matrix_accessor(
      const distributed_dense_matrix_accessor &) noexcept = default;
  constexpr distributed_dense_matrix_accessor &
  operator=(const distributed_dense_matrix_accessor &) noexcept = default;

  constexpr distributed_dense_matrix_accessor(
      std::span<tile_type> tiles, key_type grid_idx, key_type tile_idx,
      key_type grid_shape, key_type tile_shape, key_type matrix_shape) noexcept
      : grid_idx_(grid_idx), tile_idx_(tile_idx), grid_shape_(grid_shape),
        tile_shape_(tile_shape), matrix_shape_(matrix_shape), tiles_(tiles) {}

  constexpr distributed_dense_matrix_accessor &
  operator+=(difference_type offset) noexcept {
    std::size_t new_global_idx_ = get_global_idx_() + offset;
    key_type new_global_idx = {new_global_idx_ / matrix_shape_[1],
                               new_global_idx_ % matrix_shape_[1]};
    key_type new_grid_idx = {new_global_idx[0] / tile_shape_[0],
                             new_global_idx[1] / tile_shape_[1]};

    key_type new_tile_idx = {new_global_idx[0] % tile_shape_[0],
                             new_global_idx[1] % tile_shape_[1]};

    grid_idx_ = new_grid_idx;
    tile_idx_ = new_tile_idx;
    return *this;
  }

  constexpr bool operator==(const iterator_accessor &other) const noexcept {
    return grid_idx_ == other.grid_idx_ && tile_idx_ == other.tile_idx_;
  }

  constexpr difference_type
  operator-(const iterator_accessor &other) const noexcept {
    return difference_type(get_global_idx_()) - other.get_global_idx_();
  }

  constexpr bool operator<(const iterator_accessor &other) const noexcept {
    if (get_grid_idx() < other.get_grid_idx()) {
      return true;
    } else if (get_grid_idx() == other.get_grid_idx()) {
      return get_local_idx() < other.get_local_idx();
    } else {
      return false;
    }
  }

  constexpr reference operator*() const noexcept {
    auto &&tile = tiles_[get_grid_idx()];
    auto &&value = tile[get_local_idx()];
    key_type idx = {tile_idx_[0] + grid_idx_[0] * tile_shape_[0],
                    tile_idx_[1] + grid_idx_[1] * tile_shape_[1]};
    return reference(idx, value);
  }

private:
  size_type get_global_idx_() const noexcept {
    auto gidx = get_global_idx();
    return gidx[0] * matrix_shape_[1] + gidx[1];
  }

  key_type get_global_idx() const noexcept {
    return {grid_idx_[0] * tile_shape_[0] + tile_idx_[0],
            grid_idx_[1] * tile_shape_[1] + tile_idx_[1]};
  }

  size_type get_grid_idx() const noexcept {
    return grid_idx_[0] * grid_shape_[1] + grid_idx_[1];
  }

  size_type get_local_idx() const noexcept {
    return tile_idx_[0] * tile_shape_[1] + tile_idx_[1];
  }

  size_type get_tile_size() const noexcept {
    return tile_shape_[0] * tile_shape_[1];
  }

private:
  key_type grid_idx_;
  key_type tile_idx_;

  key_type grid_shape_;
  key_type tile_shape_;
  key_type matrix_shape_;

  std::span<tile_type> tiles_;
};

template <typename T, typename L>
using distributed_dense_matrix_iterator =
    dr::iterator_adaptor<distributed_dense_matrix_accessor<T, L>>;

template <typename T> class distributed_dense_matrix {
public:
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;

  using value_type = dr::shp::matrix_entry<T>;

  using scalar_reference = rng::range_reference_t<
      dr::shp::device_vector<T, dr::shp::device_allocator<T>>>;
  using const_scalar_reference = rng::range_reference_t<
      const dr::shp::device_vector<T, dr::shp::device_allocator<T>>>;

  using reference = dr::shp::matrix_ref<T, scalar_reference>;
  using const_reference = dr::shp::matrix_ref<const T, const_scalar_reference>;

  using key_type = dr::index<>;

  using iterator = distributed_dense_matrix_iterator<
      T, dr::shp::device_vector<T, dr::shp::device_allocator<T>>>;

  distributed_dense_matrix(key_type shape)
      : shape_(shape), partition_(new dr::shp::block_cyclic()) {
    init_();
  }

  distributed_dense_matrix(key_type shape, const matrix_partition &partition)
      : shape_(shape), partition_(partition.clone()) {
    init_();
  }

  size_type size() const noexcept { return shape()[0] * shape()[1]; }

  key_type shape() const noexcept { return shape_; }

  scalar_reference operator[](key_type index) {
    std::size_t tile_i = index[0] / tile_shape_[0];
    std::size_t tile_j = index[1] / tile_shape_[1];

    std::size_t local_i = index[0] % tile_shape_[0];
    std::size_t local_j = index[1] % tile_shape_[1];

    auto &&tile = tiles_[tile_i * grid_shape_[1] + tile_j];

    return tile[local_i * tile_shape_[1] + local_j];
  }

  const_scalar_reference operator[](key_type index) const {
    std::size_t tile_i = index[0] / tile_shape_[0];
    std::size_t tile_j = index[1] / tile_shape_[1];

    std::size_t local_i = index[0] % tile_shape_[0];
    std::size_t local_j = index[1] % tile_shape_[1];

    auto &&tile = tiles_[tile_i * grid_shape_[1] + tile_j];

    return tile[local_i * tile_shape_[1] + local_j];
  }

  iterator begin() {
    return iterator(tiles_, key_type({0, 0}), key_type({0, 0}), grid_shape_,
                    tile_shape_, shape_);
  }

  iterator end() { return begin() + shape()[0] * shape()[1]; }

  key_type tile_shape() const noexcept { return tile_shape_; }

  key_type grid_shape() const noexcept { return grid_shape_; }

  auto tile(key_type tile_index) {
    auto &&[i, j] = tile_index;
    auto iter = tiles_[i * grid_shape()[1] + j].begin();

    std::size_t tm =
        std::min(tile_shape()[0], shape()[0] - i * tile_shape()[0]);
    std::size_t tn =
        std::min(tile_shape()[1], shape()[1] - j * tile_shape()[1]);

    return dense_matrix_view<T, rng::iterator_t<dr::shp::device_vector<
                                    T, dr::shp::device_allocator<T>>>>(
        iter, key_type{tm, tn}, tile_shape()[1],
        tiles_[i * grid_shape()[1] + j].rank());
  }

  std::vector<dense_matrix_view<T, rng::iterator_t<dr::shp::device_vector<
                                       T, dr::shp::device_allocator<T>>>>>
  tiles() {
    std::vector<dense_matrix_view<T, rng::iterator_t<dr::shp::device_vector<
                                         T, dr::shp::device_allocator<T>>>>>
        views_;

    for (std::size_t i = 0; i < grid_shape_[0]; i++) {
      for (std::size_t j = 0; j < grid_shape_[1]; j++) {
        auto iter = tiles_[i * grid_shape_[1] + j].begin();

        std::size_t tm =
            std::min(tile_shape_[0], shape()[0] - i * tile_shape_[0]);
        std::size_t tn =
            std::min(tile_shape_[1], shape()[1] - j * tile_shape_[1]);

        views_.emplace_back(iter, key_type{tm, tn}, tile_shape_[1],
                            tiles_[i * grid_shape_[1] + j].rank());
      }
    }
    return views_;
  }

  template <typename Allocator = std::allocator<T>>
  auto get_tile(key_type tile_index, const Allocator &alloc = Allocator{}) {
    std::size_t nrows = get_tile_shape_(tile_index)[0];
    std::size_t ld = tile_shape_[1];
    std::size_t tile_size = nrows * ld;
    dense_matrix<T, Allocator> local_tile(get_tile_shape_(tile_index), ld,
                                          alloc);
    auto remote_tile = tile(tile_index);
    shp::copy(remote_tile.data(), remote_tile.data() + tile_size,
              local_tile.data());
    return local_tile;
  }

  template <typename Allocator = std::allocator<T>>
  auto get_tile_async(key_type tile_index,
                      const Allocator &alloc = Allocator{}) {
    std::size_t nrows = get_tile_shape_(tile_index)[0];
    std::size_t ld = tile_shape_[1];
    std::size_t tile_size = nrows * ld;
    dense_matrix<T, Allocator> local_tile(get_tile_shape_(tile_index), ld,
                                          alloc);
    auto remote_tile = tile(tile_index);
    auto event = shp::copy_async(
        remote_tile.data(), remote_tile.data() + tile_size, local_tile.data());
    return future(std::move(local_tile), {event});
  }

  auto segments() {
    std::vector<dense_matrix_view<T, rng::iterator_t<dr::shp::device_vector<
                                         T, dr::shp::device_allocator<T>>>>>
        views_;

    for (std::size_t i = 0; i < grid_shape_[0]; i++) {
      for (std::size_t j = 0; j < grid_shape_[1]; j++) {
        auto iter = tiles_[i * grid_shape_[1] + j].begin();

        std::size_t tm =
            std::min(tile_shape_[0], shape()[0] - i * tile_shape_[0]);
        std::size_t tn =
            std::min(tile_shape_[1], shape()[1] - j * tile_shape_[1]);

        std::size_t m_offset = i * tile_shape_[0];
        std::size_t n_offset = j * tile_shape_[1];

        views_.emplace_back(iter, key_type{tm, tn},
                            key_type{m_offset, n_offset}, tile_shape_[1],
                            tiles_[i * grid_shape_[1] + j].rank());
      }
    }
    return dr::__detail::owning_view(std::move(views_));
  }

private:
  void init_() {
    grid_shape_ = partition_->grid_shape(shape());
    tile_shape_ = partition_->tile_shape(shape());

    tiles_.reserve(grid_shape_[0] * grid_shape_[1]);

    for (std::size_t i = 0; i < grid_shape_[0]; i++) {
      for (std::size_t j = 0; j < grid_shape_[1]; j++) {
        std::size_t rank = partition_->tile_rank(shape(), {i, j});

        auto device = dr::shp::devices()[rank];
        dr::shp::device_allocator<T> alloc(dr::shp::context(), device);

        std::size_t tile_size = tile_shape_[0] * tile_shape_[1];

        tiles_.emplace_back(tile_size, alloc, rank);
      }
    }
  }

  key_type get_tile_shape_(key_type tile_index) {
    auto &&[i, j] = tile_index;
    std::size_t tm = std::min(tile_shape_[0], shape()[0] - i * tile_shape_[0]);
    std::size_t tn = std::min(tile_shape_[1], shape()[1] - j * tile_shape_[1]);
    return key_type{tm, tn};
  }

private:
  key_type shape_;
  key_type grid_shape_;
  key_type tile_shape_;
  std::unique_ptr<dr::shp::matrix_partition> partition_;

  std::vector<dr::shp::device_vector<T, dr::shp::device_allocator<T>>> tiles_;
};

} // namespace dr::shp
