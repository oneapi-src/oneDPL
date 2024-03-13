// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <dr/detail/index.hpp>
#include <dr/shp/algorithms/copy.hpp>
#include <dr/shp/containers/matrix_entry.hpp>
#include <dr/shp/containers/matrix_partition.hpp>
#include <dr/shp/device_vector.hpp>
#include <dr/shp/distributed_span.hpp>
#include <dr/shp/init.hpp>
#include <dr/shp/util/generate_random.hpp>
#include <dr/shp/views/csr_matrix_view.hpp>
#include <iterator>

namespace dr::shp {

template <rng::random_access_range Segments>
  requires(rng::viewable_range<Segments>)
class distributed_range_accessor {
public:
  using segment_type = rng::range_value_t<Segments>;

  using value_type = rng::range_value_t<segment_type>;

  using size_type = rng::range_size_t<segment_type>;
  using difference_type = rng::range_difference_t<segment_type>;

  using reference = rng::range_reference_t<segment_type>;

  using iterator_category = std::random_access_iterator_tag;

  using iterator_accessor = distributed_range_accessor;
  using const_iterator_accessor = iterator_accessor;
  using nonconst_iterator_accessor = iterator_accessor;

  constexpr distributed_range_accessor() noexcept = default;
  constexpr ~distributed_range_accessor() noexcept = default;
  constexpr distributed_range_accessor(
      const distributed_range_accessor &) noexcept = default;
  constexpr distributed_range_accessor &
  operator=(const distributed_range_accessor &) noexcept = default;

  constexpr distributed_range_accessor(Segments segments, size_type segment_id,
                                       size_type idx) noexcept
      : segments_(rng::views::all(std::forward<Segments>(segments))),
        segment_id_(segment_id), idx_(idx) {}

  constexpr distributed_range_accessor &
  operator+=(difference_type offset) noexcept {

    while (offset > 0) {
      difference_type current_offset = std::min(
          offset,
          difference_type(rng::size(*(segments_.begin() + segment_id_))) -
              difference_type(idx_));
      idx_ += current_offset;
      offset -= current_offset;

      if (idx_ >= rng::size((*(segments_.begin() + segment_id_)))) {
        segment_id_++;
        idx_ = 0;
      }
    }

    while (offset < 0) {
      difference_type current_offset =
          std::min(-offset, difference_type(idx_) + 1);

      difference_type new_idx = difference_type(idx_) - current_offset;

      if (new_idx < 0) {
        segment_id_--;
        new_idx = rng::size(*(segments_.begin() + segment_id_)) - 1;
      }

      idx_ = new_idx;
    }

    return *this;
  }

  constexpr bool operator==(const iterator_accessor &other) const noexcept {
    return segment_id_ == other.segment_id_ && idx_ == other.idx_;
  }

  constexpr difference_type
  operator-(const iterator_accessor &other) const noexcept {
    return difference_type(get_global_idx()) - other.get_global_idx();
  }

  constexpr bool operator<(const iterator_accessor &other) const noexcept {
    if (segment_id_ < other.segment_id_) {
      return true;
    } else if (segment_id_ == other.segment_id_) {
      return idx_ < other.idx_;
    } else {
      return false;
    }
  }

  constexpr reference operator*() const noexcept {
    return *((*(segments_.begin() + segment_id_)).begin() + idx_);
  }

private:
  size_type get_global_idx() const noexcept {
    size_type cumulative_size = 0;
    for (std::size_t i = 0; i < segment_id_; i++) {
      cumulative_size += segments_[i].size();
    }
    return cumulative_size + idx_;
  }

  rng::views::all_t<Segments> segments_;
  size_type segment_id_ = 0;
  size_type idx_ = 0;
};

template <typename Segments>
using distributed_sparse_matrix_iterator =
    dr::iterator_adaptor<distributed_range_accessor<Segments>>;

template <typename T, std::integral I = std::int64_t> class sparse_matrix {
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

  using key_type = dr::index<I>;

  using segment_type = dr::shp::csr_matrix_view<
      T, I,
      rng::iterator_t<dr::shp::device_vector<T, dr::shp::device_allocator<T>>>,
      rng::iterator_t<dr::shp::device_vector<I, dr::shp::device_allocator<I>>>>;

  // using iterator = sparse_matrix_iterator<T, dr::shp::device_vector<T,
  // dr::shp::device_allocator<T>>>;
  using iterator =
      distributed_sparse_matrix_iterator<std::span<segment_type> &&>;

  sparse_matrix(key_type shape)
      : shape_(shape), partition_(new dr::shp::block_cyclic()) {
    init_();
  }

  sparse_matrix(key_type shape, double density)
      : shape_(shape), partition_(new dr::shp::block_cyclic()) {
    init_random_(density);
  }

  sparse_matrix(key_type shape, double density,
                const matrix_partition &partition)
      : shape_(shape), partition_(partition.clone()) {
    init_random_(density);
  }

  sparse_matrix(key_type shape, const matrix_partition &partition)
      : shape_(shape), partition_(partition.clone()) {
    init_();
  }

  size_type size() const noexcept { return total_nnz_; }

  key_type shape() const noexcept { return shape_; }

  iterator begin() { return iterator(segments(), 0, 0); }

  iterator end() {
    return iterator(segments(), grid_shape_[0] * grid_shape_[1], 0);
  }

  segment_type tile(key_type tile_index) {
    std::size_t tile_idx = tile_index[0] * grid_shape_[1] + tile_index[1];
    auto values = values_[tile_idx].begin();
    auto rowptr = rowptr_[tile_idx].begin();
    auto colind = colind_[tile_idx].begin();
    auto nnz = nnz_[tile_idx];

    std::size_t tm =
        std::min(tile_shape_[0], shape()[0] - tile_index[0] * tile_shape_[0]);
    std::size_t tn =
        std::min(tile_shape_[1], shape()[1] - tile_index[1] * tile_shape_[1]);

    return segment_type(values, rowptr, colind, key_type(tm, tn), nnz,
                        values_[tile_idx].rank());
  }

  // Note: this function is currently *not* asynchronous due to a deadlock
  // in `gemv_benchmark`.  I believe this is a SYCL bug.
  template <typename... Args>
  auto copy_tile_async(key_type tile_index,
                       csr_matrix_view<T, I, Args...> tile_view) {
    std::size_t tile_idx = tile_index[0] * grid_shape_[1] + tile_index[1];
    auto &&values = values_[tile_idx];
    auto &&colind = colind_[tile_idx];
    auto &&rowptr = rowptr_[tile_idx];
    auto &&nnz = nnz_[tile_idx];

    total_nnz_ -= nnz;
    nnz = tile_view.size();

    total_nnz_ += tile_view.size();

    values.resize(tile_view.size());
    colind.resize(tile_view.size());
    rowptr.resize(tile_view.shape()[0] + 1);

    auto v_e = dr::shp::copy_async(tile_view.values_data(),
                                   tile_view.values_data() + values.size(),
                                   values.data());

    auto c_e = dr::shp::copy_async(tile_view.colind_data(),
                                   tile_view.colind_data() + colind.size(),
                                   colind.data());

    auto r_e = dr::shp::copy_async(tile_view.rowptr_data(),
                                   tile_view.rowptr_data() + rowptr.size(),
                                   rowptr.data());

    tiles_ = generate_tiles_();
    segments_ = generate_segments_();

    v_e.wait();
    c_e.wait();
    r_e.wait();

    return __detail::combine_events({v_e, c_e, r_e});
  }

  template <typename... Args>
  void copy_tile(key_type tile_index,
                 csr_matrix_view<T, I, Args...> tile_view) {
    copy_tile_async(tile_index, tile_view).wait();
  }

  key_type tile_shape() const noexcept { return tile_shape_; }

  key_type grid_shape() const noexcept { return grid_shape_; }

  std::span<segment_type> tiles() { return std::span(tiles_); }

  std::span<segment_type> segments() { return std::span(segments_); }

private:
  std::vector<segment_type> generate_tiles_() {
    std::vector<segment_type> views_;

    for (std::size_t i = 0; i < grid_shape_[0]; i++) {
      for (std::size_t j = 0; j < grid_shape_[1]; j++) {
        std::size_t tm = std::min<std::size_t>(tile_shape_[0],
                                               shape()[0] - i * tile_shape_[0]);
        std::size_t tn = std::min<std::size_t>(tile_shape_[1],
                                               shape()[1] - j * tile_shape_[1]);

        std::size_t tile_idx = i * grid_shape_[1] + j;

        auto values = values_[tile_idx].begin();
        auto rowptr = rowptr_[tile_idx].begin();
        auto colind = colind_[tile_idx].begin();
        auto nnz = nnz_[tile_idx];

        views_.emplace_back(values, rowptr, colind, key_type(tm, tn), nnz,
                            values_[tile_idx].rank());
      }
    }
    return views_;
  }

  std::vector<segment_type> generate_segments_() {
    std::vector<segment_type> views_;

    for (std::size_t i = 0; i < grid_shape_[0]; i++) {
      for (std::size_t j = 0; j < grid_shape_[1]; j++) {
        std::size_t tm = std::min<std::size_t>(tile_shape_[0],
                                               shape()[0] - i * tile_shape_[0]);
        std::size_t tn = std::min<std::size_t>(tile_shape_[1],
                                               shape()[1] - j * tile_shape_[1]);

        std::size_t tile_idx = i * grid_shape_[1] + j;

        auto values = values_[tile_idx].begin();
        auto rowptr = rowptr_[tile_idx].begin();
        auto colind = colind_[tile_idx].begin();
        auto nnz = nnz_[tile_idx];

        std::size_t m_offset = i * tile_shape_[0];
        std::size_t n_offset = j * tile_shape_[1];

        views_.emplace_back(values, rowptr, colind, key_type(tm, tn), nnz,
                            values_[i * grid_shape_[1] + j].rank(),
                            key_type(m_offset, n_offset));
      }
    }
    return views_;
  }

private:
  void init_() {
    grid_shape_ = key_type(partition_->grid_shape(shape()));
    tile_shape_ = key_type(partition_->tile_shape(shape()));

    values_.reserve(grid_shape_[0] * grid_shape_[1]);
    rowptr_.reserve(grid_shape_[0] * grid_shape_[1]);
    colind_.reserve(grid_shape_[0] * grid_shape_[1]);
    nnz_.reserve(grid_shape_[0] * grid_shape_[1]);

    for (std::size_t i = 0; i < grid_shape_[0]; i++) {
      for (std::size_t j = 0; j < grid_shape_[1]; j++) {
        std::size_t rank = partition_->tile_rank(shape(), {i, j});

        auto device = dr::shp::devices()[rank];
        dr::shp::device_allocator<T> alloc(dr::shp::context(), device);
        dr::shp::device_allocator<I> i_alloc(dr::shp::context(), device);

        values_.emplace_back(1, alloc, rank);
        rowptr_.emplace_back(2, i_alloc, rank);
        colind_.emplace_back(1, i_alloc, rank);
        nnz_.push_back(0);
        rowptr_.back()[0] = 0;
        rowptr_.back()[1] = 0;
      }
    }
    tiles_ = generate_tiles_();
    segments_ = generate_segments_();
  }

  void init_random_(double density) {
    grid_shape_ = key_type(partition_->grid_shape(shape()));
    tile_shape_ = key_type(partition_->tile_shape(shape()));

    values_.reserve(grid_shape_[0] * grid_shape_[1]);
    rowptr_.reserve(grid_shape_[0] * grid_shape_[1]);
    colind_.reserve(grid_shape_[0] * grid_shape_[1]);
    nnz_.reserve(grid_shape_[0] * grid_shape_[1]);

    for (std::size_t i = 0; i < grid_shape_[0]; i++) {
      for (std::size_t j = 0; j < grid_shape_[1]; j++) {
        std::size_t rank = partition_->tile_rank(shape(), {i, j});

        std::size_t tm = std::min<std::size_t>(tile_shape_[0],
                                               shape()[0] - i * tile_shape_[0]);
        std::size_t tn = std::min<std::size_t>(tile_shape_[1],
                                               shape()[1] - j * tile_shape_[1]);

        auto device = dr::shp::devices()[rank];
        dr::shp::device_allocator<T> alloc(dr::shp::context(), device);
        dr::shp::device_allocator<I> i_alloc(dr::shp::context(), device);

        auto seed = i * grid_shape_[1] + j;

        auto csr = generate_random_csr<T, I>(key_type(tm, tn), density, seed);
        std::size_t nnz = csr.size();

        dr::shp::device_vector<T, dr::shp::device_allocator<T>> values(
            csr.size(), alloc, rank);
        dr::shp::device_vector<I, dr::shp::device_allocator<I>> rowptr(
            tm + 1, i_alloc, rank);

        dr::shp::device_vector<I, dr::shp::device_allocator<I>> colind(
            csr.size(), i_alloc, rank);

        dr::shp::copy(csr.values_data(), csr.values_data() + csr.size(),
                      values.data());
        dr::shp::copy(csr.rowptr_data(), csr.rowptr_data() + tm + 1,
                      rowptr.data());
        dr::shp::copy(csr.colind_data(), csr.colind_data() + csr.size(),
                      colind.data());

        values_.push_back(std::move(values));
        rowptr_.emplace_back(std::move(rowptr));
        colind_.emplace_back(std::move(colind));
        nnz_.push_back(nnz);
        total_nnz_ += nnz;

        delete[] csr.values_data();
        delete[] csr.rowptr_data();
        delete[] csr.colind_data();
      }
    }
    tiles_ = generate_tiles_();
    segments_ = generate_segments_();
  }

private:
  key_type shape_;
  key_type grid_shape_;
  key_type tile_shape_;
  std::unique_ptr<dr::shp::matrix_partition> partition_;

  std::vector<dr::shp::device_vector<T, dr::shp::device_allocator<T>>> values_;
  std::vector<dr::shp::device_vector<I, dr::shp::device_allocator<I>>> rowptr_;
  std::vector<dr::shp::device_vector<I, dr::shp::device_allocator<I>>> colind_;

  std::vector<std::size_t> nnz_;
  std::size_t total_nnz_ = 0;

  std::vector<segment_type> tiles_;
  std::vector<segment_type> segments_;
};

} // namespace dr::shp
