// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <dr/concepts/concepts.hpp>
#include <dr/shp/distributed_span.hpp>

namespace dr::shp {

template <int dimensions = 1> class id {
public:
  static_assert(dimensions == 1);

  id() noexcept = default;

  id(std::size_t segment_id, std::size_t local_id, std::size_t global_id)
      : segment_id_(segment_id), local_id_(local_id), global_id_(global_id) {}

  std::size_t get(int dimension) const { return global_id_; }

  operator std::size_t() const { return global_id_; }

  std::size_t segment() const { return segment_id_; }

  std::size_t local_id() const { return local_id_; }

private:
  std::size_t segment_id_ = 0;
  std::size_t local_id_ = 0;
  std::size_t global_id_ = 0;
};

class segment_range_accessor {
public:
  using element_type = id<1>;
  using value_type = element_type;

  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;

  // using pointer = typename segment_type::pointer;
  using reference = value_type;

  using iterator_category = std::random_access_iterator_tag;

  using iterator_accessor = segment_range_accessor;
  using const_iterator_accessor = iterator_accessor;
  using nonconst_iterator_accessor = iterator_accessor;

  constexpr segment_range_accessor() noexcept = default;
  constexpr ~segment_range_accessor() noexcept = default;
  constexpr segment_range_accessor(const segment_range_accessor &) noexcept =
      default;
  constexpr segment_range_accessor &
  operator=(const segment_range_accessor &) noexcept = default;

  constexpr segment_range_accessor(size_type segment_id, size_type idx,
                                   size_type global_offset) noexcept
      : global_offset_(global_offset), segment_id_(segment_id), idx_(idx) {}

  constexpr segment_range_accessor &
  operator+=(difference_type offset) noexcept {
    idx_ += offset;
    return *this;
  }

  constexpr bool operator==(const iterator_accessor &other) const noexcept {
    return segment_id_ == other.segment_id_ && idx_ == other.idx_;
  }

  constexpr difference_type
  operator-(const iterator_accessor &other) const noexcept {
    return difference_type(idx_) - difference_type(other.idx_);
  }

  // Comparing iterators from different segments is undefined
  constexpr bool operator<(const iterator_accessor &other) const noexcept {
    return idx_ < other.idx_;
  }

  reference operator*() const noexcept {
    return value_type(segment_id_, idx_, get_global_idx());
  }

private:
  size_type get_global_idx() const noexcept { return global_offset_ + idx_; }

  size_type global_offset_ = 0;
  size_type segment_id_ = 0;
  size_type idx_ = 0;
};

using segment_range_iterator = dr::iterator_adaptor<segment_range_accessor>;

template <int dimensions = 1> class segment_range {
public:
  static_assert(dimensions == 1);

  using value_type = id<dimensions>;
  using size_type = std::size_t;
  using different_type = std::ptrdiff_t;

  using reference = value_type;

  using iterator = segment_range_iterator;

  segment_range(std::size_t segment_id, std::size_t segment_size,
                std::size_t global_offset)
      : segment_id_(segment_id), segment_size_(segment_size),
        global_offset_(global_offset) {}

  iterator begin() const { return iterator(segment_id_, 0, global_offset_); }

  iterator end() const {
    return iterator(segment_id_, segment_size_, global_offset_);
  }

  size_type size() const noexcept { return segment_size_; }

  value_type operator[](std::size_t idx) { return *(begin() + idx); }

  size_type rank() const noexcept { return 0; }

private:
  std::size_t segment_size_;
  std::size_t segment_id_;
  std::size_t global_offset_;
};

/*
template <rng::forward_range R> auto distributed_iota_view(R &&r) {
  static_assert(dr::distributed_contiguous_range<R>);
  if constexpr (dr::distributed_contiguous_range<R>) {
    std::vector<segment_range<>> iota_segments;
    std::size_t global_offset = 0;
    std::size_t segment_id = 0;
    for (auto &&segment : r.segments()) {
      iota_segments.push_back(
          segment_range(segment_id, segment.size(), global_offset));
      global_offset += segment.size();
      segment_id++;
    }
    return dr::shp::distributed_span(iota_segments);
  } else {
    return segment_range(0, rng::size(r), 0);
  }
}
*/

} // namespace dr::shp
