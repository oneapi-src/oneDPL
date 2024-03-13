// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <dr/shp/allocators.hpp>
#include <dr/shp/device_vector.hpp>

namespace dr::shp {

template <typename T, typename Allocator = dr::shp::device_allocator<T>>
class duplicated_vector {
public:
  using segment_type = dr::shp::device_vector<T, Allocator>;

  using value_type = T;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;

  using allocator_type = Allocator;

  duplicated_vector(std::size_t count = 0) {
    size_ = count;
    capacity_ = count;

    std::size_t rank = 0;
    for (auto &&device : dr::shp::devices()) {
      segments_.emplace_back(
          segment_type(size(), Allocator(dr::shp::context(), device), rank++));
    }
  }

  size_type size() const noexcept { return size_; }

  segment_type &local_vector(std::size_t rank) { return segments_[rank]; }

  const segment_type &local_vector(std::size_t rank) const {
    return segments_[rank];
  }

private:
  std::vector<segment_type> segments_;
  std::size_t capacity_ = 0;
  std::size_t size_ = 0;
};

} // namespace dr::shp
