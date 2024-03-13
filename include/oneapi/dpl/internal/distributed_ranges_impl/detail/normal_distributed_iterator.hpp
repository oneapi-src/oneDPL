// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <dr/detail/segments_tools.hpp>

namespace dr {

template <rng::viewable_range V>
/*
requires(dr::remote_range<rng::range_reference_t<V>> &&
       rng::random_access_range<rng::range_reference_t<V>>)
       */
class normal_distributed_iterator_accessor {
public:
  using value_type = rng::range_value_t<rng::range_reference_t<V>>;

  using segment_type = rng::range_value_t<V>;

  using size_type = rng::range_size_t<segment_type>;
  using difference_type = rng::range_difference_t<segment_type>;

  using reference = rng::range_reference_t<segment_type>;

  using iterator_category = std::random_access_iterator_tag;

  using iterator_accessor = normal_distributed_iterator_accessor;
  using const_iterator_accessor = iterator_accessor;
  using nonconst_iterator_accessor = iterator_accessor;

  constexpr normal_distributed_iterator_accessor() noexcept = default;
  constexpr ~normal_distributed_iterator_accessor() noexcept = default;
  constexpr normal_distributed_iterator_accessor(
      const normal_distributed_iterator_accessor &) noexcept = default;
  constexpr normal_distributed_iterator_accessor &
  operator=(const normal_distributed_iterator_accessor &) noexcept = default;

  constexpr normal_distributed_iterator_accessor(V segments,
                                                 size_type segment_id,
                                                 size_type idx) noexcept
      : segments_(segments), segment_id_(segment_id), idx_(idx) {}

  constexpr normal_distributed_iterator_accessor &
  operator+=(difference_type offset) noexcept {

    while (offset > 0) {
      difference_type current_offset =
          std::min(offset, difference_type(segments_[segment_id_].size()) -
                               difference_type(idx_));
      idx_ += current_offset;
      offset -= current_offset;

      if (idx_ >= segments_[segment_id_].size()) {
        segment_id_++;
        idx_ = 0;
      }
    }

    while (offset < 0) {
      difference_type current_offset =
          std::min(-offset, difference_type(idx_) + 1);

      difference_type new_idx = difference_type(idx_) - current_offset;
      offset += current_offset;

      if (new_idx < 0) {
        segment_id_--;
        new_idx = segments_[segment_id_].size() - 1;
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
    return segments_[segment_id_][idx_];
  }

  auto segments() const noexcept {
    return dr::__detail::drop_segments(segments_, segment_id_, idx_);
  }

private:
  size_type get_global_idx() const noexcept {
    size_type cumulative_size = 0;
    for (std::size_t i = 0; i < segment_id_; i++) {
      cumulative_size += segments_[i].size();
    }
    return cumulative_size + idx_;
  }

  rng::views::all_t<V> segments_;
  size_type segment_id_ = 0;
  size_type idx_ = 0;
};

template <rng::viewable_range T>
using normal_distributed_iterator =
    dr::iterator_adaptor<normal_distributed_iterator_accessor<T>>;

} // namespace dr
