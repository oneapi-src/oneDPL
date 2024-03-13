// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <vector>

#include <dr/concepts/concepts.hpp>
#include <dr/detail/iterator_adaptor.hpp>
#include <dr/detail/ranges.hpp>
#include <dr/detail/segments_tools.hpp>
#include <dr/shp/device_span.hpp>

namespace dr::shp {

template <typename T, typename L> class distributed_span_accessor {
public:
  using element_type = T;
  using value_type = std::remove_cv_t<T>;

  using segment_type = L;

  using size_type = rng::range_size_t<segment_type>;
  using difference_type = rng::range_difference_t<segment_type>;

  // using pointer = typename segment_type::pointer;
  using reference = rng::range_reference_t<segment_type>;

  using iterator_category = std::random_access_iterator_tag;

  using iterator_accessor = distributed_span_accessor;
  using const_iterator_accessor = iterator_accessor;
  using nonconst_iterator_accessor = iterator_accessor;

  constexpr distributed_span_accessor() noexcept = default;
  constexpr ~distributed_span_accessor() noexcept = default;
  constexpr distributed_span_accessor(
      const distributed_span_accessor &) noexcept = default;
  constexpr distributed_span_accessor &
  operator=(const distributed_span_accessor &) noexcept = default;

  constexpr distributed_span_accessor(std::span<segment_type> segments,
                                      size_type segment_id,
                                      size_type idx) noexcept
      : segments_(segments), segment_id_(segment_id), idx_(idx) {}

  constexpr distributed_span_accessor &
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

  std::span<segment_type> segments_;
  size_type segment_id_ = 0;
  size_type idx_ = 0;
};

template <typename T, typename L>
using distributed_span_iterator =
    dr::iterator_adaptor<distributed_span_accessor<T, L>>;

template <typename T, typename L>
class distributed_span : public rng::view_interface<distributed_span<T, L>> {
public:
  using element_type = T;
  using value_type = std::remove_cv_t<T>;

  using segment_type = dr::shp::device_span<T, L>;

  using size_type = rng::range_size_t<segment_type>;
  using difference_type = rng::range_difference_t<segment_type>;

  // using pointer = typename segment_type::pointer;
  using reference = rng::range_reference_t<segment_type>;

  // Note: creating the "global view" will be trivial once #44178 is resolved.
  // (https://github.com/llvm/llvm-project/issues/44178)
  // The "global view" is simply all of the segmented views joined together.
  // However, this code does not currently compile due to a bug in Clang,
  // so I am currently implementing my own global iterator manually.
  // using joined_view_type =
  // rng::join_view<rng::ref_view<std::vector<segment_type>>>;
  // using iterator = rng::iterator_t<joined_view_type>;

  using iterator = distributed_span_iterator<T, segment_type>;

  constexpr distributed_span() noexcept = default;
  constexpr distributed_span(const distributed_span &) noexcept = default;
  constexpr distributed_span &
  operator=(const distributed_span &) noexcept = default;

  template <rng::input_range R>
    requires(dr::remote_range<rng::range_reference_t<R>>)
  constexpr distributed_span(R &&segments) {
    for (auto &&segment : segments) {
      std::size_t size = rng::size(segment);
      segments_.push_back(
          segment_type(rng::begin(segment), size, dr::ranges::rank(segment)));
      size_ += size;
    }
  }

  template <dr::distributed_range R> constexpr distributed_span(R &&r) {
    for (auto &&segment : dr::ranges::segments(std::forward<R>(r))) {
      std::size_t size = rng::size(segment);
      segments_.push_back(
          segment_type(rng::begin(segment), size, dr::ranges::rank(segment)));
      size_ += size;
    }
  }

  constexpr size_type size() const noexcept { return size_; }

  constexpr size_type size_bytes() const noexcept {
    return size() * sizeof(element_type);
  }

  constexpr reference operator[](size_type idx) const {
    // TODO: optimize this
    std::size_t span_id = 0;
    for (std::size_t span_id = 0; idx >= segments()[span_id].size();
         span_id++) {
      idx -= segments()[span_id].size();
    }
    return segments()[span_id][idx];
  }

  [[nodiscard]] constexpr bool empty() const noexcept { return size() == 0; }

  constexpr distributed_span
  subspan(size_type Offset, size_type Count = std::dynamic_extent) const {
    Count = std::min(Count, size() - Offset);

    std::vector<segment_type> new_segments;

    // Forward to segment_id that contains global index `Offset`.
    std::size_t segment_id = 0;
    for (segment_id = 0; Offset >= segments()[segment_id].size();
         segment_id++) {
      Offset -= segments()[segment_id].size();
    }

    // Our Offset begins at `segment_id, Offset`

    while (Count > 0) {
      std::size_t local_count =
          std::min(Count, segments()[segment_id].size() - Offset);
      auto new_segment = segments()[segment_id].subspan(Offset, local_count);
      new_segments.push_back(new_segment);
      Count -= local_count;
      Offset = 0;
      segment_id++;
    }

    return distributed_span(new_segments);
  }

  constexpr distributed_span first(size_type Count) const {
    return subspan(0, Count);
  }

  constexpr distributed_span last(size_type Count) const {
    return subspan(size() - Count, Count);
  }

  iterator begin() { return iterator(segments(), 0, 0); }

  iterator end() { return iterator(segments(), segments().size(), 0); }

  constexpr reference front() { return segments().front().front(); }

  constexpr reference back() { return segments().back().back(); }

  std::span<segment_type> segments() { return segments_; }

  std::span<const segment_type> segments() const { return segments_; }

private:
  std::size_t size_ = 0;
  std::vector<segment_type> segments_;
};

template <rng::input_range R>
distributed_span(R &&segments)
    -> distributed_span<rng::range_value_t<rng::range_value_t<R>>,
                        rng::iterator_t<rng::range_value_t<R>>>;

template <dr::distributed_contiguous_range R>
distributed_span(R &&r) -> distributed_span<
    rng::range_value_t<R>,
    rng::iterator_t<rng::range_value_t<decltype(dr::ranges::segments(r))>>>;

} // namespace dr::shp
