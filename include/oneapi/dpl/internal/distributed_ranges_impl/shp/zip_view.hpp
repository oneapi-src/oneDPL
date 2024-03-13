// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <oneapi/dpl/iterator>

#include <dr/detail/iterator_adaptor.hpp>
#include <dr/detail/owning_view.hpp>
#include <dr/detail/ranges_shim.hpp>
#include <dr/detail/view_detectors.hpp>
#include <dr/shp/device_span.hpp>

namespace dr {

template <typename T> struct is_owning_view : std::false_type {};
// template <rng::range R>
// struct is_owning_view<rng::owning_view<R>> : std::true_type {};

template <typename T>
inline constexpr bool is_owning_view_v = is_owning_view<T>{};

}; // namespace dr

namespace dr::shp {

namespace __detail {

template <typename... Args> struct tuple_or_pair {
  using type = std::tuple<Args...>;
};

template <typename T, typename U> struct tuple_or_pair<T, U> {
  using type = std::pair<T, U>;
};

template <typename... Args>
using tuple_or_pair_t = typename tuple_or_pair<Args...>::type;

}; // namespace __detail

template <rng::random_access_iterator... Iters> class zip_accessor {
public:
  using element_type = __detail::tuple_or_pair_t<std::iter_value_t<Iters>...>;
  using value_type = element_type;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using reference = __detail::tuple_or_pair_t<std::iter_reference_t<Iters>...>;

  using iterator_category = std::random_access_iterator_tag;

  using iterator_accessor = zip_accessor;
  using const_iterator_accessor = iterator_accessor;
  using nonconst_iterator_accessor = iterator_accessor;

  constexpr zip_accessor() noexcept = default;
  constexpr ~zip_accessor() noexcept = default;
  constexpr zip_accessor(const zip_accessor &) noexcept = default;
  constexpr zip_accessor &operator=(const zip_accessor &) noexcept = default;

  constexpr zip_accessor(Iters... iters) : iterators_(iters...) {}

  zip_accessor &operator+=(difference_type offset) {
    auto increment = [&](auto &&iter) { iter += offset; };
    iterators_apply_impl_<0>(increment);
    return *this;
  }

  constexpr bool operator==(const iterator_accessor &other) const noexcept {
    return std::get<0>(iterators_) == std::get<0>(other.iterators_);
  }

  constexpr difference_type
  operator-(const iterator_accessor &other) const noexcept {
    return std::get<0>(iterators_) - std::get<0>(other.iterators_);
  }

  constexpr bool operator<(const iterator_accessor &other) const noexcept {
    return std::get<0>(iterators_) < std::get<0>(other.iterators_);
  }

  constexpr reference operator*() const noexcept {
    return get_impl_(std::make_index_sequence<sizeof...(Iters)>{});
  }

private:
  template <std::size_t... Ints>
  reference get_impl_(std::index_sequence<Ints...>) const noexcept {
    return reference(*std::get<Ints>(iterators_)...);
  }

  template <std::size_t I, typename Fn> void iterators_apply_impl_(Fn &&fn) {
    fn(std::get<I>(iterators_));
    if constexpr (I + 1 < sizeof...(Iters)) {
      iterators_apply_impl_<I + 1>(fn);
    }
  }

  std::tuple<Iters...> iterators_;
};

template <rng::random_access_iterator... Iters>
using zip_iterator = dr::iterator_adaptor<zip_accessor<Iters...>>;

/// zip
template <rng::random_access_range... Rs>
class zip_view : public rng::view_interface<zip_view<Rs...>> {
public:
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;

  zip_view(Rs... rs) : views_(rng::views::all(std::forward<Rs>(rs))...) {
    std::array<std::size_t, sizeof...(Rs)> sizes = {
        std::size_t(rng::distance(rs))...};

    // TODO: support zipped views with some ranges shorter than others
    size_ = sizes[0];

    for (auto &&size : sizes) {
      size_ = std::min(size_, size);
    }
  }

  std::size_t size() const noexcept { return size_; }

  auto begin() const {
    return begin_impl_(std::make_index_sequence<sizeof...(Rs)>{});
  }

  auto end() const { return begin() + size(); }

  auto operator[](std::size_t idx) const { return *(begin() + idx); }

  static constexpr bool num_views = sizeof...(Rs);

  template <std::size_t I> decltype(auto) get_view() const {
    auto &&view = std::get<I>(views_);

    if constexpr (dr::is_ref_view_v<std::remove_cvref_t<decltype(view)>> ||
                  dr::is_owning_view_v<std::remove_cvref_t<decltype(view)>>) {
      return view.base();
    } else {
      return view;
    }
  }

  // If there is at least one distributed range, expose segments
  // of overlapping remote ranges.
  auto segments() const
    requires(dr::distributed_range<Rs> || ...)
  {
    std::array<std::size_t, sizeof...(Rs)> segment_ids;
    std::array<std::size_t, sizeof...(Rs)> local_idx;
    segment_ids.fill(0);
    local_idx.fill(0);

    std::size_t cumulative_size = 0;

    using segment_view_type = decltype(get_zipped_view_impl_(
        segment_ids, local_idx, 0, std::make_index_sequence<sizeof...(Rs)>{}));
    std::vector<segment_view_type> segment_views;

    while (cumulative_size < size()) {
      auto size = get_next_segment_size(segment_ids, local_idx);

      cumulative_size += size;

      // Create zipped segment with
      // zip_view(segments()[Is].subspan(local_idx[Is], size)...) And some rank
      // (e.g. get_view<0>.rank())
      auto segment_view =
          get_zipped_view_impl_(segment_ids, local_idx, size,
                                std::make_index_sequence<sizeof...(Rs)>{});

      segment_views.push_back(std::move(segment_view));

      increment_local_idx(segment_ids, local_idx, size);
    }

    return dr::__detail::owning_view(std::move(segment_views));
  }

  // Return a range corresponding to each segment in `segments()`,
  // but with a tuple of the constituent ranges instead of a
  // `zip_view` of the ranges.
  auto zipped_segments() const
    requires(dr::distributed_range<Rs> || ...)
  {
    std::array<std::size_t, sizeof...(Rs)> segment_ids;
    std::array<std::size_t, sizeof...(Rs)> local_idx;
    segment_ids.fill(0);
    local_idx.fill(0);

    std::size_t cumulative_size = 0;

    using segment_view_type = decltype(get_zipped_segments_impl_(
        segment_ids, local_idx, 0, std::make_index_sequence<sizeof...(Rs)>{}));
    std::vector<segment_view_type> segment_views;

    while (cumulative_size < size()) {
      auto size = get_next_segment_size(segment_ids, local_idx);

      cumulative_size += size;

      // Get zipped segments with
      // std::tuple(segments()[Is].subspan(local_idx[Is], size)...)
      auto segment_view =
          get_zipped_segments_impl_(segment_ids, local_idx, size,
                                    std::make_index_sequence<sizeof...(Rs)>{});

      segment_views.push_back(std::move(segment_view));

      increment_local_idx(segment_ids, local_idx, size);
    }

    return dr::__detail::owning_view(std::move(segment_views));
  }

  auto local() const noexcept
    requires(!(dr::distributed_range<Rs> || ...))
  {
    return local_impl_(std::make_index_sequence<sizeof...(Rs)>());
  }

  // If:
  //   - There is at least one remote range in the zip
  //   - There are no distributed ranges in the zip
  // Expose a rank.
  std::size_t rank() const
    requires((dr::remote_range<Rs> || ...) &&
             !(dr::distributed_range<Rs> || ...))
  {
    return get_rank_impl_<0, Rs...>();
  }

private:
  template <std::size_t... Ints>
  auto local_impl_(std::index_sequence<Ints...>) const noexcept {
    return rng::views::zip(__detail::local(std::get<Ints>(views_))...);
  }

  template <std::size_t I, typename R> std::size_t get_rank_impl_() const {
    static_assert(I < sizeof...(Rs));
    return dr::ranges::rank(get_view<I>());
  }

  template <std::size_t I, typename R, typename... Rs_>
    requires(sizeof...(Rs_) > 0)
  std::size_t get_rank_impl_() const {
    static_assert(I < sizeof...(Rs));
    if constexpr (dr::remote_range<R>) {
      return dr::ranges::rank(get_view<I>());
    } else {
      return get_rank_impl_<I + 1, Rs_...>();
    }
  }

  template <typename T> auto create_view_impl_(T &&t) const {
    if constexpr (dr::remote_range<T>) {
      return dr::shp::device_span(std::forward<T>(t));
    } else {
      return dr::shp::span(std::forward<T>(t));
    }
  }

  template <std::size_t... Is>
  auto get_zipped_view_impl_(auto &&segment_ids, auto &&local_idx,
                             std::size_t size,
                             std::index_sequence<Is...>) const {
    return zip_view<decltype(create_view_impl_(
                                 segment_or_orig_(get_view<Is>(),
                                                  segment_ids[Is]))
                                 .subspan(local_idx[Is], size))...>(
        create_view_impl_(segment_or_orig_(get_view<Is>(), segment_ids[Is]))
            .subspan(local_idx[Is], size)...);
  }

  template <std::size_t... Is>
  auto get_zipped_segments_impl_(auto &&segment_ids, auto &&local_idx,
                                 std::size_t size,
                                 std::index_sequence<Is...>) const {
    return std::tuple(
        create_view_impl_(segment_or_orig_(get_view<Is>(), segment_ids[Is]))
            .subspan(local_idx[Is], size)...);
  }

  template <std::size_t I = 0>
  void increment_local_idx(auto &&segment_ids, auto &&local_idx,
                           std::size_t size) const {
    local_idx[I] += size;

    if (local_idx[I] >=
        rng::distance(segment_or_orig_(get_view<I>(), segment_ids[I]))) {
      local_idx[I] = 0;
      segment_ids[I]++;
    }

    if constexpr (I + 1 < sizeof...(Rs)) {
      increment_local_idx<I + 1>(segment_ids, local_idx, size);
    }
  }

  template <std::size_t... Is>
  auto begin_impl_(std::index_sequence<Is...>) const {
    return zip_iterator<rng::iterator_t<Rs>...>(
        rng::begin(std::get<Is>(views_))...);
  }

  template <dr::distributed_range T>
  decltype(auto) segment_or_orig_(T &&t, std::size_t idx) const {
    return dr::ranges::segments(t)[idx];
  }

  template <typename T>
  decltype(auto) segment_or_orig_(T &&t, std::size_t idx) const {
    return t;
  }

  template <std::size_t... Is>
  std::size_t get_next_segment_size_impl_(auto &&segment_ids, auto &&local_idx,
                                          std::index_sequence<Is...>) const {
    return std::min({std::size_t(rng::distance(
                         segment_or_orig_(get_view<Is>(), segment_ids[Is]))) -
                     local_idx[Is]...});
  }

  std::size_t get_next_segment_size(auto &&segment_ids,
                                    auto &&local_idx) const {
    return get_next_segment_size_impl_(
        segment_ids, local_idx, std::make_index_sequence<sizeof...(Rs)>{});
  }

  std::tuple<rng::views::all_t<Rs>...> views_;
  std::size_t size_;
};

template <typename... Rs> zip_view(Rs &&...rs) -> zip_view<Rs...>;

namespace views {

/// Zip
template <rng::random_access_range... Rs> auto zip(Rs &&...rs) {
  return dr::shp::zip_view(std::forward<Rs>(rs)...);
}

} // namespace views

} // namespace dr::shp
