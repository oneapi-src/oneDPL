// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <concepts>
#include <iterator>
#include <type_traits>

#include <dr/concepts/concepts.hpp>
#include <dr/detail/ranges_shim.hpp>

namespace dr {

template <std::random_access_iterator Iter, std::copy_constructible F>
class transform_iterator {
public:
  using value_type = std::invoke_result_t<F, std::iter_value_t<Iter>>;
  using difference_type = std::iter_difference_t<Iter>;
  using iterator = transform_iterator<Iter, F>;
  using reference = value_type;

  using pointer = iterator;

  using iterator_category = std::random_access_iterator_tag;

  transform_iterator(Iter iter, F fn) noexcept : iter_(iter) {}
  transform_iterator() noexcept = default;
  ~transform_iterator() noexcept = default;
  transform_iterator(const transform_iterator &) noexcept = default;
  transform_iterator &operator=(const transform_iterator &) noexcept = default;

  bool operator==(const transform_iterator &other) const noexcept {
    return iter_ == other.iter_;
  }

  bool operator!=(const transform_iterator &other) const noexcept {
    return iter_ != other.iter_;
  }

  iterator operator+(difference_type offset) const noexcept {
    return iterator(iter_ + offset, fn_);
  }

  iterator operator-(difference_type offset) const noexcept {
    return iterator(iter_ - offset, fn_);
  }

  difference_type operator-(iterator other) const noexcept {
    return iter_ - other.iter_;
  }

  bool operator<(iterator other) const noexcept { return iter_ < other.iter_; }

  bool operator>(iterator other) const noexcept { return iter_ > iter_; }

  bool operator<=(iterator other) const noexcept {
    return iter_ <= other.iter_;
  }

  bool operator>=(iterator other) const noexcept {
    return iter_ >= other.iter_;
  }

  iterator &operator++() noexcept {
    ++iter_;
    return *this;
  }

  iterator operator++(int) noexcept {
    iterator other = *this;
    ++(*this);
    return other;
  }

  iterator &operator--() noexcept {
    --iter_;
    return *this;
  }

  iterator operator--(int) noexcept {
    iterator other = *this;
    --(*this);
    return other;
  }

  iterator &operator+=(difference_type offset) noexcept {
    iter_ += offset;
    return *this;
  }

  iterator &operator-=(difference_type offset) noexcept {
    iter_ -= offset;
    return *this;
  }

  reference operator*() const noexcept { return fn_(*iter_); }

  reference operator[](difference_type offset) const noexcept {
    return *(*this + offset);
  }

  friend iterator operator+(difference_type n, iterator iter) {
    return iter.iter_ + n;
  }

  auto local() const
    requires(dr::ranges::__detail::has_local<Iter>)
  {
    auto iter = dr::ranges::__detail::local(iter_);
    return transform_iterator<decltype(iter), F>(iter, fn_);
  }

private:
  Iter iter_;
  F fn_;
};

template <rng::random_access_range V, std::copy_constructible F>
class transform_view : public rng::view_interface<transform_view<V, F>> {
public:
  template <rng::viewable_range R>
  transform_view(R &&r, F fn)
      : base_(rng::views::all(std::forward<R>(r))), fn_(fn) {}

  auto begin() const { return transform_iterator(rng::begin(base_), fn_); }

  auto end() const { return transform_iterator(rng::end(base_), fn_); }

  auto size() const
    requires(rng::sized_range<V>)
  {
    return rng::size(base_);
  }

  auto segments() const
    requires(dr::distributed_range<V>)
  {
    auto fn = fn_;
    return dr::ranges::segments(base_) |
           rng::views::transform([fn]<typename T>(T &&segment) {
             return transform_view<rng::views::all_t<decltype(segment)>, F>(
                 std::forward<T>(segment), fn);
           });
  }

  auto rank() const
    requires(dr::remote_range<V>)
  {
    return dr::ranges::rank(base_);
  }

  V base() const { return base_; }

private:
  V base_;
  F fn_;
};

template <rng::viewable_range R, std::copy_constructible F>
transform_view(R &&r, F fn) -> transform_view<rng::views::all_t<R>, F>;

namespace views {

template <std::copy_constructible F> class transform_adapter_closure {
public:
  transform_adapter_closure(F fn) : fn_(fn) {}

  template <rng::viewable_range R> auto operator()(R &&r) const {
    return dr::transform_view(std::forward<R>(r), fn_);
  }

  template <rng::viewable_range R>
  friend auto operator|(R &&r, const transform_adapter_closure &closure) {
    return closure(std::forward<R>(r));
  }

private:
  F fn_;
};

class transform_fn_ {
public:
  template <rng::viewable_range R, std::copy_constructible F>
  auto operator()(R &&r, F &&f) const {
    return transform_adapter_closure(std::forward<F>(f))(std::forward<R>(r));
  }

  template <std::copy_constructible F> auto operator()(F &&fn) const {
    return transform_adapter_closure(std::forward<F>(fn));
  }
};

inline constexpr auto transform = transform_fn_{};
} // namespace views

} // namespace dr

#if !defined(DR_SPEC)

// Needed to satisfy rng::viewable_range
template <rng::random_access_range V, std::copy_constructible F>
inline constexpr bool rng::enable_borrowed_range<dr::transform_view<V, F>> =
    true;

#endif
