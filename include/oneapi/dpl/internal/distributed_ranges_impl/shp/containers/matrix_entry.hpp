// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <concepts>
#include <limits>
#include <type_traits>

#include <dr/detail/index.hpp>

namespace dr::shp {

template <typename T, typename I = std::size_t> class matrix_entry {
public:
  using index_type = I;
  using map_type = T;

  matrix_entry(dr::index<I> index, const map_type &value)
      : index_(index), value_(value) {}
  matrix_entry(dr::index<I> index, map_type &&value)
      : index_(index), value_(std::move(value)) {}

  template <typename U>
    requires(std::is_constructible_v<T, U>)
  matrix_entry(dr::index<I> index, U &&value)
      : index_(index), value_(std::forward<U>(value)) {}

  template <typename Entry>
  matrix_entry(Entry &&entry)
      : index_(std::get<0>(entry)), value_(std::get<1>(entry)) {}

  template <std::size_t Index> auto get() const noexcept {
    if constexpr (Index == 0) {
      return index();
    }
    if constexpr (Index == 1) {
      return value();
    }
  }

  operator std::pair<std::pair<I, I>, T>() const noexcept {
    return {{index_[0], index_[1]}, value_};
  }

  dr::index<I> index() const noexcept { return index_; }

  map_type value() const noexcept { return value_; }

  template <std::integral U>
    requires(!std::is_same_v<I, U> &&
             std::numeric_limits<U>::max() >= std::numeric_limits<I>::max())
  operator matrix_entry<T, U>() const noexcept {
    return matrix_entry<T, U>(index_, value_);
  }

  template <std::integral U>
    requires(!std::is_const_v<T> && !std::is_same_v<I, U> &&
             std::numeric_limits<U>::max() >= std::numeric_limits<I>::max())
  operator matrix_entry<std::add_const_t<T>, U>() const noexcept {
    return matrix_entry<std::add_const_t<T>, U>(index_, value_);
  }

  bool operator<(const matrix_entry &other) const noexcept {
    if (index()[0] < other.index()[0]) {
      return true;
    } else if (index()[0] == other.index()[0] &&
               index()[1] < other.index()[1]) {
      return true;
    }
    return false;
  }

  matrix_entry() = default;
  ~matrix_entry() = default;

  matrix_entry(const matrix_entry &) = default;
  matrix_entry(matrix_entry &&) = default;
  matrix_entry &operator=(const matrix_entry &) = default;
  matrix_entry &operator=(matrix_entry &&) = default;

private:
  dr::index<I> index_;
  map_type value_;
};

} // namespace dr::shp

namespace std {

template <typename T, typename I>
  requires(!std::is_const_v<T>)
void swap(dr::shp::matrix_entry<T, I> a, dr::shp::matrix_entry<T, I> b) {
  dr::shp::matrix_entry<T, I> other = a;
  a = b;
  b = other;
}

template <std::size_t Index, typename T, typename I>
struct tuple_element<Index, dr::shp::matrix_entry<T, I>>
    : tuple_element<Index, std::tuple<dr::index<I>, T>> {};

template <typename T, typename I>
struct tuple_size<dr::shp::matrix_entry<T, I>> : integral_constant<size_t, 2> {
};

} // namespace std

namespace dr::shp {

template <typename T, typename I = std::size_t, typename TRef = T &>
class matrix_ref {
public:
  using scalar_type = T;
  using index_type = I;

  using key_type = dr::index<I>;
  using map_type = T;

  using scalar_reference = TRef;

  using value_type = dr::shp::matrix_entry<T, I>;

  matrix_ref(dr::index<I> index, scalar_reference value)
      : index_(index), value_(value) {}

  operator value_type() const noexcept { return value_type(index_, value_); }

  operator std::pair<std::pair<I, I>, T>() const noexcept {
    return {{index_[0], index_[1]}, value_};
  }

  template <std::size_t Index>
  decltype(auto) get() const noexcept
    requires(Index <= 1)
  {
    if constexpr (Index == 0) {
      return index();
    }
    if constexpr (Index == 1) {
      return value();
    }
  }

  dr::index<I> index() const noexcept { return index_; }

  scalar_reference value() const noexcept { return value_; }

  template <std::integral U>
    requires(!std::is_same_v<I, U> &&
             std::numeric_limits<U>::max() >= std::numeric_limits<I>::max())
  operator matrix_ref<T, U, TRef>() const noexcept {
    return matrix_ref<T, U, TRef>(index_, value_);
  }

  template <std::integral U>
    requires(!std::is_const_v<T> && !std::is_same_v<I, U> &&
             std::numeric_limits<U>::max() >= std::numeric_limits<I>::max())
  operator matrix_ref<std::add_const_t<T>, U, TRef>() const noexcept {
    return matrix_ref<std::add_const_t<T>, U, TRef>(index_, value_);
  }

  bool operator<(matrix_entry<T, I> other) const noexcept {
    if (index()[0] < other.index()[0]) {
      return true;
    } else if (index()[0] == other.index()[0] &&
               index()[1] < other.index()[1]) {
      return true;
    }
    return false;
  }

  matrix_ref() = delete;
  ~matrix_ref() = default;

  matrix_ref(const matrix_ref &) = default;
  matrix_ref &operator=(const matrix_ref &) = delete;
  matrix_ref(matrix_ref &&) = default;
  matrix_ref &operator=(matrix_ref &&) = default;

private:
  dr::index<I> index_;
  scalar_reference value_;
};

} // namespace dr::shp

namespace std {

template <typename T, typename I, typename TRef>
  requires(!std::is_const_v<T>)
void swap(dr::shp::matrix_ref<T, I, TRef> a,
          dr::shp::matrix_ref<T, I, TRef> b) {
  dr::shp::matrix_entry<T, I> other = a;
  a = b;
  b = other;
}

template <std::size_t Index, typename T, typename I, typename TRef>
struct tuple_element<Index, dr::shp::matrix_ref<T, I, TRef>>
    : tuple_element<Index, std::tuple<dr::index<I>, TRef>> {};

template <typename T, typename I, typename TRef>
struct tuple_size<dr::shp::matrix_ref<T, I, TRef>>
    : integral_constant<std::size_t, 2> {};

template <std::size_t Index, typename T, typename I, typename TRef>
inline decltype(auto) get(dr::shp::matrix_ref<T, I, TRef> ref)
  requires(Index <= 1)
{
  if constexpr (Index == 0) {
    return ref.index();
  }
  if constexpr (Index == 1) {
    return ref.value();
  }
}

template <std::size_t Index, typename T, typename I, typename TRef>
inline decltype(auto) get(dr::shp::matrix_entry<T, I> entry)
  requires(Index <= 1)
{
  if constexpr (Index == 0) {
    return entry.index();
  }
  if constexpr (Index == 1) {
    return entry.value();
  }
}

} // namespace std
