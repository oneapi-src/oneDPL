// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <any>
#include <concepts>
#include <limits>
#include <tuple>

namespace dr {

namespace {
template <typename T, std::size_t I, typename U = std::any>
concept TupleElementGettable = requires(T tuple) {
  { std::get<I>(tuple) } -> std::convertible_to<U>;
};
} // namespace

template <typename T, typename... Args>
concept TupleLike =
    requires {
      typename std::tuple_size<std::remove_cvref_t<T>>::type;
      requires std::same_as<
          std::remove_cvref_t<
              decltype(std::tuple_size_v<std::remove_cvref_t<T>>)>,
          std::size_t>;
    } && sizeof...(Args) == std::tuple_size_v<std::remove_cvref_t<T>> &&
    []<std::size_t... I>(std::index_sequence<I...>) {
      return (TupleElementGettable<T, I, Args> && ...);
    }(std::make_index_sequence<std::tuple_size_v<std::remove_cvref_t<T>>>());

template <std::integral T = std::size_t> class index {
public:
  using index_type = T;

  using first_type = T;
  using second_type = T;

  constexpr index_type operator[](index_type dim) const noexcept {
    if (dim == 0) {
      return first;
    } else {
      return second;
    }
  }

  template <std::integral U>
    requires(std::numeric_limits<U>::max() >= std::numeric_limits<T>::max())
  constexpr operator index<U>() const noexcept {
    return index<U>(first, second);
  }

  template <std::integral U>
    requires(std::numeric_limits<U>::max() < std::numeric_limits<T>::max())
  constexpr explicit operator index<U>() const noexcept {
    return index<U>(first, second);
  }

  constexpr index(index_type first, index_type second)
      : first(first), second(second) {}

  template <TupleLike<T, T> Tuple>
  constexpr index(Tuple tuple)
      : first(std::get<0>(tuple)), second(std::get<1>(tuple)) {}

  template <std::integral U> constexpr index(std::initializer_list<U> tuple) {
    assert(tuple.size() == 2);
    first = *tuple.begin();
    second = *(tuple.begin() + 1);
  }

  constexpr bool operator==(const index &) const noexcept = default;

  template <std::size_t Index>
  constexpr T get() const noexcept
    requires(Index <= 1)
  {
    if constexpr (Index == 0) {
      return first;
    }
    if constexpr (Index == 1) {
      return second;
    }
  }

  index() = default;
  ~index() = default;
  index(const index &) = default;
  index &operator=(const index &) = default;
  index(index &&) = default;
  index &operator=(index &&) = default;

  index_type first;
  index_type second;
};

} // namespace dr

namespace std {

template <std::size_t Index, std::integral I>
struct tuple_element<Index, dr::index<I>>
    : tuple_element<Index, std::tuple<I, I>> {};

template <std::integral I>
struct tuple_size<dr::index<I>> : integral_constant<std::size_t, 2> {};

template <std::size_t Index, std::integral I>
inline constexpr I get(dr::index<I> index)
  requires(Index <= 1)
{
  if constexpr (Index == 0) {
    return index.first;
  }
  if constexpr (Index == 1) {
    return index.second;
  }
}

} // namespace std
