// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This file incorporates work covered by the following copyright and permission
// notice:
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#pragma once

constexpr std::size_t EVENLY_DIVIDABLE_SIZE =
    2 * 3 * 5 * 7 * 11 * 13; // good up to 16 processes

template <dr::distributed_range DR>
using LocalVec = std::vector<typename DR::value_type>;

struct AOS_Struct {
  bool operator==(const AOS_Struct &other) const {
    return first == other.first && second == other.second;
  }

  int first, second;
};

#ifndef DRISHMEM
struct OpsAOS {

  using dist_vec_type = xp::distributed_vector<AOS_Struct>;
  using vec_type = std::vector<AOS_Struct>;

  OpsAOS(std::size_t n) : dist_vec(n), vec(n) {
    for (std::size_t i = 0; i < n; i++) {
      AOS_Struct s{100 + int(i), 200 + int(i)};
      dist_vec[i] = s;
      vec[i] = s;
    }
    fence();
  }

  dist_vec_type dist_vec;
  vec_type vec;
};

inline std::ostream &operator<<(std::ostream &os, const AOS_Struct &st) {
  os << "[ " << st.first << " " << st.second << " ]";
  return os;
}
#endif

template <typename T> struct Ops1 {
  Ops1(std::size_t n) : dist_vec(n), vec(n) {
    iota(dist_vec, 100);
    stdrng::iota(vec, 100);
  }

  T dist_vec;
  LocalVec<T> vec;
};

template <typename T> struct Ops2 {
  Ops2(std::size_t n) : dist_vec0(n), dist_vec1(n), vec0(n), vec1(n) {
    iota(dist_vec0, 100);
    iota(dist_vec1, 200);
    stdrng::iota(vec0, 100);
    stdrng::iota(vec1, 200);
  }

  T dist_vec0, dist_vec1;
  LocalVec<T> vec0, vec1;
};

template <typename T> struct Ops3 {
  Ops3(std::size_t n)
      : dist_vec0(n), dist_vec1(n), dist_vec2(n), vec0(n), vec1(n), vec2(n) {
    iota(dist_vec0, 100);
    iota(dist_vec1, 200);
    iota(dist_vec2, 300);
    stdrng::iota(vec0, 100);
    stdrng::iota(vec1, 200);
    stdrng::iota(vec2, 300);
  }

  T dist_vec0, dist_vec1, dist_vec2;
  LocalVec<T> vec0, vec1, vec2;
};

template <std::floating_point T>
bool fp_equal(T a, T b, T epsilon = 128 * std::numeric_limits<T>::epsilon()) {
  if (a == b) {
    return true;
  }

  auto abs_th = std::numeric_limits<T>::min();

  auto diff = std::abs(a - b);

  auto norm =
      std::min((std::abs(a) + std::abs(b)), std::numeric_limits<float>::max());
  return diff < std::max(abs_th, epsilon * norm);
}

template <std::floating_point T>
bool fp_equal(std::vector<T> a, std::vector<T> b,
              T epsilon = 128 * std::numeric_limits<T>::epsilon()) {
  if (a.size() != b.size()) {
    return false;
  }

  for (std::size_t i = 0; i < a.size(); i++) {
    if (!fp_equal(a[i], b[i])) {
      return false;
    }
  }

  return true;
}

template <class A, class B, class C, class D>
bool operator==(std::pair<A, B> const &x, std::tuple<C, D> const &y) {
  return x.first == std::get<0>(y) && x.second == std::get<1>(y);
}

template <class A, class B, class C, class D>
bool operator==(std::tuple<C, D> const &y, std::pair<A, B> const &x) {
  return x == y;
}

template <class A, class B, class C, class D>
bool operator==(std::pair<C, D> const &y, std::pair<A, B> const &x) {
  return x.first == y.first && x.second == y.second;
}

template <stdrng::range R1, stdrng::range R2> bool is_equal(R1 &&r1, R2 &&r2) {
  if (stdrng::distance(stdrng::begin(r1), stdrng::end(r1)) !=
      stdrng::distance(stdrng::begin(r2), stdrng::end(r2))) {
    return false;
  }

  // TODO: why r2.begin() is not working here?
  auto r1i = r1.begin();
  for (const auto &v2 : r2) {
    if (*r1i++ != v2) {
      return false;
    }
  }

  return true;
}

bool is_equal(std::forward_iterator auto it, stdrng::range auto &&r) {
  for (auto e : r) {
    if (*it++ != e) {
      return false;
    }
  }
  return true;
}

std::string equal_message(stdrng::range auto &&ref, stdrng::range auto &&actual,
                   std::string title = " ") {
  if (is_equal(ref, actual)) {
    return "";
  }
  return drfmt::format("\n{}"
                     "    ref:    {}\n"
                     "    actual: {}\n  ",
                     title == "" ? "" : "    " + title + "\n",
                     stdrng::views::all(ref), stdrng::views::all(actual));
}

std::string unary_check_message(stdrng::range auto &&in, stdrng::range auto &&ref,
                                stdrng::range auto &&tst, std::string title = "") {
  if (is_equal(ref, tst)) {
    return "";
  } else {
    return drfmt::format("\n{}"
                         "    in:     {}\n"
                         "    ref:    {}\n"
                         "    actual: {}\n  ",
                         title == "" ? "" : "    " + title + "\n", in, ref,
                         tst);
  }
}

bool contains_empty(auto &&r) {
  if (stdrng::distance(r) == 1) {
    return false;
  }

  for (auto &&x : r) {
    if (stdrng::empty(x)) {
      return true;
    }
  }

  return false;
}

std::string check_segments_message(auto &&r) {
  auto segments = dr::ranges::segments(r);
  auto flat = stdrng::views::join(segments);
  if (contains_empty(segments) || !is_equal(r, flat)) {
    return drfmt::format("\n"
                       "    Segment error\n"
                       "      range:    {}\n"
                       "      segments: {}\n  ",
                       stdrng::views::all(r), stdrng::views::all(segments));
  }
  return "";
}

auto check_view_message(stdrng::range auto &&ref, stdrng::range auto &&actual) {
  return check_segments_message(actual) +
         equal_message(ref, actual, "view mismatch");
}

auto check_mutate_view_message(auto &ops, stdrng::range auto &&ref,
                               stdrng::range auto &&actual) {
  // Check view
  auto message = check_view_message(ref, actual);

  barrier();

  // Mutate view
  auto negate = [](auto &&val) { val = -val; };
  auto input_vector = ops.vec;
  std::vector input_view(ref.begin(), ref.end());
  xp::for_each(actual, negate);
  stdrng::for_each(ref, negate);

  // Check mutated view
  message +=
      unary_check_message(input_view, actual, ref, "mutated view mismatch");

  // Check underlying dv
  message += unary_check_message(input_vector, ops.vec, ops.dist_vec,
                                 "mutated distributed range mismatch");

  return message;
}

auto gtest_result(const auto &message) {
  if (message == "") {
    return testing::AssertionSuccess();
  } else {
    return testing::AssertionFailure() << message;
  }
}

auto gtest_equal(stdrng::range auto &&ref, stdrng::range auto &&actual,
                 std::string title = " ") {
  return gtest_result(equal_message(ref, actual, title));
}

template <stdrng::range Rng>
auto gtest_equal(std::initializer_list<stdrng::range_value_t<Rng>> ref, Rng &&actual,
                 std::string title = " ") {
  return gtest_result(
    equal_message(std::vector<stdrng::range_value_t<Rng>>(ref), actual, title));
}

auto check_unary_op(stdrng::range auto &&in, stdrng::range auto &&ref,
                    stdrng::range auto &&tst, std::string title = "") {
  return gtest_result(unary_check_message(in, ref, tst, title));
}

auto check_binary_check_op(stdrng::range auto &&a, stdrng::range auto &&b,
                           stdrng::range auto &&ref, stdrng::range auto &&actual) {
  if (is_equal(ref, actual)) {
    return testing::AssertionSuccess();
  } else {
    return testing::AssertionFailure() << drfmt::format(
               "\n        a: {}\n        b: {}\n      ref: {}\n    "
               "actual: {}\n  ",
               a, b, ref, actual);
  }
}

auto check_segments(std::forward_iterator auto di) {
  auto segments = dr::ranges::segments(di);
  auto flat = stdrng::join_view(segments);
  if (contains_empty(segments) || !is_equal(di, flat)) {
    return testing::AssertionFailure()
           << drfmt::format("\n    segments: {}\n  ", segments);
  } else {
    return testing::AssertionSuccess();
  }
}

auto check_segments(stdrng::forward_range auto &&dr) {
  return gtest_result(check_segments_message(dr));
}

auto check_view(stdrng::range auto &&ref, stdrng::range auto &&actual) {
  return gtest_result(check_view_message(ref, actual));
}

auto check_mutate_view(auto &op, stdrng::range auto &&ref,
                       stdrng::range auto &&actual) {
  return gtest_result(check_mutate_view_message(op, ref, actual));
}

template <typename T>
std::vector<T> generate_random(std::size_t n, std::size_t bound = 25) {
  std::vector<T> v;
  v.reserve(n);

  for (std::size_t i = 0; i < n; i++) {
    auto r = lrand48() % bound;
    v.push_back(r);
  }

  return v;
}

template <typename T>
concept streamable = requires(std::ostream &os, T value) {
  { os << value } -> std::convertible_to<std::ostream &>;
};

namespace oneapi::dpl::experimental::dr::sp {

// gtest relies on ADL to find the printer
template <typename T>
std::ostream &operator<<(std::ostream &os, const distributed_vector<T> &dist) {
  os << "{ ";
  bool first = true;
  for (const auto &val : dist) {
    if (first) {
      first = false;
    } else {
      os << ", ";
    }
    if constexpr (streamable<T>) {
      os << val;
    } else {
      os << "Unstreamable";
    }
  }
  os << " }";
  return os;
}

template <typename T>
bool operator==(const distributed_vector<T> &dist_vec,
                const std::vector<T> &local_vec) {
  return is_equal(dist_vec, local_vec);
}

} // namespace oneapi::dpl::experimental::dr::sp

namespace DR_RANGES_NAMESPACE {

template <stdrng::range R1, stdrng::range R2> bool operator==(R1 &&r1, R2 &&r2) {
  return is_equal(std::forward<R1>(r1), std::forward<R2>(r2));
}

} // namespace DR_RANGES_NAMESPACE
