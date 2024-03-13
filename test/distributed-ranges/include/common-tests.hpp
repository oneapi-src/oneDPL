// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause
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

  using dist_vec_type = xhp::distributed_vector<AOS_Struct>;
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
    rng::iota(vec, 100);
  }

  T dist_vec;
  LocalVec<T> vec;
};

template <typename T> struct Ops2 {
  Ops2(std::size_t n) : dist_vec0(n), dist_vec1(n), vec0(n), vec1(n) {
    iota(dist_vec0, 100);
    iota(dist_vec1, 200);
    rng::iota(vec0, 100);
    rng::iota(vec1, 200);
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
    rng::iota(vec0, 100);
    rng::iota(vec1, 200);
    rng::iota(vec2, 300);
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

template <rng::range R1, rng::range R2> bool is_equal(R1 &&r1, R2 &&r2) {
  if (rng::distance(rng::begin(r1), rng::end(r1)) !=
      rng::distance(rng::begin(r2), rng::end(r2))) {
    return false;
  }
  auto r2i = r2.begin();
  for (const auto &v1 : r1) {
    if (v1 != *r2i++) {
      return false;
    }
  }

  return true;
}

bool is_equal(std::forward_iterator auto it, rng::range auto &&r) {
  for (auto e : r) {
    if (*it++ != e) {
      return false;
    }
  }
  return true;
}

auto equal_message(rng::range auto &&ref, rng::range auto &&actual,
                   std::string title = " ") {
  if (is_equal(ref, actual)) {
    return fmt::format("");
  }
  return fmt::format("\n{}"
                     "    ref:    {}\n"
                     "    actual: {}\n  ",
                     title == "" ? "" : "    " + title + "\n",
                     rng::views::all(ref), rng::views::all(actual));
}

std::string unary_check_message(rng::range auto &&in, rng::range auto &&ref,
                                rng::range auto &&tst, std::string title = "") {
  if (is_equal(ref, tst)) {
    return "";
  } else {
    return fmt::format("\n{}"
                       "    in:     {}\n"
                       "    ref:    {}\n"
                       "    actual: {}\n  ",
                       title == "" ? "" : "    " + title + "\n", in, ref, tst);
  }
}

bool contains_empty(auto &&r) {
  if (rng::distance(r) == 1) {
    return false;
  }

  for (auto &&x : r) {
    if (rng::empty(x)) {
      return true;
    }
  }

  return false;
}

std::string check_segments_message(auto &&r) {
  auto segments = dr::ranges::segments(r);
  auto flat = rng::views::join(segments);
  if (contains_empty(segments) || !is_equal(r, flat)) {
    return fmt::format("\n"
                       "    Segment error\n"
                       "      range:    {}\n"
                       "      segments: {}\n  ",
                       rng::views::all(r), rng::views::all(segments));
  }
  return "";
}

auto check_view_message(rng::range auto &&ref, rng::range auto &&actual) {
  return check_segments_message(actual) +
         equal_message(ref, actual, "view mismatch");
}

auto check_mutate_view_message(auto &ops, rng::range auto &&ref,
                               rng::range auto &&actual) {
  // Check view
  auto message = check_view_message(ref, actual);

  barrier();

  // Mutate view
  auto negate = [](auto &&val) { val = -val; };
  auto input_vector = ops.vec;
  std::vector input_view(ref.begin(), ref.end());
  xhp::for_each(actual, negate);
  rng::for_each(ref, negate);

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

auto equal(rng::range auto &&ref, rng::range auto &&actual,
           std::string title = " ") {
  return gtest_result(equal_message(ref, actual, title));
}

template <rng::range Rng>
auto equal(std::initializer_list<rng::range_value_t<Rng>> ref, Rng &&actual,
           std::string title = " ") {
  return gtest_result(
      equal_message(std::vector<rng::range_value_t<Rng>>(ref), actual, title));
}

auto check_unary_op(rng::range auto &&in, rng::range auto &&ref,
                    rng::range auto &&tst, std::string title = "") {
  return gtest_result(unary_check_message(in, ref, tst, title));
}

auto check_binary_check_op(rng::range auto &&a, rng::range auto &&b,
                           rng::range auto &&ref, rng::range auto &&actual) {
  if (is_equal(ref, actual)) {
    return testing::AssertionSuccess();
  } else {
    return testing::AssertionFailure()
           << fmt::format("\n        a: {}\n        b: {}\n      ref: {}\n    "
                          "actual: {}\n  ",
                          a, b, ref, actual);
  }
}

auto check_segments(std::forward_iterator auto di) {
  auto segments = dr::ranges::segments(di);
  auto flat = rng::join_view(segments);
  if (contains_empty(segments) || !is_equal(di, flat)) {
    return testing::AssertionFailure()
           << fmt::format("\n    segments: {}\n  ", segments);
  } else {
    return testing::AssertionSuccess();
  }
}

auto check_segments(rng::forward_range auto &&dr) {
  return gtest_result(check_segments_message(dr));
}

auto check_view(rng::range auto &&ref, rng::range auto &&actual) {
  return gtest_result(check_view_message(ref, actual));
}

auto check_mutate_view(auto &op, rng::range auto &&ref,
                       rng::range auto &&actual) {
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

namespace dr::mhp {

// gtest relies on ADL to find the printer
template <typename T, typename B>
std::ostream &operator<<(std::ostream &os,
                         const xhp::distributed_vector<T, B> &dist) {
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

template <typename T, typename B>
bool operator==(const xhp::distributed_vector<T, B> &dist_vec,
                const std::vector<T> &local_vec) {
  return is_equal(local_vec, dist_vec);
}

} // namespace dr::mhp

namespace dr::shp {

// gtest relies on ADL to find the printer
template <typename T>
std::ostream &operator<<(std::ostream &os,
                         const xhp::distributed_vector<T> &dist) {
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
bool operator==(const xhp::distributed_vector<T> &dist_vec,
                const std::vector<T> &local_vec) {
  return is_equal(dist_vec, local_vec);
}

} // namespace dr::shp

namespace DR_RANGES_NAMESPACE {

template <rng::range R1, rng::range R2> bool operator==(R1 &&r1, R2 &&r2) {
  return is_equal(std::forward<R1>(r1), std::forward<R2>(r2));
}

template <typename... Ts>
inline std::ostream &operator<<(std::ostream &os,
                                const rng::common_tuple<Ts...> &obj) {
  os << fmt::format("{}", obj);
  return os;
}

template <typename T1, typename T2>
inline std::ostream &operator<<(std::ostream &os,
                                const rng::common_pair<T1, T2> &obj) {
  os << fmt::format("{}", obj);
  return os;
}

} // namespace DR_RANGES_NAMESPACE

namespace std {

template <typename... Ts>
inline std::ostream &operator<<(std::ostream &os,
                                const std::tuple<Ts...> &obj) {
  os << fmt::format("{}", obj);
  return os;
}

template <typename T1, typename T2>
inline std::ostream &operator<<(std::ostream &os,
                                const std::pair<T1, T2> &obj) {
  os << fmt::format("{}", obj);
  return os;
}

} // namespace std
