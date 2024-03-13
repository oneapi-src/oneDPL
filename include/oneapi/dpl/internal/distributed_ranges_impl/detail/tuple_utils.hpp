// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

namespace dr::__detail {

auto tuple_transform(auto tuple, auto op) {
  auto transform = [op](auto &&...items) {
    return std::make_tuple(op(items)...);
  };
  return std::apply(transform, tuple);
}

auto tie_transform(auto tuple, auto op) {
  auto transform = [op]<typename... Items>(Items &&...items) {
    return std::tie(op(std::forward<Items>(items))...);
  };
  return std::apply(transform, tuple);
}

auto tuple_foreach(auto tuple, auto op) {
  auto transform = [op](auto... items) { (op(items), ...); };
  std::apply(transform, tuple);
}

} // namespace dr::__detail
