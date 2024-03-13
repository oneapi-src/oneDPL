// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

namespace dr::__detail {

inline std::size_t round_up(std::size_t n, std::size_t multiple) {
  if (multiple == 0) {
    return n;
  }

  int remainder = n % multiple;
  if (remainder == 0) {
    return n;
  }

  return n + multiple - remainder;
}

inline std::size_t partition_up(std::size_t n, std::size_t multiple) {
  if (multiple == 0) {
    return n;
  }

  return round_up(n, multiple) / multiple;
}

} // namespace dr::__detail
