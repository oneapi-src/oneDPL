// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

<<<<<<< HEAD
namespace experimental::dr::__detail {
=======
namespace experimental::__detail {
>>>>>>> cd565891f4ffdd0b4641810a38c60c683e5f1fe0

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

<<<<<<< HEAD
} // namespace experimental::dr::__detail
=======
} // namespace experimental::__detail
>>>>>>> cd565891f4ffdd0b4641810a38c60c683e5f1fe0
