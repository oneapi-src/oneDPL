// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cmath>

namespace dr::shp {

namespace detail {

// Factor n into 2 roughly equal factors
// n = pq, p >= q
inline std::tuple<std::size_t, std::size_t> factor(std::size_t n) {
  std::size_t q = std::sqrt(n);

  while (q > 1 && n / q != static_cast<double>(n) / q) {
    q--;
  }
  std::size_t p = n / q;

  return {p, q};
}

} // namespace detail

} // namespace dr::shp
