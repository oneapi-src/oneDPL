// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cmath>

<<<<<<< HEAD
namespace experimental::dr::shp {
=======
namespace experimental::shp {
>>>>>>> cd565891f4ffdd0b4641810a38c60c683e5f1fe0

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

<<<<<<< HEAD
} // namespace experimental::dr::shp
=======
} // namespace experimental::shp
>>>>>>> cd565891f4ffdd0b4641810a38c60c683e5f1fe0
