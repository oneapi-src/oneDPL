// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <limits>

#include <oneapi/dpl/internal/distributed_ranges_impl/concepts/concepts.hpp>
#include <oneapi/dpl/internal/distributed_ranges_impl/detail/ranges_shim.hpp>
#include <oneapi/dpl/internal/distributed_ranges_impl/shp/algorithms/for_each.hpp>
#include <oneapi/dpl/internal/distributed_ranges_impl/views/iota.hpp>

<<<<<<< HEAD
namespace experimental::dr::shp {
=======
namespace experimental::shp {
>>>>>>> cd565891f4ffdd0b4641810a38c60c683e5f1fe0

template <experimental::dr::distributed_range R, std::integral T> void iota(R &&r, T value) {
  auto iota_view = rng::views::iota(value, T(value + rng::distance(r)));

  for_each(par_unseq, views::zip(iota_view, r), [](auto &&elem) {
    auto &&[idx, v] = elem;
    v = idx;
  });
}

template <experimental::dr::distributed_iterator Iter, std::integral T>
void iota(Iter begin, Iter end, T value) {
  auto r = rng::subrange(begin, end);
  iota(r, value);
}

<<<<<<< HEAD
} // namespace experimental::dr::shp
=======
} // namespace experimental::shp
>>>>>>> cd565891f4ffdd0b4641810a38c60c683e5f1fe0
