// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <oneapi/dpl/internal/distributed_ranges_impl/detail/index.hpp>
#include <oneapi/dpl/internal/distributed_ranges_impl/detail/segments_tools.hpp>
#include <oneapi/dpl/internal/distributed_ranges_impl/shp/distributed_span.hpp>
#include <oneapi/dpl/internal/distributed_ranges_impl/shp/views/enumerate.hpp>
#include <oneapi/dpl/internal/distributed_ranges_impl/shp/zip_view.hpp>
#include <oneapi/dpl/internal/distributed_ranges_impl/views/transform.hpp>

<<<<<<< HEAD
namespace experimental::dr::shp {
=======
namespace experimental::shp {
>>>>>>> cd565891f4ffdd0b4641810a38c60c683e5f1fe0

namespace views {

template <experimental::dr::distributed_range R>
auto slice(R &&r, experimental::dr::index<> slice_indices) {
  return experimental::dr::shp::distributed_span(experimental::dr::ranges::segments(std::forward<R>(r)))
      .subspan(slice_indices[0], slice_indices[1] - slice_indices[0]);
}

class slice_adaptor_closure {
public:
  slice_adaptor_closure(experimental::dr::index<> slice_indices) : idx_(slice_indices) {}

  template <rng::random_access_range R> auto operator()(R &&r) const {
    return slice(std::forward<R>(r), idx_);
  }

  template <rng::random_access_range R>
  friend auto operator|(R &&r, const slice_adaptor_closure &closure) {
    return closure(std::forward<R>(r));
  }

private:
  experimental::dr::index<> idx_;
};

inline auto slice(experimental::dr::index<> slice_indices) {
  return slice_adaptor_closure(slice_indices);
}

} // namespace views

<<<<<<< HEAD
} // namespace experimental::dr::shp
=======
} // namespace experimental::shp
>>>>>>> cd565891f4ffdd0b4641810a38c60c683e5f1fe0
