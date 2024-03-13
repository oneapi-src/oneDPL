// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

namespace dr::views {

//
// range-v3 iota uses sentinels that are not the same type as the
// iterator. A zip that uses an iota has the same issue. Make our own.
//

struct iota_fn_ {
  template <std::integral W> auto operator()(W value) const {
    return rng::views::iota(value, std::numeric_limits<W>::max());
  }

  template <std::integral W, std::integral Bound>
  auto operator()(W value, Bound bound) const {
    return rng::views::iota(value, W(bound));
  }
};

inline constexpr auto iota = iota_fn_{};

} // namespace dr::views
