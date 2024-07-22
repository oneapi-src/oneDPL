// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <oneapi/dpl/internal/distributed_ranges_impl/shp/algorithms/fill.hpp>
#include <oneapi/dpl/internal/distributed_ranges_impl/shp/algorithms/reduce.hpp>
#include <oneapi/dpl/internal/distributed_ranges_impl/shp/detail.hpp>
#include <oneapi/dpl/internal/distributed_ranges_impl/shp/init.hpp>
#include <oneapi/dpl/internal/distributed_ranges_impl/shp/util.hpp>
#include <oneapi/dpl/internal/distributed_ranges_impl/shp/views/views.hpp>
#include <oneapi/dpl/internal/distributed_ranges_impl/shp/zip_view.hpp>

namespace oneapi::dpl::experimental::dr::shp
{

template <typename ExecutionPolicy, dr::distributed_range R1, dr::distributed_range R2>
requires std::equality_comparable_with<stdrng::range_value_t<R1>, stdrng::range_value_t<R2>>
bool
equal(ExecutionPolicy&& policy, R1&& r1, R2&& r2)
{

    if (stdrng::distance(r1) != stdrng::distance(r2))
    {
        return false;
    }

    // we must use ints instead of bools, because distributed ranges do not
    // support bools
    auto compare = [](auto&& elems) { return elems.first == elems.second ? 1 : 0; };

    auto zipped_views = views::zip(r1, r2);
    auto compared = shp::views::transform(zipped_views, compare);
    auto min = [](double x, double y) { return std::min(x, y); };
    auto result = shp::reduce(policy, compared, 1, min);
    return result == 1;
}

template <dr::distributed_range R1, dr::distributed_range R2>
bool
equal(R1&& r1, R2&& r2)
{
    return equal(dr::shp::par_unseq, r1, r2);
}
} // namespace oneapi::dpl::experimental::dr::shp
