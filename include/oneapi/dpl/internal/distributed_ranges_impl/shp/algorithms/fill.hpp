// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <memory>
#include <type_traits>

#include <sycl/sycl.hpp>

#include <dr/concepts/concepts.hpp>
#include <dr/detail/segments_tools.hpp>
#include <dr/shp/detail.hpp>
#include <dr/shp/device_ptr.hpp>
#include <dr/shp/util.hpp>

namespace dr::shp {

template <std::contiguous_iterator Iter>
  requires(!std::is_const_v<std::iter_value_t<Iter>> &&
           std::is_trivially_copyable_v<std::iter_value_t<Iter>>)
sycl::event fill_async(Iter first, Iter last,
                       const std::iter_value_t<Iter> &value) {
  auto &&q = __detail::get_queue_for_pointer(first);
  std::iter_value_t<Iter> *arr = std::to_address(first);
  // not using q.fill because of CMPLRLLVM-46438
  return dr::__detail::parallel_for(q, sycl::range<>(last - first),
                                    [=](auto idx) { arr[idx] = value; });
}

template <std::contiguous_iterator Iter>
  requires(!std::is_const_v<std::iter_value_t<Iter>>)
void fill(Iter first, Iter last, const std::iter_value_t<Iter> &value) {
  fill_async(first, last, value).wait();
}

template <typename T, typename U>
  requires(std::indirectly_writable<device_ptr<T>, U>)
sycl::event fill_async(device_ptr<T> first, device_ptr<T> last,
                       const U &value) {
  auto &&q = __detail::get_queue_for_pointer(first);
  auto *arr = first.get_raw_pointer();
  // not using q.fill because of CMPLRLLVM-46438
  return dr::__detail::parallel_for(q, sycl::range<>(last - first),
                                    [=](auto idx) { arr[idx] = value; });
}

template <typename T, typename U>
  requires(std::indirectly_writable<device_ptr<T>, U>)
void fill(device_ptr<T> first, device_ptr<T> last, const U &value) {
  fill_async(first, last, value).wait();
}

template <typename T, dr::remote_contiguous_range R>
sycl::event fill_async(R &&r, const T &value) {
  auto &&q = __detail::queue(dr::ranges::rank(r));
  auto *arr = std::to_address(rng::begin(dr::ranges::local(r)));
  // not using q.fill because of CMPLRLLVM-46438
  return dr::__detail::parallel_for(q, sycl::range<>(rng::distance(r)),
                                    [=](auto idx) { arr[idx] = value; });
}

template <typename T, dr::remote_contiguous_range R>
auto fill(R &&r, const T &value) {
  fill_async(r, value).wait();
  return rng::end(r);
}

template <typename T, dr::distributed_contiguous_range DR>
sycl::event fill_async(DR &&r, const T &value) {
  std::vector<sycl::event> events;

  for (auto &&segment : dr::ranges::segments(r)) {
    auto e = dr::shp::fill_async(segment, value);
    events.push_back(e);
  }

  return dr::shp::__detail::combine_events(events);
}

template <typename T, dr::distributed_contiguous_range DR>
auto fill(DR &&r, const T &value) {
  fill_async(r, value).wait();
  return rng::end(r);
}

template <typename T, dr::distributed_iterator Iter>
auto fill(Iter first, Iter last, const T &value) {
  fill_async(rng::subrange(first, last), value).wait();
  return last;
}

} // namespace dr::shp
