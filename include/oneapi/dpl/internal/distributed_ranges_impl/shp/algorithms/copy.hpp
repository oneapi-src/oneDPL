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

// Copy between contiguous ranges
template <std::contiguous_iterator InputIt, std::contiguous_iterator OutputIt>
  requires __detail::is_syclmemcopyable<std::iter_value_t<InputIt>,
                                        std::iter_value_t<OutputIt>>
sycl::event copy_async(InputIt first, InputIt last, OutputIt d_first) {
  // auto &&q = dr::shp::__detail::default_queue();
  auto &&q = __detail::get_queue_for_pointers(first, d_first);
  return q.memcpy(std::to_address(d_first), std::to_address(first),
                  sizeof(std::iter_value_t<InputIt>) * (last - first));
}

/// Copy
template <std::contiguous_iterator InputIt, std::contiguous_iterator OutputIt>
  requires __detail::is_syclmemcopyable<std::iter_value_t<InputIt>,
                                        std::iter_value_t<OutputIt>>
OutputIt copy(InputIt first, InputIt last, OutputIt d_first) {
  copy_async(first, last, d_first).wait();
  return d_first + (last - first);
}

// Copy from contiguous range to device
template <std::contiguous_iterator Iter, typename T>
  requires __detail::is_syclmemcopyable<std::iter_value_t<Iter>, T>
sycl::event copy_async(Iter first, Iter last, device_ptr<T> d_first) {
  // auto &&q = dr::shp::__detail::default_queue();
  auto &&q = __detail::get_queue_for_pointers(first, d_first);
  return q.memcpy(d_first.get_raw_pointer(), std::to_address(first),
                  sizeof(T) * (last - first));
}

template <std::contiguous_iterator Iter, typename T>
  requires __detail::is_syclmemcopyable<std::iter_value_t<Iter>, T>
device_ptr<T> copy(Iter first, Iter last, device_ptr<T> d_first) {
  copy_async(first, last, d_first).wait();
  return d_first + (last - first);
}

// Copy from device to contiguous range
template <typename T, std::contiguous_iterator Iter>
  requires __detail::is_syclmemcopyable<T, std::iter_value_t<Iter>>
sycl::event copy_async(device_ptr<T> first, device_ptr<T> last, Iter d_first) {
  // auto &&q = dr::shp::__detail::default_queue();
  auto &&q = __detail::get_queue_for_pointers(first, d_first);
  return q.memcpy(std::to_address(d_first), first.get_raw_pointer(),
                  sizeof(T) * (last - first));
}

template <typename T, std::contiguous_iterator Iter>
  requires __detail::is_syclmemcopyable<T, std::iter_value_t<Iter>>
Iter copy(device_ptr<T> first, device_ptr<T> last, Iter d_first) {
  copy_async(first, last, d_first).wait();
  return d_first + (last - first);
}

// Copy from device to device
template <typename T>
  requires(!std::is_const_v<T> && std::is_trivially_copyable_v<T>)
sycl::event copy_async(device_ptr<std::add_const_t<T>> first,
                       device_ptr<std::add_const_t<T>> last,
                       device_ptr<T> d_first) {
  // auto &&q = dr::shp::__detail::default_queue();
  auto &&q = __detail::get_queue_for_pointers(first, d_first);
  return q.memcpy(d_first.get_raw_pointer(), first.get_raw_pointer(),
                  sizeof(T) * (last - first));
}

template <typename T>
  requires(!std::is_const_v<T> && std::is_trivially_copyable_v<T>)
sycl::event copy_async(sycl::queue &q, device_ptr<std::add_const_t<T>> first,
                       device_ptr<std::add_const_t<T>> last,
                       device_ptr<T> d_first) {
  return q.memcpy(d_first.get_raw_pointer(), first.get_raw_pointer(),
                  sizeof(T) * (last - first));
}

template <typename T>
  requires(!std::is_const_v<T> && std::is_trivially_copyable_v<T>)
device_ptr<T> copy(device_ptr<std::add_const_t<T>> first,
                   device_ptr<std::add_const_t<T>> last,
                   device_ptr<T> d_first) {
  copy_async(first, last, d_first).wait();
  return d_first + (last - first);
}

// Copy from local range to distributed range
template <std::forward_iterator InputIt, dr::distributed_iterator OutputIt>
  requires __detail::is_syclmemcopyable<std::iter_value_t<InputIt>,
                                        std::iter_value_t<OutputIt>>
sycl::event copy_async(InputIt first, InputIt last, OutputIt d_first) {
  auto &&segments = dr::ranges::segments(d_first);
  auto segment_iter = rng::begin(segments);

  std::vector<sycl::event> events;

  while (first != last) {
    auto &&segment = *segment_iter;
    auto size = rng::distance(segment);

    std::size_t n_to_copy = std::min<size_t>(size, rng::distance(first, last));

    auto local_last = first;
    rng::advance(local_last, n_to_copy);

    events.emplace_back(
        dr::shp::copy_async(first, local_last, rng::begin(segment)));

    ++segment_iter;
    rng::advance(first, n_to_copy);
  }

  return dr::shp::__detail::combine_events(events);
}

auto copy(rng::contiguous_range auto r, dr::distributed_iterator auto d_first) {
  return copy(rng::begin(r), rng::end(r), d_first);
}

auto copy(dr::distributed_range auto r, std::contiguous_iterator auto d_first) {
  return copy(rng::begin(r), rng::end(r), d_first);
}

template <std::forward_iterator InputIt, dr::distributed_iterator OutputIt>
  requires __detail::is_syclmemcopyable<std::iter_value_t<InputIt>,
                                        std::iter_value_t<OutputIt>>
OutputIt copy(InputIt first, InputIt last, OutputIt d_first) {
  copy_async(first, last, d_first).wait();
  return d_first + (last - first);
}

// Copy from distributed range to local range
template <dr::distributed_iterator InputIt, std::forward_iterator OutputIt>
  requires __detail::is_syclmemcopyable<std::iter_value_t<InputIt>,
                                        std::iter_value_t<OutputIt>>
sycl::event copy_async(InputIt first, InputIt last, OutputIt d_first) {
  auto dist = rng::distance(first, last);
  auto segments =
      dr::__detail::take_segments(dr::ranges::segments(first), dist);

  std::vector<sycl::event> events;

  for (auto &&segment : segments) {
    auto size = rng::distance(segment);

    events.emplace_back(
        dr::shp::copy_async(rng::begin(segment), rng::end(segment), d_first));

    rng::advance(d_first, size);
  }

  return dr::shp::__detail::combine_events(events);
}

template <dr::distributed_iterator InputIt, std::forward_iterator OutputIt>
  requires __detail::is_syclmemcopyable<std::iter_value_t<InputIt>,
                                        std::iter_value_t<OutputIt>>
OutputIt copy(InputIt first, InputIt last, OutputIt d_first) {
  copy_async(first, last, d_first).wait();
  return d_first + (last - first);
}

// Copy from distributed range to distributed range
template <dr::distributed_iterator InputIt, dr::distributed_iterator OutputIt>
  requires __detail::is_syclmemcopyable<std::iter_value_t<InputIt>,
                                        std::iter_value_t<OutputIt>>
sycl::event copy_async(InputIt first, InputIt last, OutputIt d_first) {
  auto dist = rng::distance(first, last);
  auto segments =
      dr::__detail::take_segments(dr::ranges::segments(first), dist);

  std::vector<sycl::event> events;

  for (auto &&segment : segments) {
    auto size = rng::distance(segment);

    events.emplace_back(
        dr::shp::copy_async(rng::begin(segment), rng::end(segment), d_first));

    rng::advance(d_first, size);
  }

  return dr::shp::__detail::combine_events(events);
}

template <dr::distributed_iterator InputIt, dr::distributed_iterator OutputIt>
  requires __detail::is_syclmemcopyable<std::iter_value_t<InputIt>,
                                        std::iter_value_t<OutputIt>>
OutputIt copy(InputIt first, InputIt last, OutputIt d_first) {
  copy_async(first, last, d_first).wait();
  return d_first + (last - first);
}

// Ranges versions

// Distributed to distributed
template <dr::distributed_range R, dr::distributed_iterator O>
  requires __detail::is_syclmemcopyable<rng::range_value_t<R>,
                                        std::iter_value_t<O>>
sycl::event copy_async(R &&r, O result) {
  return copy_async(rng::begin(r), rng::end(r), result);
}

template <dr::distributed_range R, dr::distributed_iterator O>
  requires __detail::is_syclmemcopyable<rng::range_value_t<R>,
                                        std::iter_value_t<O>>
O copy(R &&r, O result) {
  return copy(rng::begin(r), rng::end(r), result);
}

} // namespace dr::shp
