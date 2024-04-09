// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <iterator>
#include <oneapi/dpl/internal/distributed_ranges_impl/detail/ranges.hpp>
#include <oneapi/dpl/internal/distributed_ranges_impl/detail/sycl_utils.hpp>
#include <oneapi/dpl/internal/distributed_ranges_impl/shp/device_ptr.hpp>
#include <oneapi/dpl/internal/distributed_ranges_impl/shp/init.hpp>
#include <sycl/sycl.hpp>

namespace oneapi::dpl::experimental::dr::shp {

namespace __detail {

inline constexpr auto local = ranges::__detail::local;

template <typename Src, typename Dest>
concept is_syclmemcopyable = std::is_same_v<std::remove_const_t<Src>, Dest> &&
                             std::is_trivially_copyable_v<Dest>;

template <std::contiguous_iterator Iter>
sycl::usm::alloc get_pointer_type(Iter iter) {
  return sycl::get_pointer_type(std::to_address(iter), context());
}

template <typename T>
sycl::usm::alloc get_pointer_type(device_ptr<T> ptr) {
  return sycl::get_pointer_type(ptr.get_raw_pointer(), context());
}

template <std::contiguous_iterator Iter>
sycl::device get_pointer_device(Iter iter) {
  return sycl::get_pointer_device(std::to_address(iter), context());
}

template <typename T> sycl::device get_pointer_device(device_ptr<T> ptr) {
  return sycl::get_pointer_device(ptr.get_raw_pointer(), context());
}

template <typename InputIt> sycl::queue &get_queue_for_pointer(InputIt iter) {
  if (get_pointer_type(iter) == sycl::usm::alloc::device) {
    auto device = get_pointer_device(iter);
    return __detail::queue(device);
  } else {
    return default_queue();
  }
}

template <typename InputIt, typename OutputIt>
sycl::queue &get_queue_for_pointers(InputIt iter, OutputIt iter2) {
  if (get_pointer_type(iter) == sycl::usm::alloc::device) {
    auto device = get_pointer_device(iter);
    return __detail::queue(device);
  } else if (get_pointer_type(iter2) == sycl::usm::alloc::device) {
    auto device = get_pointer_device(iter2);
    return __detail::queue(device);
  } else {
    return default_queue();
  }
}

inline sycl::event combine_events(sycl::queue &q,
                                  const std::vector<sycl::event> &events) {
  auto e = q.submit([&](auto &&h) {
    h.depends_on(events);
    h.host_task([] {});
  });

  return e;
}

inline sycl::event combine_events(const std::vector<sycl::event> &events) {
  auto &&q = __detail::queue(0);
  return combine_events(q, events);
}

inline void wait(sycl::event &event) { event.wait(); }

inline void wait(sycl::event &&event) { event.wait(); }

inline void wait(const std::vector<sycl::event> &events) {
  sycl::event::wait(events);
}

} // namespace __detail

} // namespace oneapi::dpl::experimental::dr::shp
