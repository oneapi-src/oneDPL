// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <memory>
#include <sycl/sycl.hpp>

#include <dr/shp/detail.hpp>

namespace dr::shp {

template <typename T, typename Event = sycl::event> class future {
public:
  using event_type = Event;

  future(std::unique_ptr<T> &&value, const std::vector<Event> &events)
      : value_(std::move(value)), events_(events) {}

  future(T &&value, const std::vector<Event> &events)
      : value_(new T(std::move(value))), events_(events) {}

  void update(const Event &event) { events_.push_back(event); }

  future(future &&) = default;
  future &operator=(future &&) = default;

  future(const future &) = delete;
  future &operator=(const future &) = delete;

  T get() {
    wait();
    return std::move(*value_);
  }

  std::vector<Event> events() const { return events_; }

  T &value() const { return *value_; }

  void wait() { __detail::wait(events_); }

private:
  std::unique_ptr<T> value_;
  std::vector<Event> events_;
};

} // namespace dr::shp
