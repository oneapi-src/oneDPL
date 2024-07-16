// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This file incorporates work covered by the following copyright and permission
// notice:
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <oneapi/dpl/internal/distributed_ranges_impl/shp/init.hpp>
#include <sycl/sycl.hpp>
#include <type_traits>

namespace oneapi::dpl::experimental::dr::shp
{

template <typename T>
requires(std::is_trivially_copyable_v<T> || std::is_void_v<T>) class device_ref
{
  public:
    device_ref() = delete;
    ~device_ref() = default;
    device_ref(const device_ref&) = default;

    device_ref(T* pointer) : pointer_(pointer) {}
    device_ref(T& value) : pointer_(&value) {}

    operator T() const
    {
#ifdef __SYCL_DEVICE_ONLY__
        return *pointer_;
#else
        auto&& q = __detail::default_queue();
        char buffer[sizeof(T)] __attribute__((aligned(sizeof(T))));
        q.memcpy(reinterpret_cast<T*>(buffer), pointer_, sizeof(T)).wait();
        return *reinterpret_cast<T*>(buffer);
#endif
    }

    device_ref
    operator=(const T& value) const requires(!std::is_const_v<T>)
    {
#ifdef __SYCL_DEVICE_ONLY__
        *pointer_ = value;
#else
        auto&& q = __detail::default_queue();
        q.memcpy(pointer_, &value, sizeof(T)).wait();
#endif
        return *this;
    }

    device_ref
    operator=(const device_ref& other) const
    {
#ifdef __SYCL_DEVICE_ONLY__
        *pointer_ = *other.pointer_;
#else
        T value = other;
        *this = value;
#endif
        return *this;
    }

  private:
    T* pointer_;
};

} // namespace oneapi::dpl::experimental::dr::shp
