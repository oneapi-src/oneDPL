// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <limits>

#include <dr/detail/utils.hpp>

#ifdef SYCL_LANGUAGE_VERSION

#include <sycl/sycl.hpp>

namespace dr::__detail {

// With the ND-range workaround, the maximum kernel size is
// `std::numeric_limits<std::int32_t>::max()` rounded down to
// the nearest multiple of the block size.
inline std::size_t max_kernel_size_(std::size_t block_size = 128) {
  std::size_t max_kernel_size = std::numeric_limits<std::int32_t>::max();
  return (max_kernel_size / block_size) * block_size;
}

// This is a workaround to avoid performance degradation
// in DPC++ for odd range sizes.
template <typename Fn>
sycl::event parallel_for_workaround(sycl::queue &q, sycl::range<1> numWorkItems,
                                    Fn &&fn, std::size_t block_size = 128) {
  std::size_t num_blocks = (numWorkItems.size() + block_size - 1) / block_size;

  int32_t range_size = numWorkItems.size();

  auto event = q.parallel_for(
      sycl::nd_range<>(num_blocks * block_size, block_size), [=](auto nd_idx) {
        auto idx = nd_idx.get_global_id(0);
        if (idx < range_size) {
          fn(idx);
        }
      });
  return event;
}

template <typename Fn>
sycl::event parallel_for_64bit(sycl::queue &q, sycl::range<1> numWorkItems,
                               Fn &&fn) {
  std::size_t block_size = 128;
  std::size_t max_kernel_size = max_kernel_size_(block_size);

  std::vector<sycl::event> events;
  for (std::size_t base_idx = 0; base_idx < numWorkItems.size();
       base_idx += max_kernel_size) {
    std::size_t launch_size =
        std::min(numWorkItems.size() - base_idx, max_kernel_size);

    auto e = parallel_for_workaround(
        q, launch_size,
        [=](sycl::id<1> idx_) {
          sycl::id<1> idx(base_idx + idx_);
          fn(idx);
        },
        block_size);

    events.push_back(e);
  }

  auto e = q.submit([&](auto &&h) {
    h.depends_on(events);
    // Empty host task necessary due to [CMPLRLLVM-46542]
    h.host_task([] {});
  });

  return e;
}

//
// return true if the device can be partitioned by affinity domain
//
inline auto partitionable(sycl::device device) {
  // Earlier commits used the query API, but they return true even
  // though a partition will fail:  Intel MPI mpirun with multiple
  // processes.
  try {
    device.create_sub_devices<
        sycl::info::partition_property::partition_by_affinity_domain>(
        sycl::info::partition_affinity_domain::numa);
  } catch (sycl::exception const &e) {
    if (e.code() == sycl::errc::invalid ||
        e.code() == sycl::errc::feature_not_supported) {
      return false;
    } else {
      throw;
    }
  }

  return true;
}

// Convert a global range to a nd_range using generic block size level
// gpu requires uniform size workgroup, so round up to a multiple of a
// workgroup.
template <int Dim> auto nd_range(sycl::range<Dim> global) {
  if constexpr (Dim == 1) {
    sycl::range local(128);
    return sycl::nd_range<Dim>(sycl::range(round_up(global[0], local[0])),
                               local);
  } else if constexpr (Dim == 2) {
    sycl::range local(16, 16);
    return sycl::nd_range<Dim>(sycl::range(round_up(global[0], local[0]),
                                           round_up(global[1], local[1])),
                               local);
  } else if constexpr (Dim == 3) {
    sycl::range local(8, 8, 8);
    return sycl::nd_range<Dim>(sycl::range(round_up(global[0], local[0]),
                                           round_up(global[1], local[1]),
                                           round_up(global[2], local[2])),
                               local);
  } else {
    assert(false);
    return sycl::range(0);
  }
}

template <typename Fn>
sycl::event parallel_for_nd(sycl::queue &q, sycl::range<1> global, Fn &&fn) {
  return q.parallel_for(nd_range(global), [=](auto nd_idx) {
    auto idx0 = nd_idx.get_global_id(0);
    if (idx0 < global[0]) {
      fn(idx0);
    }
  });
}

template <typename Fn>
sycl::event parallel_for_nd(sycl::queue &q, sycl::range<2> global, Fn &&fn) {
  return q.parallel_for(nd_range(global), [=](auto nd_idx) {
    auto idx0 = nd_idx.get_global_id(0);
    auto idx1 = nd_idx.get_global_id(1);
    if (idx0 < global[0] && idx1 < global[1]) {
      fn(std::array{idx0, idx1});
    }
  });
}

template <typename Fn>
sycl::event parallel_for_nd(sycl::queue &q, sycl::range<3> global, Fn &&fn) {
  return q.parallel_for(nd_range(global), [=](auto nd_idx) {
    auto idx0 = nd_idx.get_global_id(0);
    auto idx1 = nd_idx.get_global_id(1);
    auto idx2 = nd_idx.get_global_id(2);
    if (idx0 < global[0] && idx1 < global[1] && idx2 < global[2]) {
      fn(std::array{idx0, idx1, idx2});
    }
  });
}

auto combine_events(sycl::queue &q, const auto &events) {
  return q.submit([&](auto &&h) {
    h.depends_on(events);
    // Empty host task necessary due to [CMPLRLLVM-46542]
    h.host_task([] {});
  });
}

template <typename Fn>
sycl::event parallel_for(sycl::queue &q, sycl::range<1> numWorkItems, Fn &&fn) {
  std::size_t block_size = 128;
  std::size_t max_kernel_size = max_kernel_size_();

  if (numWorkItems.size() < max_kernel_size) {
    return parallel_for_workaround(q, numWorkItems, std::forward<Fn>(fn),
                                   block_size);
  } else {
    return parallel_for_64bit(q, numWorkItems, std::forward<Fn>(fn));
  }
}

template <typename Fn>
sycl::event parallel_for(sycl::queue &q, sycl::range<2> global, Fn &&fn) {
  auto max = std::numeric_limits<std::int32_t>::max();
  assert(global[0] < max && global[1] < max);
  return parallel_for_nd(q, global, fn);
}

template <typename Fn>
sycl::event parallel_for(sycl::queue &q, sycl::range<3> global, Fn &&fn) {
  auto max = std::numeric_limits<std::int32_t>::max();
  assert(global[0] < max && global[1] < max && global[2] < max);
  return parallel_for_nd(q, global, fn);
}

using event = sycl::event;

} // namespace dr::__detail

#else

namespace dr::__detail {

class event {
public:
  void wait() {}
};

} // namespace dr::__detail

#endif // SYCL_LANGUAGE_VERSION
