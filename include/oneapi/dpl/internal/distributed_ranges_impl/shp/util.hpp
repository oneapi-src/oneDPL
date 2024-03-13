// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <iostream>
#include <sycl/sycl.hpp>

namespace dr::shp {

template <typename Selector> sycl::device select_device(Selector &&selector) {
  sycl::device d;

  try {
    d = sycl::device(std::forward<Selector>(selector));
    std::cout << "Running on device \""
              << d.get_info<sycl::info::device::name>() << "\"" << std::endl;
  } catch (sycl::exception const &e) {
    std::cout << "Cannot select an accelerator\n" << e.what() << "\n";
    std::cout << "Using a CPU device\n";
    d = sycl::device(sycl::cpu_selector_v);
  }
  return d;
}

inline void list_devices() {
  auto platforms = sycl::platform::get_platforms();

  for (auto &platform : platforms) {
    std::cout << "Platform: " << platform.get_info<sycl::info::platform::name>()
              << std::endl;

    auto devices = platform.get_devices();
    for (auto &device : devices) {
      std::cout << "  Device: " << device.get_info<sycl::info::device::name>()
                << std::endl;
    }
  }
}

inline void print_device_details(std::span<sycl::device> devices) {
  std::size_t device_id = 0;
  for (auto &&device : devices) {
    std::cout << "Device " << device_id << ": "
              << device.get_info<sycl::info::device::name>() << std::endl;
    device_id++;
  }
}

template <typename Selector> void list_devices(Selector &&selector) {
  sycl::platform p(std::forward<Selector>(selector));
  auto devices = p.get_devices();

  printf("--Platform Info-----------------\n");

  printf("Platform %s has %lu root devices.\n",
         p.get_info<sycl::info::platform::name>().c_str(), devices.size());

  for (std::size_t i = 0; i < devices.size(); i++) {
    auto &&device = devices[i];

    printf("  %lu %s\n", i,
           device.get_info<sycl::info::device::name>().c_str());

    auto subdevices = device.create_sub_devices<
        sycl::info::partition_property::partition_by_affinity_domain>(
        sycl::info::partition_affinity_domain::numa);

    printf("   Subdevices:\n");
    for (std::size_t j = 0; j < subdevices.size(); j++) {
      auto &&subdevice = subdevices[j];
      printf("     %lu.%lu %s\n", i, j,
             subdevice.get_info<sycl::info::device::name>().c_str());
    }
  }

  printf("--------------------------------\n");
}

inline std::vector<sycl::device>
trim_devices(const std::vector<sycl::device> &devices, std::size_t n_devices) {
  std::vector<sycl::device> trimmed_devices = devices;

  if (n_devices < devices.size()) {
    trimmed_devices.resize(n_devices);
  }
  return trimmed_devices;
}

template <typename Selector>
std::vector<sycl::device> get_numa_devices_impl_(Selector &&selector) {
  std::vector<sycl::device> devices;

  sycl::platform p(std::forward<Selector>(selector));
  auto root_devices = p.get_devices();

  for (auto &&root_device : root_devices) {
    auto subdevices = root_device.create_sub_devices<
        sycl::info::partition_property::partition_by_affinity_domain>(
        sycl::info::partition_affinity_domain::numa);

    for (auto &&subdevice : subdevices) {
      devices.push_back(subdevice);
    }
  }

  return devices;
}

template <typename Selector>
std::vector<sycl::device> get_devices(Selector &&selector) {
  sycl::platform p(std::forward<Selector>(selector));
  return p.get_devices();
}

template <typename Selector>
std::vector<sycl::device> get_numa_devices(Selector &&selector) {
  try {
    return get_numa_devices_impl_(std::forward<Selector>(selector));
  } catch (sycl::exception const &e) {
    if (e.code() == sycl::errc::feature_not_supported) {
      std::cerr << "NUMA partitioning not supported, returning root devices..."
                << std::endl;
      return get_devices(std::forward<Selector>(selector));
    } else {
      throw;
    }
  }
}

// Return exactly `n` devices obtained using the selector `selector`.
// May duplicate devices
template <typename Selector>
std::vector<sycl::device> get_duplicated_devices(Selector &&selector,
                                                 std::size_t n) {
  auto devices = get_numa_devices(std::forward<Selector>(selector));

  if (devices.size() >= n) {
    return std::vector<sycl::device>(devices.begin(), devices.begin() + n);
  } else {
    std::size_t i = 0;
    while (devices.size() < n) {
      auto d = devices[i++];
      devices.push_back(d);
    }
    return devices;
  }
}

template <typename Range> void print_range(Range &&r, std::string label = "") {
  std::size_t indent = 1;

  if (label != "") {
    std::cout << "\"" << label << "\": ";
    indent += label.size() + 4;
  }

  std::string indent_whitespace(indent, ' ');

  std::cout << "[";
  std::size_t columns = 10;
  std::size_t count = 1;
  for (auto iter = r.begin(); iter != r.end(); ++iter) {
    std::cout << static_cast<rng::range_value_t<Range>>(*iter);

    auto next = iter;
    ++next;
    if (next != r.end()) {
      std::cout << ", ";
      if (count % columns == 0) {
        std::cout << "\n" << indent_whitespace;
      }
    }
    ++count;
  }
  std::cout << "]" << std::endl;
}

template <typename Matrix>
void print_matrix(Matrix &&m, std::string label = "") {
  std::cout << m.shape()[0] << " x " << m.shape()[1] << " matrix with "
            << m.size() << " stored values";
  if (label != "") {
    std::cout << " \"" << label << "\"";
  }
  std::cout << std::endl;

  for (auto &&tuple : m) {
    auto &&[index, value] = tuple;
    auto &&[i, j] = index;

    std::cout << "(" << i << ", " << j << "): " << value << std::endl;
  }
}

template <typename R> void print_range_details(R &&r, std::string label = "") {
  if (label != "") {
    std::cout << "\"" << label << "\" ";
  }

  std::cout << "distributed range with " << rng::size(dr::ranges::segments(r))
            << " segments." << std::endl;

  std::size_t idx = 0;
  for (auto &&segment : dr::ranges::segments(r)) {
    std::cout << "Seg " << idx++ << ", size " << segment.size() << " (rank "
              << dr::ranges::rank(segment) << ")" << std::endl;
  }
}

template <dr::distributed_range R>
void range_details(R &&r, std::size_t width = 80) {
  std::size_t size = rng::size(r);

  for (auto &&[idx, segment] :
       dr::__detail::enumerate(dr::ranges::segments(r))) {
    std::size_t local_size = rng::size(segment);

    double percent = double(local_size) / size;

    std::size_t num_chars = percent * width;
    num_chars = std::max(num_chars, std::size_t(3));

    std::size_t whitespace = num_chars - 3;

    std::size_t initial_whitespace = whitespace / 2;
    std::size_t after_whitespace = whitespace - initial_whitespace;

    std::cout << "[" << std::string(initial_whitespace, ' ')
              << dr::ranges::rank(segment) << std::string(after_whitespace, ' ')
              << "]";
  }
  std::cout << std::endl;
}

namespace __detail {

template <typename T>
concept sycl_device_selector = requires(T &t, const sycl::device &device) {
  { t(device) } -> std::convertible_to<int>;
};

}

} // namespace dr::shp
