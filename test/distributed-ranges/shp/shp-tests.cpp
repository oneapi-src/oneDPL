// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include <iostream>
#include <string>
#include <vector>

#include "xhp-tests.hpp"

void printhelp() {
  std::cout << "Usage: shp-tests [options]\n"
            << "Options:\n"
            << "  --drhelp\t\tPrint help\n"
            << "  -d, --devicesCount\tNumber of GPUs to create\n";
}

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);

  const std::string drhelpOption = "--drhelp";
  const std::string dOption = "-d";
  const std::string devicesCountOption = "--devicesCount";

  bool drhelp = false;
  unsigned int devicesCount = 0;

  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == drhelpOption) {
      printhelp();
      return 0;
    } else if (arg == dOption || arg == devicesCountOption) {
      if (i + 1 < argc) {
        devicesCount = std::stoi(argv[i + 1]);
        i++; // Skip the next argument
      } else {
        std::cout << "Missing value for " << arg << " option\n";
        return 1;
      }
    } else {
      std::cout << "Unknown option: " << arg << "\n";
      return 1;
    }
  }

  auto devices = xhp::get_numa_devices(sycl::default_selector_v);

  if (devicesCount > 0) {
    unsigned int i = 0;
    while (devices.size() < devicesCount) {
      devices.push_back(devices[i++]);
    }
    devices.resize(devicesCount); // if too many devices
  }

  dr::shp::init(devices);

  for (auto &device : devices) {
    std::cout << "  Device: " << device.get_info<sycl::info::device::name>()
              << "\n";
  }

  return RUN_ALL_TESTS();
}
