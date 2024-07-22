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

#include <iostream>
#include <string>
#include <vector>

#include "xhp_tests.hpp"

void printhelp() {
  std::cout << "Usage: sp_tests [options]\n"
            << "Options:\n"
            << "  --drhelp\t\tPrint help\n"
            << "  -d, --devicesCount\tNumber of GPUs to create\n";
}

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);

  const std::string drhelpOption = "--drhelp";
  const std::string dOption = "-d";
  const std::string devicesCountOption = "--devicesCount";

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

  dr::sp::init(devices);

  for (auto &device : devices) {
    std::cout << "  Device: " << device.get_info<sycl::info::device::name>()
              << "\n";
  }

  return RUN_ALL_TESTS();
}
