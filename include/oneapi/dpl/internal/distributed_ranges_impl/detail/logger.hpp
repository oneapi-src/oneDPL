// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <chrono>
#include <fstream>
#include <iostream>

#include <vendor/source_location/source_location.hpp>

#include <dr/detail/format_shim.hpp>
#include <dr/detail/ranges_shim.hpp>

namespace dr {

class timer {
public:
  timer() : begin_(std::chrono::high_resolution_clock::now()) {}

  auto elapsed() {
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(end - begin_).count();
  }

private:
  std::chrono::time_point<std::chrono::high_resolution_clock> begin_;
};

class logger {
public:
  enum filters { base, for_each, transpose, mdspan_view, mpi, last };

  logger() { rng::fill(enabled_, true); }

  void set_file(std::ofstream &fout) { fout_ = &fout; }

  void filter(const std::vector<std::string> &names) {
    if (names.size() == 0) {
      return;
    }

    // Disable everything
    rng::fill(enabled_, false);

    // Enabled selected filters
    for (const auto &name : names) {
      std::size_t index = filters::last;
      for (std::size_t i = 0; i < filter_names_.size(); i++) {
        if (name == filter_names_[i]) {
          index = i;
        }
      }
      if (index == filters::last) {
        std::cerr << "Ignoring unrecognized filter: " << name << "\n";
      } else {
        enabled_[index] = true;
      }
    }
  }

#ifdef DR_FORMAT

  template <typename... Args>
  void debug(const nostd::source_location &location,
             fmt::format_string<Args...> format, Args &&...args) {
    if (fout_ && enabled_[filters::base]) {
      *fout_ << fmt::format(format, std::forward<Args>(args)...) << " <"
             << location.file_name() << ":" << location.line() << ">\n";
      fout_->flush();
    }
  }

  template <typename... Args>
  void debug(fmt::format_string<Args...> format, Args &&...args) {
    debug(filters::base, format, std::forward<Args>(args)...);
  }

  template <typename... Args>
  void debug(filters filter, fmt::format_string<Args...> format,
             Args &&...args) {
    if (fout_ && enabled_[filter]) {
      *fout_ << fmt::format(format, std::forward<Args>(args)...);
      fout_->flush();
    }
  }

#else

  template <typename... Args>
  void debug(const nostd::source_location &location, std::string format,
             Args &&...args) {}

  template <typename... Args> void debug(std::string format, Args &&...args) {}

  template <typename... Args>
  void debug(filters filter, std::string format, Args &&...args) {}

#endif

private:
  std::ofstream *fout_ = nullptr;
  std::array<bool, filters::last> enabled_;
  std::array<std::string, filters::last> filter_names_ = {
      "base", "for_each", "transpose", "mdspan_view", "mpi"};
};

inline logger drlog;

#define DRLOG(...)                                                             \
  dr::drlog.debug(nostd::source_location::current(), __VA_ARGS__)

} // namespace dr
