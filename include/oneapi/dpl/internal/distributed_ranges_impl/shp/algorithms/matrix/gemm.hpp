// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <dr/shp/algorithms/matrix/local_gemm.hpp>
#include <dr/shp/containers/distributed_dense_matrix.hpp>

namespace dr::shp {

template <typename T>
void gemm(distributed_dense_matrix<T> &a, distributed_dense_matrix<T> &b,
          distributed_dense_matrix<T> &c) {
  gemm_buffered(a, b, c);
}

template <typename T>
void gemm_inplace(distributed_dense_matrix<T> &a,
                  distributed_dense_matrix<T> &b,
                  distributed_dense_matrix<T> &c) {
  // Matrix dimensions must match (algorithm requirement)
  assert(c.shape()[0] == a.shape()[0]);
  assert(c.shape()[1] == b.shape()[1]);
  assert(a.shape()[1] == b.shape()[0]);

  // Tile grid dimensions must match (implementation limitation)

  assert(c.grid_shape()[0] == a.grid_shape()[0]);
  assert(c.grid_shape()[1] == b.grid_shape()[1]);
  assert(a.grid_shape()[1] == b.grid_shape()[0]);

  std::vector<sycl::event> events;
  events.reserve(c.grid_shape()[0] * c.grid_shape()[1] * a.grid_shape()[1]);

  for (std::size_t i = 0; i < c.grid_shape()[0]; i++) {
    for (std::size_t j = 0; j < c.grid_shape()[1]; j++) {
      // For each tile of the output C matrix
      auto &&c_tile = c.tile({i, j});

      std::vector<sycl::event> local_events;
      local_events.reserve(a.grid_shape()[1]);

      std::size_t k_offset = i + j;
      for (std::size_t k_ = 0; k_ < a.grid_shape()[1]; k_++) {
        std::size_t k = (k_ + k_offset) % a.grid_shape()[1];

        auto &&a_tile = a.tile({i, k});
        auto &&b_tile = b.tile({k, j});

        auto &&q = __detail::queue(dr::ranges::rank(c_tile));

        auto e = __detail::local_gemm(q, __detail::local(a_tile),
                                      __detail::local(b_tile),
                                      __detail::local(c_tile), local_events);

        local_events.push_back(e);
      }

      for (auto &&e : local_events) {
        events.push_back(e);
      }
    }
  }

  __detail::wait(events);
}

template <typename T>
void gemm_buffered(distributed_dense_matrix<T> &a,
                   distributed_dense_matrix<T> &b,
                   distributed_dense_matrix<T> &c) {
  // Matrix dimensions must match (algorithm requirement)
  assert(c.shape()[0] == a.shape()[0]);
  assert(c.shape()[1] == b.shape()[1]);
  assert(a.shape()[1] == b.shape()[0]);

  // Tile grid dimensions must match (implementation limitation)

  assert(c.grid_shape()[0] == a.grid_shape()[0]);
  assert(c.grid_shape()[1] == b.grid_shape()[1]);
  assert(a.grid_shape()[1] == b.grid_shape()[0]);

  std::vector<std::thread> threads;

  std::atomic<double> communication = 0;
  std::atomic<double> compute = 0;

  for (std::size_t i = 0; i < c.grid_shape()[0]; i++) {
    for (std::size_t j = 0; j < c.grid_shape()[1]; j++) {
      auto c_local = c.tile({i, j});

      threads.emplace_back([c_local, i, j, &a, &b, &communication, &compute] {
        auto &&q = __detail::queue(dr::ranges::rank(c_local));

        std::size_t a_elem = a.tile_shape()[0] * a.tile_shape()[1];
        std::size_t b_elem = b.tile_shape()[0] * b.tile_shape()[1];
        std::size_t buffer_size = std::max(a_elem, b_elem);

        dr::shp::device_allocator<T> gpu_allocator(q);
        dr::shp::buffered_allocator buffered_allocator(gpu_allocator,
                                                       buffer_size, 2);
        auto &&allocator = buffered_allocator;

        std::size_t k_offset = i + j;

        for (std::size_t k_ = 0; k_ < a.grid_shape()[1]; k_++) {
          std::size_t k = (k_ + k_offset) % a.grid_shape()[1];

          auto begin = std::chrono::high_resolution_clock::now();
          auto a_tile = a.get_tile({i, k}, allocator);
          auto b_tile = b.get_tile({k, j}, allocator);
          auto end = std::chrono::high_resolution_clock::now();
          double duration = std::chrono::duration<double>(end - begin).count();
          communication += duration;

          dr::shp::dense_matrix_view a_local(a_tile);
          dr::shp::dense_matrix_view b_local(b_tile);

          begin = std::chrono::high_resolution_clock::now();
          __detail::local_gemm(q, __detail::local(a_local),
                               __detail::local(b_local),
                               __detail::local(c_local))
              .wait();
          end = std::chrono::high_resolution_clock::now();
          duration = std::chrono::duration<double>(end - begin).count();
          compute += duration;
        }
      });
    }
  }

  for (auto &&t : threads) {
    t.join();
  }

  bool debug_print = false;

  if (debug_print) {
    std::cout << "communication total: " << (double)communication << std::endl;
    std::cout << "compute total: " << (double)compute << std::endl;
  }
}

template <typename T>
void gemm_buffered_async(distributed_dense_matrix<T> &a,
                         distributed_dense_matrix<T> &b,
                         distributed_dense_matrix<T> &c) {
  // Matrix dimensions must match (algorithm requirement)
  assert(c.shape()[0] == a.shape()[0]);
  assert(c.shape()[1] == b.shape()[1]);
  assert(a.shape()[1] == b.shape()[0]);

  // Tile grid dimensions must match (implementation limitation)

  assert(c.grid_shape()[0] == a.grid_shape()[0]);
  assert(c.grid_shape()[1] == b.grid_shape()[1]);
  assert(a.grid_shape()[1] == b.grid_shape()[0]);

  std::vector<std::thread> threads;

  std::atomic<double> issue = 0;
  std::atomic<double> sync = 0;
  std::atomic<double> compute = 0;

  for (std::size_t i = 0; i < c.grid_shape()[0]; i++) {
    for (std::size_t j = 0; j < c.grid_shape()[1]; j++) {
      auto c_local = c.tile({i, j});

      threads.emplace_back([c_local, i, j, &a, &b, &issue, &sync, &compute] {
        auto &&q = __detail::queue(dr::ranges::rank(c_local));

        std::size_t a_elem = a.tile_shape()[0] * a.tile_shape()[1];
        std::size_t b_elem = b.tile_shape()[0] * b.tile_shape()[1];
        std::size_t buffer_size = std::max(a_elem, b_elem);

        dr::shp::device_allocator<T> gpu_allocator(q);
        dr::shp::buffered_allocator buffered_allocator(gpu_allocator,
                                                       buffer_size, 4);
        auto &&allocator = buffered_allocator;

        std::size_t k_offset = i + j;

        auto begin = std::chrono::high_resolution_clock::now();
        auto a_f =
            a.get_tile_async({i, k_offset % a.grid_shape()[1]}, allocator);
        // a_f.wait();
        auto b_f =
            b.get_tile_async({k_offset % a.grid_shape()[1], j}, allocator);
        // b_f.wait();
        auto end = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration<double>(end - begin).count();
        issue += duration;

        for (std::size_t k_ = 0; k_ < a.grid_shape()[1]; k_++) {
          std::size_t k = (k_ + k_offset) % a.grid_shape()[1];

          auto begin = std::chrono::high_resolution_clock::now();
          auto a_tile = a_f.get();
          auto b_tile = b_f.get();
          auto end = std::chrono::high_resolution_clock::now();
          double duration = std::chrono::duration<double>(end - begin).count();
          sync += duration;

          dr::shp::dense_matrix_view a_local(a_tile);
          dr::shp::dense_matrix_view b_local(b_tile);

          if (k_ + 1 < a.grid_shape()[1]) {
            begin = std::chrono::high_resolution_clock::now();
            a_f = a.get_tile_async({i, (k + 1) % a.grid_shape()[1]}, allocator);
            // a_f.wait();
            b_f = b.get_tile_async({(k + 1) % a.grid_shape()[1], j}, allocator);
            // b_f.wait();
            end = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration<double>(end - begin).count();
            issue += duration;
          }

          begin = std::chrono::high_resolution_clock::now();
          __detail::local_gemm(q, __detail::local(a_local),
                               __detail::local(b_local),
                               __detail::local(c_local))
              .wait();
          end = std::chrono::high_resolution_clock::now();
          duration = std::chrono::duration<double>(end - begin).count();
          compute += duration;
        }
      });
    }
  }

  for (auto &&t : threads) {
    t.join();
  }

  bool debug_print = false;

  if (debug_print) {
    std::cout << "sync total: " << (double)sync << std::endl;
    std::cout << "issue total: " << (double)issue << std::endl;
    std::cout << "compute total: " << (double)compute << std::endl;
  }
}

} // namespace dr::shp
