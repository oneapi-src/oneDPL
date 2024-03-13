// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <dr/shp/views/dense_matrix_view.hpp>

#ifdef USE_MKL
#include <oneapi/mkl.hpp>
#endif

namespace dr::shp {

namespace __detail {

template <typename T>
auto custom_gemm(sycl::queue &q, shp::dense_matrix_view<T> a,
                 shp::dense_matrix_view<T> b, shp::dense_matrix_view<T> c,
                 const std::vector<sycl::event> &dependencies = {}) {
  assert(c.shape()[0] == a.shape()[0]);
  assert(c.shape()[1] == b.shape()[1]);
  assert(a.shape()[1] == b.shape()[0]);

  std::size_t M = c.shape()[0];
  std::size_t N = c.shape()[1];
  std::size_t K = a.shape()[1];

  auto a_p = a.data();
  auto b_p = b.data();
  auto c_p = c.data();

  auto e = q.parallel_for(sycl::range<3>{M, K, N}, [=](auto idx) {
    auto i = idx[0];
    auto k = idx[1];
    auto j = idx[2];

    sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device>
        c_ref(c_p[i * N + j]);

    c_ref += a_p[i * K + k] * b_p[k * N + j];
  });
  return e;
}

#ifdef USE_MKL

template <typename T>
auto mkl_gemm(sycl::queue &q, shp::dense_matrix_view<T> a,
              shp::dense_matrix_view<T> b, shp::dense_matrix_view<T> c,
              const std::vector<sycl::event> &dependencies = {}) {
  assert(c.shape()[0] == a.shape()[0]);
  assert(c.shape()[1] == b.shape()[1]);
  assert(a.shape()[1] == b.shape()[0]);

  auto event = oneapi::mkl::blas::row_major::gemm(
      q, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans,
      c.shape()[0], c.shape()[1], a.shape()[1], T(1), a.data(), a.ld(),
      b.data(), b.ld(), T(1), c.data(), c.ld(), dependencies);

  return event;
}

template <typename T>
auto local_gemm(sycl::queue &q, shp::dense_matrix_view<T> a,
                shp::dense_matrix_view<T> b, shp::dense_matrix_view<T> c,
                const std::vector<sycl::event> &dependencies = {}) {
  return mkl_gemm(q, a, b, c, dependencies);
}

#else

template <typename T>
auto local_gemm(sycl::queue &q, shp::dense_matrix_view<T> a,
                shp::dense_matrix_view<T> b, shp::dense_matrix_view<T> c,
                const std::vector<sycl::event> &dependencies = {}) {
  return custom_gemm(q, a, b, c, dependencies);
}

#endif

} // namespace __detail

} // namespace dr::shp
