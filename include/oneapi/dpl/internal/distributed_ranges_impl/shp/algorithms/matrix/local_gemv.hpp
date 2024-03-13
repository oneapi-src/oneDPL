// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <dr/detail/ranges_shim.hpp>
#include <dr/shp/containers/sparse_matrix.hpp>
#include <dr/shp/util.hpp>

#ifdef USE_MKL
#include <oneapi/mkl.hpp>
#endif

namespace dr::shp {

namespace __detail {

template <typename T, typename I, std::random_access_iterator Iter,
          typename... Args>
  requires(std::is_same_v<std::iter_value_t<Iter>, T>)
auto custom_gemv(sycl::queue &q, csr_matrix_view<T, I, Args...> a, Iter b,
                 Iter c, const std::vector<sycl::event> &dependencies = {}) {
  std::size_t wg = 32;

  auto event = q.submit([&](auto &&h) {
    h.depends_on(dependencies);
    h.parallel_for(sycl::nd_range<1>(a.shape()[0] * wg, wg), [=](auto item) {
      auto row_index = item.get_group(0);
      auto local_id = item.get_local_id();
      auto group_size = item.get_local_range(0);

      auto row = a.row(row_index);

      for (std::size_t idx = local_id; idx < row.size(); idx += group_size) {
        auto &&[index, a_v] = row[idx];
        auto &&[i, k] = index;

        auto &&b_v = *(b + k);
        auto &&c_v = *(c + i);

        sycl::atomic_ref<T, sycl::memory_order::relaxed,
                         sycl::memory_scope::device>
            c_ref(c_v);

        c_ref += a_v * b_v;
      }
    });
  });
  return event;
}

#ifdef USE_MKL

template <typename T, typename I, std::random_access_iterator Iter,
          typename... Args>
  requires(std::is_same_v<std::iter_value_t<Iter>, T>)
auto mkl_gemv(sycl::queue &q, csr_matrix_view<T, I, Args...> a, Iter b, Iter c,
              const std::vector<sycl::event> &dependencies = {}) {

  oneapi::mkl::sparse::matrix_handle_t a_handle;
  oneapi::mkl::sparse::init_matrix_handle(&a_handle);

  auto rowptr = dr::shp::__detail::local(a.rowptr_data());
  auto colind = dr::shp::__detail::local(a.colind_data());
  auto values = dr::shp::__detail::local(a.values_data());

  oneapi::mkl::sparse::set_csr_data(q, a_handle, a.shape()[0], a.shape()[1],
                                    oneapi::mkl::index_base::zero, rowptr,
                                    colind, values);

  auto event =
      oneapi::mkl::sparse::gemv(q, oneapi::mkl::transpose::nontrans, T(1),
                                a_handle, b, T(1), c, dependencies);
  return event;
}

template <typename T, typename I, std::random_access_iterator Iter,
          typename... Args>
  requires(std::is_same_v<std::iter_value_t<Iter>, T>)
auto local_gemv(sycl::queue &q, csr_matrix_view<T, I, Args...> a, Iter b,
                Iter c, const std::vector<sycl::event> &dependencies = {}) {
  return mkl_gemv(q, a, b, c, dependencies);
}

#else

template <typename T, typename I, std::random_access_iterator Iter,
          typename... Args>
  requires(std::is_same_v<std::iter_value_t<Iter>, T>)
auto local_gemv(sycl::queue &q, csr_matrix_view<T, I, Args...> a, Iter b,
                Iter c, const std::vector<sycl::event> &dependencies = {}) {
  return custom_gemv(q, a, b, c, dependencies);
}

#endif

} // namespace __detail

} // namespace dr::shp
