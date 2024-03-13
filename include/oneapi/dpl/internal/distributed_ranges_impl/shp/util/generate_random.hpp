// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <concepts>
#include <dr/shp/views/csr_matrix_view.hpp>
#include <map>
#include <random>

namespace dr::shp {

namespace {

template <typename T> struct uniform_distribution {
  using type = std::uniform_int_distribution<T>;
};

template <std::floating_point T> struct uniform_distribution<T> {
  using type = std::uniform_real_distribution<T>;
};

template <typename T>
using uniform_distribution_t = typename uniform_distribution<T>::type;

} // namespace

template <typename T = float, std::integral I = std::size_t>
auto generate_random_csr(dr::index<I> shape, double density = 0.01,
                         unsigned int seed = 0) {

  assert(density >= 0.0 && density < 1.0);

  std::map<std::pair<I, I>, T> tuples;

  std::size_t nnz = density * shape[0] * shape[1];

  std::mt19937 gen(seed);
  std::uniform_int_distribution<I> row(0, shape[0] - 1);
  std::uniform_int_distribution<I> column(0, shape[1] - 1);

  uniform_distribution_t<T> value_gen(0, 1);

  while (tuples.size() < nnz) {
    auto i = row(gen);
    auto j = column(gen);
    if (tuples.find({i, j}) == tuples.end()) {
      T value = value_gen(gen);
      tuples.insert({{i, j}, value});
    }
  }

  T *values = new T[nnz];
  I *rowptr = new I[shape[0] + 1];
  I *colind = new I[nnz];

  rowptr[0] = 0;

  std::size_t r = 0;
  std::size_t c = 0;
  for (auto iter = tuples.begin(); iter != tuples.end(); ++iter) {
    auto &&[index, value] = *iter;
    auto &&[i, j] = index;

    values[c] = value;
    colind[c] = j;

    while (r < i) {
      if (r + 1 > shape[0]) {
        // TODO: exception?
        // throw std::runtime_error("csr_matrix_impl_: given invalid matrix");
      }
      rowptr[r + 1] = c;
      r++;
    }
    c++;

    if (c > nnz) {
      // TODO: exception?
      // throw std::runtime_error("csr_matrix_impl_: given invalid matrix");
    }
  }

  for (; r < shape[0]; r++) {
    rowptr[r + 1] = nnz;
  }

  return csr_matrix_view(values, rowptr, colind, shape, nnz, 0);
}

} // namespace dr::shp
