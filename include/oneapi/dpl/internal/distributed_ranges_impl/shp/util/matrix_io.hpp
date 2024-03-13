// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <algorithm>
#include <cassert>
#include <fstream>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include <dr/shp/util/coo_matrix.hpp>
#include <dr/shp/views/csr_matrix_view.hpp>

namespace dr::shp {

namespace __detail {

// Preconditions:
// 1) `tuples` sorted by row, column
// 2) `tuples` has shape `shape`
// 3) `tuples` has `nnz` elements
template <typename Tuples, typename Allocator>
auto convert_to_csr(Tuples &&tuples, dr::index<> shape, std::size_t nnz,
                    Allocator &&allocator) {
  auto &&[index, v] = *tuples.begin();
  auto &&[i, j] = index;

  using T = std::remove_reference_t<decltype(v)>;
  using I = std::remove_reference_t<decltype(i)>;

  typename std::allocator_traits<Allocator>::template rebind_alloc<I>
      i_allocator(allocator);

  T *values = allocator.allocate(nnz);
  I *rowptr = i_allocator.allocate(shape[0] + 1);
  I *colind = i_allocator.allocate(nnz);

  rowptr[0] = 0;

  std::size_t r = 0;
  std::size_t c = 0;
  for (auto iter = tuples.begin(); iter != tuples.end(); ++iter) {
    auto &&[index, value] = *iter;
    auto &&[i, j] = index;

    values[c] = value;
    colind[c] = j;

    while (r < i) {
      assert(r + 1 <= shape[0]);
      // throw std::runtime_error("csr_matrix_impl_: given invalid matrix");
      rowptr[r + 1] = c;
      r++;
    }
    c++;

    assert(c <= nnz);
    // throw std::runtime_error("csr_matrix_impl_: given invalid matrix");
  }

  for (; r < shape[0]; r++) {
    rowptr[r + 1] = nnz;
  }

  return csr_matrix_view(values, rowptr, colind,
                         dr::index<I>(shape[0], shape[1]), nnz, 0);
}

/// Read in the Matrix Market file at location `file_path` and a return
/// a coo_matrix data structure with its contents.
template <typename T, typename I = std::size_t>
inline coo_matrix<T, I> mmread(std::string file_path, bool one_indexed = true) {
  using size_type = std::size_t;

  std::ifstream f;

  f.open(file_path.c_str());

  if (!f.is_open()) {
    // TODO better choice of exception.
    throw std::runtime_error("mmread: cannot open " + file_path);
  }

  std::string buf;

  // Make sure the file is matrix market matrix, coordinate, and check whether
  // it is symmetric. If the matrix is symmetric, non-diagonal elements will
  // be inserted in both (i, j) and (j, i).  Error out if skew-symmetric or
  // Hermitian.
  std::getline(f, buf);
  std::istringstream ss(buf);
  std::string item;
  ss >> item;
  if (item != "%%MatrixMarket") {
    throw std::runtime_error(file_path +
                             " could not be parsed as a Matrix Market file.");
  }
  ss >> item;
  if (item != "matrix") {
    throw std::runtime_error(file_path +
                             " could not be parsed as a Matrix Market file.");
  }
  ss >> item;
  if (item != "coordinate") {
    throw std::runtime_error(file_path +
                             " could not be parsed as a Matrix Market file.");
  }
  bool pattern;
  ss >> item;
  if (item == "pattern") {
    pattern = true;
  } else {
    pattern = false;
  }
  // TODO: do something with real vs. integer vs. pattern?
  ss >> item;
  bool symmetric;
  if (item == "general") {
    symmetric = false;
  } else if (item == "symmetric") {
    symmetric = true;
  } else {
    throw std::runtime_error(file_path + " has an unsupported matrix type");
  }

  bool outOfComments = false;
  while (!outOfComments) {
    std::getline(f, buf);

    if (buf[0] != '%') {
      outOfComments = true;
    }
  }

  I m, n, nnz;
  // std::istringstream ss(buf);
  ss.clear();
  ss.str(buf);
  ss >> m >> n >> nnz;

  // NOTE for symmetric matrices: `nnz` holds the number of stored values in
  // the matrix market file, while `matrix.nnz_` will hold the total number of
  // stored values (including "mirrored" symmetric values).
  coo_matrix<T, I> matrix({m, n});
  if (symmetric) {
    matrix.reserve(2 * nnz);
  } else {
    matrix.reserve(nnz);
  }

  size_type c = 0;
  while (std::getline(f, buf)) {
    I i, j;
    T v;
    std::istringstream ss(buf);
    if (!pattern) {
      ss >> i >> j >> v;
    } else {
      ss >> i >> j;
      v = T(1);
    }
    if (one_indexed) {
      i--;
      j--;
    }

    if (i >= m || j >= n) {
      throw std::runtime_error(
          "read_MatrixMarket: file has nonzero out of bounds.");
    }

    matrix.push_back({{i, j}, v});

    if (symmetric && i != j) {
      matrix.push_back({{j, i}, v});
    }

    c++;
    if (c > nnz) {
      throw std::runtime_error("read_MatrixMarket: error reading Matrix Market "
                               "file, file has more nonzeros than reported.");
    }
  }

  auto sort_fn = [](const auto &a, const auto &b) {
    auto &&[a_index, a_value] = a;
    auto &&[b_index, b_value] = b;
    auto &&[a_i, a_j] = a_index;
    auto &&[b_i, b_j] = b_index;
    if (a_i < b_i) {
      return true;
    } else if (a_i == b_i) {
      if (a_j < b_j) {
        return true;
      }
    }
    return false;
  };

  std::sort(matrix.begin(), matrix.end(), sort_fn);

  f.close();

  return matrix;
}

template <typename T, typename I, typename Allocator, typename... Args>
void destroy_csr_matrix_view(dr::shp::csr_matrix_view<T, I, Args...> view,
                             Allocator &&alloc) {
  alloc.deallocate(view.values_data(), view.size());
  typename std::allocator_traits<Allocator>::template rebind_alloc<I> i_alloc(
      alloc);
  i_alloc.deallocate(view.colind_data(), view.size());
  i_alloc.deallocate(view.rowptr_data(), view.shape()[0] + 1);
}

} // namespace __detail

template <typename T, typename I>
auto create_distributed(dr::shp::csr_matrix_view<T, I> local_mat,
                        const matrix_partition &partition) {
  dr::shp::sparse_matrix<T, I> a(local_mat.shape(), partition);

  std::vector<dr::shp::csr_matrix_view<T, I>> views;
  std::vector<sycl::event> events;
  views.reserve(a.grid_shape()[0] * a.grid_shape()[1]);

  for (I i = 0; i < a.grid_shape()[0]; i++) {
    for (I j = 0; j < a.grid_shape()[1]; j++) {
      auto &&tile = a.tile({i, j});
      dr::index<I> row_bounds(i * a.tile_shape()[0],
                              i * a.tile_shape()[0] + tile.shape()[0]);
      dr::index<I> column_bounds(j * a.tile_shape()[1],
                                 j * a.tile_shape()[1] + tile.shape()[1]);

      auto local_submat = local_mat.submatrix(row_bounds, column_bounds);

      auto submatrix_shape = dr::index<I>(row_bounds[1] - row_bounds[0],
                                          column_bounds[1] - column_bounds[0]);

      auto copied_submat = __detail::convert_to_csr(
          local_submat, submatrix_shape, rng::distance(local_submat),
          std::allocator<T>{});

      auto e = a.copy_tile_async({i, j}, copied_submat);

      views.push_back(copied_submat);
      events.push_back(e);
    }
  }
  __detail::wait(events);

  for (auto &&view : views) {
    __detail::destroy_csr_matrix_view(view, std::allocator<T>{});
  }

  return a;
}

template <typename T, typename I = std::size_t>
auto mmread(std::string file_path, const matrix_partition &partition,
            bool one_indexed = true) {
  auto m = __detail::mmread<T, I>(file_path, one_indexed);
  auto shape = m.shape();
  auto nnz = m.size();

  auto local_mat = __detail::convert_to_csr(m, shape, nnz, std::allocator<T>{});

  auto a = create_distributed(local_mat, partition);

  __detail::destroy_csr_matrix_view(local_mat, std::allocator<T>{});

  return a;
}

template <typename T, typename I = std::size_t>
auto mmread(std::string file_path, bool one_indexed = true) {
  return mmread<T, I>(
      file_path,
      dr::shp::block_cyclic({dr::shp::tile::div, dr::shp::tile::div},
                            {dr::shp::nprocs(), 1}),
      one_indexed);
}

} // namespace dr::shp
