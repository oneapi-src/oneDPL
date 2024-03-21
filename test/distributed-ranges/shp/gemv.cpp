// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "xhp-tests.hpp"

TEST(SparseMatrix, Gemv) {
  std::size_t m = 100;
  std::size_t k = 100;

  experimental::dr::shp::sparse_matrix<float> a(
      {m, k}, 0.1f,
      experimental::dr::shp::block_cyclic({experimental::dr::shp::tile::div, experimental::dr::shp::tile::div},
                            {experimental::dr::shp::nprocs(), 1}));

  experimental::dr::shp::distributed_vector<float> b(k, 1.f);
  experimental::dr::shp::distributed_vector<float> c(m, 0.f);

  experimental::dr::shp::gemv(c, a, b);

  std::vector<float> c_local(m);

  experimental::dr::shp::copy(c.begin(), c.end(), c_local.begin());

  std::vector<float> c_ref(m, 0.f);

  for (auto &&[index, v] : a) {
    auto &&[i, k] = index;

    c_ref[i] += v;
  }

  EXPECT_TRUE(fp_equal(c_ref, c_local))
      << fmt::format("Reference:\n  {}\nActual:\n  {}\n", c_ref, c_local);
}
