// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "xhp-tests.hpp"

TEST(SparseMatrix, Gemv) {
  std::size_t m = 100;
  std::size_t k = 100;

  dr::shp::sparse_matrix<float> a(
      {m, k}, 0.1f,
      dr::shp::block_cyclic({dr::shp::tile::div, dr::shp::tile::div},
                            {dr::shp::nprocs(), 1}));

  dr::shp::distributed_vector<float> b(k, 1.f);
  dr::shp::distributed_vector<float> c(m, 0.f);

  dr::shp::gemv(c, a, b);

  std::vector<float> c_local(m);

  dr::shp::copy(c.begin(), c.end(), c_local.begin());

  std::vector<float> c_ref(m, 0.f);

  for (auto &&[index, v] : a) {
    auto &&[i, k] = index;

    c_ref[i] += v;
  }

  EXPECT_TRUE(fp_equal(c_ref, c_local))
      << fmt::format("Reference:\n  {}\nActual:\n  {}\n", c_ref, c_local);
}
