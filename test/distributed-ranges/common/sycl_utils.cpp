// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "xhp-tests.hpp"

#ifdef SYCL_LANGUAGE_VERSION

using T = float;

TEST(SYCLUtils, ParalelFor1D) {
  const std::size_t size = 10;
  sycl::queue q;
  sycl::range range(size - 1);

  auto a = sycl::malloc_shared<T>(size, q);
  auto b = sycl::malloc_shared<T>(size, q);
  std::fill(a, a + size, 99);
  std::fill(b, b + size, 99);
  auto seta = [a](auto i) { a[i] = i; };
  auto setb = [b](auto i) { b[i] = i; };
  q.parallel_for(range, seta).wait();
  dr::__detail::parallel_for(q, range, setb).wait();

  EXPECT_EQ(rng::span(a, size), rng::span(b, size));
}

void set(auto col_size, auto base, auto index) {
  base[(index[0] + 1) * col_size + index[1] + 1] = 22;
}

TEST(SYCLUtils, ParalelFor2D) {
  const std::size_t row_size = 5, col_size = row_size,
                    size = row_size * col_size;
  sycl::queue q;
  sycl::range range(row_size - 2, col_size - 2);

  auto a = sycl::malloc_shared<T>(size, q);
  auto b = sycl::malloc_shared<T>(size, q);
  md::mdspan mda(a, row_size, col_size);
  md::mdspan mdb(b, row_size, col_size);
  std::fill(a, a + size, 99);
  std::fill(b, b + size, 99);
  auto seta = [mda](auto index) { mda(index[0], index[1]) = 22; };
  auto setb = [mdb](auto index) { mdb(index[0], index[1]) = 22; };

  q.parallel_for(range, seta).wait();
  dr::__detail::parallel_for(q, range, setb).wait();

  EXPECT_EQ(rng::span(a, size), rng::span(b, size))
      << fmt::format("a:\n{}b:\n{}", mda, mdb);
}

TEST(SYCLUtils, ParalelFor3D) {
  const std::size_t x_size = 5, y_size = x_size, z_size = x_size,
                    size = x_size * y_size * z_size;
  sycl::queue q;
  sycl::range range(x_size - 2, y_size - 2, z_size - 2);

  auto a = sycl::malloc_shared<T>(size, q);
  auto b = sycl::malloc_shared<T>(size, q);
  md::mdspan mda(a, x_size, y_size, z_size);
  md::mdspan mdb(b, x_size, y_size, z_size);

  std::fill(a, a + size, 99);
  std::fill(b, b + size, 99);
  auto seta = [mda](auto index) { mda(index[0], index[1], index[2]) = 22; };
  auto setb = [mdb](auto index) { mdb(index[0], index[1], index[2]) = 22; };

  q.parallel_for(range, seta).wait();
  dr::__detail::parallel_for(q, range, setb).wait();

  EXPECT_EQ(rng::span(a, size), rng::span(b, size))
      << fmt::format("a:\n{}b:\n{}", mda, mdb);
}

#endif // SYCL_LANGUAGE_VERSION
