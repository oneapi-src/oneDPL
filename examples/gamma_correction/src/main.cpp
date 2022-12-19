//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

// oneDPL headers should be included before standard headers
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/iterator>

#if !BUILD_FOR_HOST
#include <sycl/sycl.hpp>
#endif
#include <iomanip>
#include <iostream>

#include "utils.hpp"

int main() {
  // Image size is width x height
  int width = 2560;
  int height = 1600;

  Img<ImgFormat::BMP> image{width, height};
  ImgFractal fractal{width, height};

  // Lambda to process image with gamma = 2
  auto gamma_f = [](ImgPixel& pixel) {
    auto v = (0.3f * pixel.r + 0.59f * pixel.g + 0.11f * pixel.b) / 255.0f;

    auto gamma_pixel = static_cast<uint8_t>(255 * v * v);
    if (gamma_pixel > 255) gamma_pixel = 255;
    pixel.set(gamma_pixel, gamma_pixel, gamma_pixel, gamma_pixel);
  };

  // fill image with created fractal
  int index = 0;
  image.fill([&index, width, &fractal](ImgPixel& pixel) {
    int x = index % width;
    int y = index / width;

    auto fractal_pixel = fractal(x, y);
    if (fractal_pixel < 0) fractal_pixel = 0;
    if (fractal_pixel > 255) fractal_pixel = 255;
    pixel.set(fractal_pixel, fractal_pixel, fractal_pixel, fractal_pixel);

    ++index;
  });

  string original_image = "fractal_original.bmp";
  string processed_image = "fractal_gamma.bmp";
  Img<ImgFormat::BMP> image2 = image;
  image.write(original_image);

  // call standard serial function for correctness check
  image.fill(gamma_f);

#if BUILD_FOR_HOST
  using Row = ImgPixel*;
  std::vector<Row> rows;
  for (size_t i = 0; i < height; ++i) {
    rows.push_back(image2.data() + i * width);
  }
  std::for_each(oneapi::dpl::execution::par, rows.begin(), rows.end(),
                [width, gamma_f](Row& r) {
                  std::transform(oneapi::dpl::execution::unseq, r, r + width, r,
                                 [gamma_f](ImgPixel& p) {
                                   gamma_f(p);
                                   return p;
                                 });
                });
#else
  // use default policy for algorithms execution
  auto policy = oneapi::dpl::execution::dpcpp_default;
  // We need to have the scope to have data in image2 after buffer's destruction
  {
    // create a buffer, being responsible for moving data around and counting
    // dependencies
    sycl::buffer<ImgPixel> b(image2.data(), image2.width() * image2.height());

    // create iterator to pass buffer to the algorithm
    auto b_begin = oneapi::dpl::begin(b);
    auto b_end = oneapi::dpl::end(b);

    // call std::for_each with DPC++ support
    std::for_each(policy, b_begin, b_end, gamma_f);
  }
#endif

  image2.write(processed_image);
  // check correctness
  if (check(image.begin(), image.end(), image2.begin())) {
    cout << "success\n";
  } else {
    cout << "fail\n";
    return 1;
  }
#if !BUILD_FOR_HOST
  cout << "Run on "
       << policy.queue()
              .get_device()
              .template get_info<sycl::info::device::name>()
       << "\n";
#endif
  cout << "Original image is in " << original_image << "\n";
  cout << "Image after applying gamma correction on the device is in "
       << processed_image << "\n";

  return 0;
}
