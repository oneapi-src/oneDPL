//==============================================================
// Copyright © 2019 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <iomanip>
#include <iostream>

#include <CL/sycl.hpp>

#include <dpstd/iterator>
#include <dpstd/algorithm>
#include <dpstd/execution>

#include "utils.hpp"

#if !__SYCL_UNNAMED_LAMBDA__
// In case of missing the -fsycl-unnamed-lambda option or using
// the -fno-sycl-unnamed-lambda option in command line
// we should name policy explicitly.
// Policy names are below
class Gamma;
#endif

int main() {
  // Image size is width x height
  int width = 2560;
  int height = 1600;

  Img<ImgFormat::BMP> image{width, height};
  ImgFractal fractal{width, height};

  // Lambda to process image with gamma = 2
  auto gamma_f = [](ImgPixel& pixel) {
    float v = (0.3f * pixel.r + 0.59f * pixel.g + 0.11f * pixel.b) / 255.0f;

    std::uint8_t gamma_pixel = static_cast<std::uint8_t>(255 * v * v);
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

  Img<ImgFormat::BMP> image2 = image;
  image.write("fractal_original.bmp");

  // call standard serial function for correctness check
  image.fill(gamma_f);
  image.write("fractal_gamma.bmp");

  // create a queue for tasks, sent to the device
  cl::sycl::queue queue(cl::sycl::default_selector{});

  // We need to have the scope to have data in image2 after buffer's destruction
  {
    // create a buffer, being responsible for moving data around and counting
    // dependencies
    cl::sycl::buffer<ImgPixel, 1> buffer(image2.data(),
                                         image2.width() * image2.height());

    // create iterator to pass buffer to the algorithm
    auto buffer_begin = dpstd::begin(buffer);
    auto buffer_end = dpstd::end(buffer);

    // choose policy we will provide to the algorithm
#if __SYCL_UNNAMED_LAMBDA__
    auto new_policy = dpstd::execution::default_policy;
#else
    // create named policy from queue
    auto new_policy = dpstd::execution::make_device_policy<Gamma>(queue);
#endif
    // call std::for_each with DPC++ support
    std::for_each(new_policy, buffer_begin, buffer_end, gamma_f);
  }

  // check correctness
  if (check(image.begin(), image.end(), image2.begin())) {
    std::cout << "success";
  } else {
    std::cout << "fail";
  }
  std::cout << ". Run on "
            << queue.get_device().get_info<cl::sycl::info::device::name>()
            << std::endl;

  image.write("fractal_gamma_pstlwithsycl.bmp");

  return 0;
}
