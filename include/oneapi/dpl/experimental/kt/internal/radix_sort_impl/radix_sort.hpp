#pragma once

#include <CL/sycl.hpp>
#include <cstdint>

template <typename KeyT = uint32_t, bool IS_DESC = false, uint32_t WARP_THREADS = 16>
struct RadixSortKernel {
    KeyT *_array;
    uint32_t _size;
    RadixSortKernel(KeyT *array, uint32_t size) : _array(array), _size(size) {}

    void process(sycl::queue &q) {
        if (_size < std::pow(2, 15.5)) {
            RadixSortKernel_Impl<8, 128, 16, true, WARP_THREADS>(_array, _size).process(q);
        } else if(_size < std::pow(2, 18.5)) {
            RadixSortKernel_Impl<8, 256, 16, true, WARP_THREADS>(_array, _size).process(q);
        } else if(_size < std::pow(2, 19.5)) {
            RadixSortKernel_Impl<8, 256, 20, true, WARP_THREADS>(_array, _size).process(q);
        } else if(_size < std::pow(2, 22.5)) {
            RadixSortKernel_Impl<8, 256, 40, true, WARP_THREADS>(_array, _size).process(q);
        } else if(_size < std::pow(2, 23.5)) {
            RadixSortKernel_Impl<8, 256, 28, true, WARP_THREADS>(_array, _size).process(q);
        } else if(_size < std::pow(2, 24.5)) {
            RadixSortKernel_Impl<8, 128, 44, true, WARP_THREADS>(_array, _size).process(q);
        } else if(_size < std::pow(2, 27.5)) {
            RadixSortKernel_Impl<8, 256, 32, true, WARP_THREADS>(_array, _size).process(q);
        } else if(_size < std::pow(2, 28.5)) {
            RadixSortKernel_Impl<8, 256, 28, true, WARP_THREADS>(_array, _size).process(q);
        } else{
            RadixSortKernel_Impl<8, 128, 48, true, WARP_THREADS>(_array, _size).process(q);
        }
    }
};
