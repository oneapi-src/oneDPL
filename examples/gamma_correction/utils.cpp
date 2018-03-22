/*
    Copyright (c) 2017-2018 Intel Corporation

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.




*/

#include <iostream>
#include <fstream>
#include <vector>
#include <cassert>
#include "utils.h"

void image::reset(int w, int h) {

    if(w <= 0 || h <= 0) {
        std::cout << "Warning: Invalid image size.\n";
        return;
    }

    my_width = w, my_height = h;

    //reset raw data
    my_data.resize(my_width*my_height);
    my_rows.resize(my_height);

    //reset rows
    for(int i = 0; i < my_rows.size(); ++i)
        my_rows[i] = &my_data[0]+i*my_width;

    my_padSize = (4-(w*sizeof(my_data[0]))%4)%4;
    int sizeData = w*h*sizeof(my_data[0]) + h*my_padSize;
    int sizeAll = sizeData + sizeof(file) + sizeof(info);

    //BITMAPFILEHEADER
    file.sizeRest = 14;
    file.type = 0x4d42; //same as 'BM' in ASCII
    file.size = sizeAll;
    file.reserved = 0;
    file.offBits = 54;

    //BITMAPINFOHEADER
    info.size = 40;
    info.width = w;
    info.height = h;
    info.planes = 1;
    info.bitCount = 32;
    info.compression = 0;
    info.sizeImage = sizeData;
    info.yPelsPerMeter = 0;
    info.xPelsPerMeter = 0;
    info.clrUsed = 0;
    info.clrImportant = 0;
}

image::image(int w, int h) {
    reset(w, h);
}

void image::fill(std::uint8_t r, std::uint8_t g, std::uint8_t b, int x, int y) {
    if(my_data.empty())
        return;

    assert(my_data.size() == my_width*my_height);
    assert(my_rows.size() == my_height);

    if(x < 0 && y < 0) //fill whole image
        std::fill(my_data.begin(), my_data.end(), pixel(b, g, r));
    else {
        auto& bgra = my_data[my_width*x + y].bgra;
        bgra[3] = 0, bgra[2] = r, bgra[1] = g, bgra[0] = b;
    }
}

void image::write(const char* fname) const {

    if(my_data.empty()) {
        std::cout << "Warning: An image is empty.\n";
        return;
    }

    assert(my_width > 0 && my_height > 0);

    std::ofstream stream(fname);

    assert(file.sizeRest == sizeof(file)-sizeof(file.sizeRest));
    stream.write((char*)&file.type, file.sizeRest);

    assert(info.size == sizeof(info));
    stream.write((char*)&info, info.size);

    assert(info.sizeImage == my_data.size() * sizeof(my_data[0]));
    stream.write((char*)my_data[0].bgra, my_data.size()*sizeof(my_data[0]));
}
