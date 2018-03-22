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

#include <vector>
#include <algorithm>

class image {
public:
    union pixel {
        std::uint8_t bgra[4];
        std::uint32_t value;
        pixel() {}
        template <typename T> pixel(T b, T g, T r) {
            bgra[0] = (std::uint8_t)b, bgra[1] = (std::uint8_t)g, bgra[2] = (std::uint8_t)r, bgra[3] = 0;
        }
    };
public:
    image(int w = 1920, int h = 1080);

    int width() const { return my_width; }
    int height() const { return my_height; }

    void write(const char* fname) const;
    void fill(std::uint8_t r, std::uint8_t g, std::uint8_t b, int x = -1, int y = -1);

    template <typename F>
    void fill(F f) {

        if(my_data.empty())
            reset(my_width, my_height);

        int i = -1;
        int w = this->my_width;
        std::for_each(my_data.begin(), my_data.end(), [&i, w, f](image::pixel& p) {
            ++i;
            int x = i / w, y = i % w;
            auto val = f(x, y);
            if(val > 255)
                val = 255;
            p = image::pixel(val, val, val);
        });
    }

    std::vector<pixel*>& rows() { return my_rows; }

private:
    void reset(int w, int h);

private:
    //don't allow copying
    image(const image&);
    void operator=(const image&);

private:
    int my_width;
    int my_height;
    int my_padSize;

    std::vector<pixel> my_data; //raw raster data
    std::vector<pixel*> my_rows;

    //data structures 'file' and 'info' are using to store an image as BMP file
    //for more details see https://en.wikipedia.org/wiki/BMP_file_format
    using BITMAPFILEHEADER = struct {
        std::uint16_t sizeRest; // field is not from specification,
                            // was added for alignemt. store size of rest of the fields
        std::uint16_t type;
        std::uint32_t size;
        std::uint32_t reserved;
        std::uint32_t offBits;
    };
    BITMAPFILEHEADER file;

    using BITMAPINFOHEADER = struct {
        std::uint32_t size;
        std::int32_t width;
        std::int32_t height;
        std::uint16_t planes;
        std::uint16_t bitCount;
        std::uint32_t compression;
        std::uint32_t sizeImage;
        std::int32_t xPelsPerMeter;
        std::int32_t yPelsPerMeter;
        std::uint32_t clrUsed;
        std::uint32_t clrImportant;
    };
    BITMAPINFOHEADER info;
};
