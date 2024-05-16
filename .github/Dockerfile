# docker build -t oneapi-gcc13 .
# docker run -ti --rm oneapi-gcc13
FROM intel/oneapi:latest
RUN apt update
RUN add-apt-repository ppa:ubuntu-toolchain-r/test
RUN apt install -y g++-13
RUN apt install -y libgtest-dev
RUN apt install -y libcxxopts-dev

RUN apt remove --purge --auto-remove -y cmake
RUN wget https://cmake.org/files/v3.29/cmake-3.29.2.tar.gz
RUN tar -xzvf cmake-3.29.2.tar.gz && cd cmake-3.29.2 && ./bootstrap && make -j12 && make install

RUN apt install -y libfmt-dev

CMD /bin/bash