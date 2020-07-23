#FROM ubuntu:18.04
FROM nvidia/cudagl:10.0-devel-ubuntu18.04
MAINTAINER naverlabs "kapture@naverlabs.com"

# Set correct environment variables.
ENV LC_ALL C
ENV DEBIAN_FRONTEND noninteractive
ARG   MAKE_OPTIONS="-j8"

# Get dependencies
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    git \
    wget\
    curl \
    python3.6 python3-pip \
    pandoc asciidoctor \
    cmake \
    build-essential \
    libboost-all-dev \
    libsuitesparse-dev \
    libfreeimage-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libglew-dev \
    freeglut3-dev \
    libxmu-dev \
    libxi-dev \
    libatlas-base-dev \
    libsuitesparse-dev \
    libcgal-qt5-dev \
    libqt5opengl5-dev \
    qt5-default \
    x11-apps \
    mesa-utils \
  && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade setuptools wheel twine

########################################################################################################################
# COLMAP ###############################################################################################################
RUN mkdir /src

# Eigen 3.2.10
RUN cd /src && git clone -b 3.2.10 https://gitlab.com/libeigen/eigen.git eigen
RUN mkdir -p /src/eigen/build
WORKDIR /src/eigen/build
RUN     cmake \
        -DCMAKE_BUILD_TYPE=Release \
         .. && \
        make ${MAKE_OPTIONS} && make install && make clean

# ceres 1.14.0
RUN cd /src && git clone -b 1.14.0 https://github.com/ceres-solver/ceres-solver.git
RUN mkdir -p /src/ceres-solver/build
WORKDIR /src/ceres-solver/build
RUN     cmake \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_TESTING=OFF \
        -DBUILD_EXAMPLES=OFF \
        -DBUILD_BENCHMARKS=OFF \
        ../ && \
        make ${MAKE_OPTIONS} && make install && make clean

# colmap
RUN cd /src && git clone -b 3.6-dev.3 https://github.com/colmap/colmap.git
RUN mkdir -p /src/colmap/build
WORKDIR /src/colmap/build
RUN     cmake \
        -DCMAKE_BUILD_TYPE=Release \
        -DTESTS_ENABLED=OFF \
        .. && \
        make ${MAKE_OPTIONS} && make install && make clean


########################################################################################################################
# install kapture env
ADD . /opt/source/kapture
WORKDIR /opt/source/kapture
RUN git submodule update --init --recursive
RUN pip3 install -r requirements.txt
RUN python3 setup.py install
