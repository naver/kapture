#FROM ubuntu:18.04
FROM nvidia/cudagl:10.0-devel-ubuntu18.04
MAINTAINER naverlabs "kapture@naverlabs.com"

# Set correct environment variables.
ENV LC_ALL C
ENV DEBIAN_FRONTEND noninteractive
ARG MAKE_OPTIONS="-j8"
ARG SOURCE_PREFIX="/opt/src"

# Get dependencies
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    git wget curl \
    python3.6 python3-pip \
    pandoc asciidoctor \
  && rm -rf /var/lib/apt/lists/*

# make sure pip 3 is >= 20.0 to enable use-feature=2020-resolver
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install --upgrade setuptools wheel twine

########################################################################################################################
# install kapture env
ADD . ${SOURCE_PREFIX}/kapture
WORKDIR ${SOURCE_PREFIX}/kapture
RUN git submodule update --init --recursive
RUN python3 -m pip install -r requirements.txt --use-feature=2020-resolver
RUN python3 setup.py install

### FINALIZE ###################################################################
# save space: purge apt-get
RUN     rm -rf /var/lib/apt/lists/*
USER  root
WORKDIR ${SOURCE_PREFIX}/
