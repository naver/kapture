FROM ubuntu:20.04
MAINTAINER naverlabs "kapture@naverlabs.com"

# set local (see more on https://leimao.github.io/blog/Docker-Locale/)
ENV     LANG C.UTF-8
ENV     LC_ALL C.UTF-8

# Set correct environment variables.
ENV DEBIAN_FRONTEND noninteractive
ARG MAKE_OPTIONS="-j8"
ARG SOURCE_PREFIX="/opt/src"

# Get dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git wget curl pandoc asciidoctor \
    python3 python3-pip python3-dev && \
    rm -rf /var/lib/apt/lists/*

# make sure pip 3 is >= 20.0 to enable use-feature=2020-resolver
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install --upgrade setuptools wheel twine

########################################################################################################################
# install kapture env
ADD . ${SOURCE_PREFIX}/kapture
WORKDIR ${SOURCE_PREFIX}/kapture
RUN git submodule update --init --recursive
RUN python3 -m pip install -r requirements.txt
RUN python3 -m pip install .

### FINALIZE ###################################################################
# save space: purge apt-get
RUN     rm -rf /var/lib/apt/lists/*
USER  root
WORKDIR ${SOURCE_PREFIX}/
