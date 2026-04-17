FROM ubuntu:24.04
LABEL org.opencontainers.image.authors="naverlabs <kapture@naverlabs.com>"

# set local (see more on https://leimao.github.io/blog/Docker-Locale/)
ENV     LANG=C.UTF-8
ENV     LC_ALL=C.UTF-8

# Set correct environment variables.
ENV DEBIAN_FRONTEND=noninteractive
ARG MAKE_OPTIONS="-j8"
ARG SOURCE_PREFIX="/opt/src"

# Get dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    nano git wget curl pandoc asciidoctor \
    python3 python3-pip python3-dev python3-venv && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip setuptools wheel

########################################################################################################################
# install kapture env
COPY . ${SOURCE_PREFIX}/kapture
WORKDIR ${SOURCE_PREFIX}/kapture
RUN git submodule update --init --recursive
RUN python3 -m pip install .

#### FINALIZE ###################################################################
## save space: purge apt-get
RUN     rm -rf /var/lib/apt/lists/*
USER  root
WORKDIR ${SOURCE_PREFIX}/