FROM ubuntu:22.04
RUN apt-get update && apt-get install -y --no-install-recommends make pkg-config libmpc3 libmpfr6 libgmp10 && rm -rf /var/lib/apt/lists/*
