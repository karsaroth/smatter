FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
  curl \
  libmpv-dev \
  python3-virtualenv \
  python3-pip \
  python-is-python3 \
  yt-dlp \
  && rm -rf /var/lib/apt/lists/*

# TODO: build phantomjs instead? it's not happy
# RUN curl -L https://bitbucket.org/ariya/phantomjs/downloads/phantomjs-2.1.1-linux-x86_64.tar.bz2 | tar -xj  && mv /phantomjs-2.1.1-linux-x86_64/bin/phantomjs /usr/bin/  && rm /phantomjs-2.1.1-linux-x86_64 -rf

COPY . /app
RUN python -m pip install -r /app/requirements.txt


WORKDIR /app
#CMD python3 /app/main.py