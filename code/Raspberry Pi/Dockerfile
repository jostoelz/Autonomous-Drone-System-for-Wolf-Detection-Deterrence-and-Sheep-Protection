# base: ros noetic (ubuntu 20.04 / py3.8)
# acts as bridge between legacy kinetic pi and new hailo chip
FROM ros:noetic-ros-base

ENV DEBIAN_FRONTEND=noninteractive

# install system tools + build deps
RUN apt-get update && apt-get install -y \
    nano \
    wget \
    git \
    python3-pip \
    python3-dev \
    python3-opencv \
    libgl1-mesa-glx \
    libgirepository1.0-dev \
    && rm -rf /var/lib/apt/lists/*

# install hailo system driver (v4.20.0)
COPY hailort_4.20.0_arm64.deb /tmp/hailort.deb
RUN dpkg --unpack /tmp/hailort.deb \
    && rm /var/lib/dpkg/info/hailort.postinst \
    && dpkg --configure hailort \
    && rm /tmp/hailort.deb \
    && ldconfig

# install python deps
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt
RUN pip3 install Pillow

# install hailo python driver (need specific cp38 wheel for ubuntu 20.04)
COPY hailort-4.20.0-cp38-cp38-linux_aarch64.whl /tmp/
RUN pip3 install /tmp/hailort-4.20.0-cp38-cp38-linux_aarch64.whl && rm /tmp/hailort-4.20.0-cp38-cp38-linux_aarch64.whl

# set workdir
WORKDIR /hailo_examples

# auto-source ros environment on shell start
RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc

CMD ["/bin/bash"]







