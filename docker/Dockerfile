# Define the versions for the base image
ARG PYTORCH="1.13.1"
ARG CUDA="11.6"
ARG CUDNN="8"

# Select the base image
FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

# Copy the requirements folder
COPY requirements requirements

# Define environment variables
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX 8.0 8.6"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV FORCE_CUDA=1
ENV DEBIAN_FRONTEND=noninteractive


# Install OS depencencies
RUN apt update && apt install -y \
        wget zip unzip build-essential python3-opencv\
        ffmpeg libsm6 libxext6  \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*


# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements/basic.txt
RUN pip install -r requirements/dl.txt
RUN pip install -r requirements/others.txt

# Create a user to match the host user
ARG USER_ID
ARG GROUP_ID

RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user
USER user


WORKDIR /