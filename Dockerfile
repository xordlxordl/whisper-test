# Using NVIDIA PyTorch with a base image
FROM nvcr.io/nvidia/pytorch:23.12-py3

# Install necessary additional packages
# Installing Python packages using pip
RUN pip install transformers accelerate datasets

# Installing system packages using apt-get
RUN apt-get update && apt-get install -y git-lfs ffmpeg
RUN apt-get install libportaudio2

WORKDIR /workspace
COPY . /workspace

RUN mkdir -p /work/model/whisper-large-v3
RUN pip3 install huggingface_hub
RUN pip3 install huggingface_hub --break-system-packages
RUN pip3 install sounddevice --break-system-packages

#RUN python3 download_model.py
#RUN python3 mic_stt.py
