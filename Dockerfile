FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel
LABEL authors="Project Neura"

RUN apt install git
RUN pip install "git+https://github.com/MIC-DKFZ/nnUNet.git"
RUN git clone https://github.com/ProjectNeura/SegSTRONGC.git /workspace/code
ARG nnUNet_raw=/workspace/data/nnUNet_raw
ARG nnUNet_preprocessed=/workspace/data/nnUNet_preprocessed
ARG nnUNet_results=/workspace/data/nnUNet_weights
WORKDIR /workspace/code