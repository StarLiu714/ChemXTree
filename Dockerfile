FROM registry.codeocean.com/codeocean/pytorch:2.1.0-cuda11.8.0-mambaforge23.1.0-4-python3.10.12-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN pip install -U --no-cache-dir \
    torch>=1.8.* \
    category-encoders==2.5.* \
    xgboost==2.* \
    chemprop==1.6.* \
    numpy>=1.17.* \
    pandas \
    scikit-learn==1.* \
    pytorch-lightning==1.8.* \
    omegaconf==2.0.1 \
    torchmetrics==0.11.* \
    tensorboard>=2.2.0,!=2.5.0 \
    protobuf==3.20.* \
    pytorch-tabnet==4.0.* \
    PyYAML>=5.2 \
    matplotlib==3.* \
    ipywidgets \
    dataclasses \
    einops==0.6.* \
    rich==10.2.* \
    rdkit-pypi==2022.9.* \
    lightgbm==4.*
