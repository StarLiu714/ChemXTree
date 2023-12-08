#!/usr/bin/env bash

# install requirements
pip install torch==1.8.1
pip install category-encoders==2.5.*
pip install xgboost==2.0.*
pip install chemprop==1.6.*
pip install numpy==1.17.2
pip install pandas==1.1.5
pip install scikit-learn==1.0.0
pip install pytorch-lightning==1.8.*
pip install omegaconf==2.0.1
pip install torchmetrics==0.11.*
pip install tensorboard==2.2.0
pip install protobuf==3.20.*
pip install pytorch-tabnet==4.0.*
pip install PyYAML==5.4.1
# pip install importlib-metadata==1.0.*  # Uncomment if needed
pip install matplotlib==3.1.*
pip install ipywidgets
pip install einops==0.6.*
pip install rich==10.2.2
pip install rdkit-pypi==2022.9.*
pip install lightgbm==4.0.*

# Handle Python version-specific dependencies
if [ "$(python -c 'import sys; print(sys.version_info.major)')" == "3" ] && [ "$(python -c 'import sys; print(sys.version_info.minor)')" == "6" ]; then
    pip install dataclasses
fi
