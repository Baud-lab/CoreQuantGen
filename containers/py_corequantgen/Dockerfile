FROM python:3.10.14-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# System libraries useful for numpy/scipy and building old Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    procps \
    coreutils \
    && rm -rf /var/lib/apt/lists/*

# Keep packaging tools at versions that behave well with older packages
RUN python -m pip install --no-cache-dir \
    pip==23.3.2 \
    setuptools==68.2.2 \
    wheel==0.41.3

# Core scientific stack
RUN python -m pip install --no-cache-dir \
    numpy==1.21.6 \
    scipy==1.7.3 \
    pandas==1.3.5 \
    h5py==3.6.0

# PyTorch
RUN python -m pip install --no-cache-dir \
    torch==1.13.1

# LIMIX core
RUN python -m pip install --no-cache-dir \
    limix-core==1.0.2

# limix-lmm has several declared dependencies that are not needed
# for "from limix_lmm import LMM", and allowing pip to resolve them
# today can introduce newer incompatible packages.
RUN python -m pip install --no-cache-dir --no-deps \
    limix-lmm==0.1.2

# Check installation during container build
RUN python - <<'PY'
import sys
import numpy
import scipy
import pandas
import h5py
import torch

print("Python:", sys.version)
print("numpy:", numpy.__version__)
print("scipy:", scipy.__version__)
print("pandas:", pandas.__version__)
print("h5py:", h5py.__version__)
print("torch:", torch.__version__)

from limix_core.gp import GP2KronSum
from limix_core.covar import FreeFormCov
from limix_core.covar import Covariance
from limix_core.hcache import cached
from limix_lmm import LMM

# Explicitly test the old SciPy API used by limix-lmm
assert scipy.ones((2, 2)).shape == (2, 2)
assert scipy.zeros((2, 2)).shape == (2, 2)
assert scipy.dot([1, 2], [1, 2]) == 5

print("limix-core imports: OK")
print("limix-lmm LMM import: OK")
print("legacy scipy API: OK")
print("Container environment: OK")
PY
