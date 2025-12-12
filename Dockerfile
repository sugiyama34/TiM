FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ARG PYTHON_VERSION=3.10

ENV DEBIAN_FRONTEND="noninteractive"
ENV LC_ALL="C.UTF-8"
ENV LANG="C.UTF-8"

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} \
    python3-pip \
    python-is-python3 \
    git \
    wget \
    ca-certificates \
 && apt-get -y clean \
 && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 && \
    update-alternatives --set python3 /usr/bin/python${PYTHON_VERSION}
RUN pip install --upgrade pip

# TiM Environment

RUN wget -nv https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    /opt/conda/bin/conda clean -ya
ENV PATH="/opt/conda/bin:$PATH"

RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda create -y -n tim_env python=3.10 && \
    conda run -n tim_env pip install -U pip && \
    conda run -n tim_env pip install psutil && \
    conda run -n tim_env pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118 && \
    conda run -n tim_env pip install --no-cache-dir --no-build-isolation flash-attn 2> error_message && \
    conda run -n tim_env pip install -r requirements.txt && \
    conda run -n tim_env pip install -e .

CMD /bin/bash
CMD ["conda", "activate", "tim_env"]

