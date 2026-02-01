# syntax=docker/dockerfile:1
FROM rapidsai/base:25.12-cuda12-py3.11

USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda-toolkit-12-9 \
    build-essential \
    git \
    pkg-config \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# conda 환경 생성 시 bash -lc가 필요할 때가 많아서 SHELL 변경
SHELL ["/bin/bash", "-lc"]

WORKDIR /workspace

# 1) env.yml을 이미지에 포함
COPY env.yml /tmp/env.yml

# 2) env 생성 (env.yml의 name: proumap 라고 가정)
RUN conda env create -f /tmp/env.yml && conda clean -afy

# 3) 기본 인터프리터를 proumap env로 고정
# env 이름을 env.yml의 name과 반드시 맞추세요.
ENV CONDA_DEFAULT_ENV=proumap
ENV PATH=/opt/conda/envs/proumap/bin:$PATH

# (선택) 프로젝트 소스까지 이미지에 포함해서 editable 설치까지 하고 싶으면 아래 사용
# COPY . /workspace
# RUN python -m pip install -U pip wheel setuptools
# RUN python -m pip install -e . -v

CMD ["/bin/bash"]
