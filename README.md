# GPUMAP

주요 구성
- Docker 기반 개발 환경: `rapidsai/base:25.12-cuda12-py3.11` 위에 conda 환경을 생성
- CMake + CUDA C++: `cpp/`에 CUDA 소스, `python/`에 PyBind11 모듈 빌드
- Python 패키지: `python/gpumap` 패키지와 `_backend` CUDA 확장

디렉토리 구조
- `Dockerfile`: CUDA + conda 환경 포함 컨테이너 설정
- `env.yml`: conda 환경 정의 (CUDA 12.9, CMake, PyBind11 등)
- `CMakeLists.txt`: 최상위 CMake (C++/CUDA + Python 빌드)
- `cpp/`: CUDA C++ 소스/헤더
- `python/`: PyBind11 바인딩 및 Python API
- `scripts/patch_compile_commands.py`: nvcc compile_commands.json을 clangd에 맞게 패치

요구 사항
- NVIDIA GPU + 최신 NVIDIA Driver (host에 설치)
- Docker (NVIDIA Container Toolkit 권장)
- (컨테이너 내부) conda 환경은 `env.yml`로 자동 구성

빠른 시작
1) Docker 이미지 빌드
```
docker build -t gpumap:dev .
```

2) 컨테이너 실행
```
docker run --rm -it --gpus all -v "$PWD":/workspace -w /workspace gpumap:dev /bin/bash
```

3) (필요 시) conda 환경 활성화
```
conda activate gpumap
```

CUDA C++ / Python 확장 빌드

옵션 A) Python 패키지로 빌드/설치 (scikit-build-core)
```
python -m pip install -U pip wheel setuptools
python -m pip install -e . -v
```

옵션 B) CMake 빌드
```
cmake -S . -B build -GNinja -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build -j
python -m pip install -e . -v
```



주의 사항
- `python/gpumap/api.py`는 CuPy를 사용합니다. conda 환경에 CuPy가 없다면 설치가 필요합니다.
- `env.yml`의 `cuda-version`은 user-space 라이브러리만 포함합니다. 실제 드라이버는 host에 설치되어 있어야 합니다.
- clangd 사용 시 `scripts/patch_compile_commands.py`로 `build/compile_commands.json`을 패치할 수 있습니다.
