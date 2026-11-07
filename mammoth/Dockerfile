# baselabs/mammoth-1 on arm64 (DGX Spark): the model card's setup.sh as an image. The weights stay on the node's NVMe.
FROM nvidia/cuda:13.0.1-base-ubuntu24.04
ARG MODEL_REV=61cec44c281c693992e3d6ebeb44dd3402dfa186
ARG ONEJEV=81ce62f1597c91e46767d4d02d6ac2e18534fe94
ENV DEBIAN_FRONTEND=noninteractive UV_LINK_MODE=copy PATH=/usr/local/cuda/bin:/opt/venv/bin:$PATH CUDA_HOME=/usr/local/cuda \
    UV_PYTHON_INSTALL_DIR=/opt/python PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates gcc g++ libc6-dev cuda-nvcc-13-0 cuda-cudart-dev-13-0 cuda-cccl-13-0 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /usr/local/bin/uv
RUN uv venv --python 3.12 --seed /opt/venv \
    && uv pip install --python /opt/venv/bin/python --no-cache --index-url https://download.pytorch.org/whl/cu130 torch==2.13.0 torchvision
WORKDIR /opt
RUN mkdir -p /opt/card \
    && for f in serve.patch constraints.txt readout.json engine.py; do \
         curl -fsSL "https://huggingface.co/baselabs/mammoth-1/raw/$MODEL_REV/$f" -o "/opt/card/$f"; done \
    && git clone -q https://github.com/OmniJev/OneJev.git /opt/OneJev \
    && git -C /opt/OneJev checkout -q -f "$ONEJEV" \
    && git -C /opt/OneJev apply /opt/card/serve.patch
COPY parallel.py onepass.py /opt/card/
RUN python /opt/card/parallel.py /opt/OneJev/qev/server.py \
    && python /opt/card/onepass.py /opt/OneJev/qev/engine.py
RUN python -c "import torch, torchvision; print(f'torch=={torch.__version__}\ntorchvision=={torchvision.__version__}')" > /opt/image-torch.txt \
    && uv pip install --python /opt/venv/bin/python --no-cache -c /opt/card/constraints.txt -c /opt/image-torch.txt \
         -e "/opt/OneJev[train]" huggingface_hub hf_xet brotli flash-linear-attention httpx \
    && uv pip install --python /opt/venv/bin/python --no-cache --no-build-isolation -c /opt/card/constraints.txt -c /opt/image-torch.txt causal-conv1d \
    && python -c "import torch, qev.server; print('server imports on', torch.__version__, 'cuda build', torch.version.cuda)"
ENV QEV_PARALLEL=4
EXPOSE 8000
ENTRYPOINT ["qev", "serve"]
