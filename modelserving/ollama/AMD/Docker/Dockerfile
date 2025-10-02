FROM docker.io/rocm/dev-ubuntu-24.04

RUN apt-get update --quiet \
    && apt-get dist-upgrade --yes --quiet \
    && apt-get install --yes --quiet software-properties-common \
    && apt-get install --yes --quiet git wget gcc g++ gcc-10 g++-10 make zlib1g-dev zstd

RUN curl -LO https://ollama.com/download/ollama-linux-amd64.tgz

RUN tar -C /usr -xzf ollama-linux-amd64.tgz

RUN curl -L https://ollama.com/download/ollama-linux-amd64-rocm.tgz -o ollama-linux-amd64-rocm.tgz

RUN tar -C /usr -xzf ollama-linux-amd64-rocm.tgz

RUN useradd -r -s /bin/false -U -m -d /usr/share/ollama ollama

RUN usermod -a -G ollama ollama

RUN rm ollama-linux-amd64.tgz ollama-linux-amd64-rocm.tgz

USER ollama

WORKDIR /

EXPOSE 11434

CMD ["ollama", "serve"]