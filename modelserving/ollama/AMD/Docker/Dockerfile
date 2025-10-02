FROM docker.io/rocm/dev-ubuntu-24.04

RUN apt-get update --quiet \
    && apt-get dist-upgrade --yes --quiet 

RUN curl -LO https://ollama.com/download/ollama-linux-amd64.tgz

RUN tar -C /usr -xzf ollama-linux-amd64.tgz

RUN curl -L https://ollama.com/download/ollama-linux-amd64-rocm.tgz -o ollama-linux-amd64-rocm.tgz

RUN tar -C /usr -xzf ollama-linux-amd64-rocm.tgz

RUN rm ollama-linux-amd64.tgz ollama-linux-amd64-rocm.tgz

WORKDIR /

EXPOSE 11434

CMD ["ollama", "serve"]