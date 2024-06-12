FROM ubuntu:24.04

RUN apt-get update && apt-get install -y curl

COPY ./ollama_install.sh .
COPY ./ollama_serve.sh .

RUN curl -L https://ollama.com/download/ollama-linux-amd64 -o /usr/bin/ollama
RUN chmod +x /usr/bin/ollama
RUN useradd -r -s /bin/false -m -d /usr/share/ollama ollama
RUN sh ./ollama_serve.sh
