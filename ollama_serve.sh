#!/bin/sh

model_name=$1
model_dir=$2
mkdir -p $2

terminate_ollama() {
    pid=$(lsof -i:11434 -t)
    echo terminating ollama with pid $pid
    kill $pid
}

terminate_ollama
OLLAMA_MODELS=$2 ollama serve &

sleep 6

ollama pull $model_name
terminate_ollama
find $2
