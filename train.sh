#!/bin/bash

# --- Configuração ---
MEMORIA_MINIMA_LIVRE_MB=16384
INTERVALO_SEGUNDOS=150
GPU_DISPONIVEL=-1

# --- Checa se nvidia-smi está disponível ---
if ! command -v nvidia-smi &> /dev/null; then
    echo "Nenhuma GPU detectada (nvidia-smi não encontrado). Usando CPU..."
    USE_GPU=false
else
    echo "Aguardando por uma GPU com pelo menos ${MEMORIA_MINIMA_LIVRE_MB}MB de memória livre..."
    USE_GPU=true

    while [[ $GPU_DISPONIVEL -lt 0 ]]; do
        while IFS=',' read -r index free_mem; do
            free_mem_clean=$(echo $free_mem | xargs)
            if (( free_mem_clean >= MEMORIA_MINIMA_LIVRE_MB )); then
                echo "GPU de índice ${index} com ${free_mem_clean}MB livres encontrada."
                GPU_DISPONIVEL=$index
                break
            fi
        done <<< "$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)"

        if [[ $GPU_DISPONIVEL -lt 0 ]]; then
            START_WAIT_TIME=$(date +"%Y-%m-%d %H:%M:%S")
            echo "${START_WAIT_TIME}: Nenhuma GPU com espaço suficiente. Aguardando ${INTERVALO_SEGUNDOS} segundos..."
            sleep $INTERVALO_SEGUNDOS
        fi
    done
fi

# --- Execução ---
if [ "$USE_GPU" = true ]; then
    echo "Configurando para usar a GPU: ${GPU_DISPONIVEL}"
    export CUDA_VISIBLE_DEVICES=$GPU_DISPONIVEL
else
    echo "Executando em CPU (sem variável CUDA_VISIBLE_DEVICES)."
fi

python ./src/main.py --experiment_type "atomic"
python ./src/main.py --experiment_type "conceptnet"