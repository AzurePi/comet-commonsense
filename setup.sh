#!/bin/bash


python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

bash ./scripts/setup/get_atomic_data.sh &
bash ./scripts/setup/get_conceptnet_data.sh &
bash ./scripts/setup/get_model_files.sh &
python -m spacy download en_core_web_sm

wait