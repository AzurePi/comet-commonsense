#!/bin/bash

python ./scripts/data/make_atomic_data_loader.py &
python ./scripts/data/make_conceptnet_data_loader.py

wait