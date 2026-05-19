#!/usr/bin/env bash
# exit on error
set -o errexit

# Install dependencies
pip install -r requirements.txt

# Generate data and train the model
python data_generator.py
python main.py
