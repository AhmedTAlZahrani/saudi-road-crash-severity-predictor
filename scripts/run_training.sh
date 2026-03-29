#!/bin/bash
set -e

echo "Activating virtual environment..."
source .venv/Scripts/activate

echo "Starting model training..."
python -m src.train

echo "Training complete."
