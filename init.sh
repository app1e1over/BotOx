#!/bin/sh
sudo apt-get install python3.11
curl -sSL https://install.python-poetry.org | python -
poetry install
sudo poetry run uvicorn app.main:app  --host=0.0.0.0 --port=80 