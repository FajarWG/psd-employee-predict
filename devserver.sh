#!/bin/sh
source .venv/bin/activate
# python -m flask --app main run -p $PORT --debug
streamlit run main.py --server.port $PORT