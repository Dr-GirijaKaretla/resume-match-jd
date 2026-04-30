#!/bin/bash
port=${PORT:-8501}
streamlit run app.py --server.port=$port --server.address=0.0.0.0
chmod +x start.sh
