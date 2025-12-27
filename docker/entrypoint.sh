#!/usr/bin/env bash
set -e

# 캐시 디렉토리 생성
mkdir -p /app/mycache/files
mkdir -p /app/mycache/embedding
mkdir -p /app/mycache/vectorstore

echo "=== Law Service Starting ==="
echo "Data directory: /app/data"
echo "Cache directory: /app/mycache"

# Streamlit 실행
exec python -m streamlit run /app/law_page.py \
    --server.address=0.0.0.0 \
    --server.port=8501 \
    --server.headless=true \
    --browser.gatherUsageStats=false
