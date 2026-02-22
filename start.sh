#!/bin/bash
cd /workspaces/aicompliance

echo "📦 패키지 설치 중..."
pip install -r web/requirements.txt -q
pip install uvicorn streamlit -q

echo "🚀 FastAPI 서버 시작..."
cd /workspaces/aicompliance/web
uvicorn main:app --host 0.0.0.0 --port 8000 &

echo "⏳ FastAPI 준비 중..."
sleep 3

echo "🎨 Streamlit 시작..."
cd /workspaces/aicompliance
python -m streamlit run web/app.py --server.port 8501 --server.address 0.0.0.0
