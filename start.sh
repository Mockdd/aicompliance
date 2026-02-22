#!/bin/bash
cd /workspaces/aicompliance

# .env가 최상위에 있으면 web/으로 복사
if [ -f ".env" ] && [ ! -f "web/.env" ]; then
    cp .env web/.env
    echo "✅ .env 파일을 web/ 폴더로 복사했습니다."
fi

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
