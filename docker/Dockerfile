# 법률 서비스 통합 앱 (law_page.py) Docker 이미지
FROM python:3.11-slim-bookworm

WORKDIR /app

# 시스템 패키지 설치
# - libgomp1: FAISS 런타임 필요
# - default-jdk: KoNLPy(Okt) 사용 시 필요 (선택사항)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python 환경 설정
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# 의존성 설치
COPY docker/requirements.txt ./
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install --no-cache-dir -r requirements.txt

# 앱 소스 복사
COPY scripts/law_page.py /app/law_page.py
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# 포트 노출
EXPOSE 8501

# 엔트리포인트
ENTRYPOINT ["/entrypoint.sh"]
