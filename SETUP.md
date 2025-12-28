# 설치 및 실행 가이드

---

## 목차

1. [프로젝트 구조](#프로젝트-구조)
2. [사전 요구사항](#사전-요구사항)
3. [Docker 실행 (권장)](#docker-실행-권장)
4. [로컬 실행](#로컬-실행)
5. [환경 변수 설정](#환경-변수-설정)
6. [트러블슈팅](#트러블슈팅)
7. [코드 구조 설명](#코드-구조-설명)

---

## 프로젝트 구조

```
study_b/
├── README.md                # 프로젝트 설명
├── SETUP.md                 # 설치 가이드 (현재 문서)
├── .gitignore               # Git 제외 파일
├── .env.example             # 환경변수 템플릿
├── .env                     # 환경변수 (직접 생성 필요)
│
├── docker/                  # Docker 관련 파일
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── requirements.txt
│   └── entrypoint.sh
│
├── scripts/                 # 소스 코드
│   ├── law_page.py          # 메인 앱 (통합 버전)
│   ├── law_str_llm.py       # 검색 전용 앱 (LLM)
│   ├── law_str.py           # 기본 검색 앱
│   ├── my_rag_chatbot.py    # RAG 챗봇 단독
│   └── preprocessing.py     # 데이터 전처리
│
├── data/                    # 데이터 파일
│   ├── result_1_merged.zip  # 판례 데이터 1
│   ├── result_2_merged.zip  # 판례 데이터 2
│   ├── result_3_merged.zip  # 판례 데이터 3
│   ├── result.zip           # 원본 데이터
│   └── code_map.csv         # 사건코드 매핑
│
└── mycache/                 # 자동 생성됨 (Git 제외)
    ├── files/               # 업로드 파일 임시 저장
    ├── embedding/           # 임베딩 캐시
    └── vectorstore/         # FAISS 인덱스 캐시
```

---

## 사전 요구사항

### 필수
- **OpenAI API Key** - [발급 링크](https://platform.openai.com/api-keys)
- **Docker** (Docker 실행 시) 또는 **Python 3.11+** (로컬 실행 시)

### 선택
- **Java 11+** - KoNLPy 사용 시 필요 (없어도 앱 동작)

---

## Docker 실행 (권장)

### 1. 환경 변수 설정

```bash
# .env.example을 복사하여 .env 생성
cp .env.example .env

# .env 파일 편집
nano .env  # 또는 원하는 에디터
```

`.env` 파일 내용:
```
OPENAI_API_KEY=sk-proj-your-actual-api-key-here
```

### 2. Docker Compose 실행

```bash
cd docker
docker compose up -d
```

### 3. 접속

브라우저에서 http://localhost:8501 접속

### 4. 로그 확인

```bash
docker compose logs -f
```

### 5. 종료

```bash
docker compose down
```

### Docker 구성 요약

| 항목 | 값 |
|------|-----|
| 이미지 | law-service:latest |
| 포트 | 8501 |
| 데이터 마운트 | ./data → /app/data (읽기전용) |
| 캐시 마운트 | ./mycache → /app/mycache |

---

## 로컬 실행

### 1. Python 환경 설정

```bash
# 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r docker/requirements.txt
```

### 2. 환경 변수 설정

```bash
cp .env.example .env
# .env 파일에서 OPENAI_API_KEY 설정
```

### 3. 실행

```bash
cd scripts
streamlit run law_page.py
```

브라우저에서 http://localhost:8501 접속

### 4. (선택) KoNLPy 설치

KoNLPy는 한국어 형태소 분석에 사용됩니다. 없어도 앱은 동작합니다.

```bash
# Java 설치 (Ubuntu/Debian)
sudo apt-get install openjdk-11-jdk

# KoNLPy 설치
pip install konlpy JPype1
```

---

## 환경 변수 설정

### .env 파일

```bash
# 필수
OPENAI_API_KEY=sk-proj-your-api-key-here
```

### API 키 우선순위

1. `.env` 파일 → `os.getenv("OPENAI_API_KEY")`
2. 웹 UI 사이드바 입력 → `st.session_state`

---

## 트러블슈팅

### 포트 충돌 (8501)

```bash
# 포트 사용 중인 프로세스 확인
lsof -i :8501

# 기존 컨테이너 중지
docker stop <container_name>

# 또는 docker-compose.yml에서 포트 변경
ports:
  - "8502:8501"  # 외부 8502 → 내부 8501
```

### CRLF 에러 (Windows)

`entrypoint.sh` 파일의 줄바꿈 형식 문제:

```bash
# LF로 변환
sed -i 's/\r$//' docker/entrypoint.sh

# 재빌드
docker compose build --no-cache
docker compose up -d
```

### 임베딩 캐시 문제

```bash
# 캐시 삭제 후 재시작
rm -rf mycache/embedding/*
rm -rf mycache/vectorstore/*
docker compose restart
```

### API 키 오류

```
Error: Invalid API Key
```

1. `.env` 파일의 API 키 확인
2. Docker 재시작: `docker compose down && docker compose up -d`
3. 시스템 환경변수와 충돌 확인: `echo $OPENAI_API_KEY`

---

## 코드 구조 설명

### law_page.py 구조 (메인 앱)

```
law_page.py (약 800줄)
│
├── [1-45] 공통 Import
│   └── 모든 라이브러리를 상단에서 한 번에 import
│
├── [46-90] 환경 설정
│   ├── load_dotenv() - .env 파일 로드
│   ├── st.set_page_config() - 페이지 설정
│   ├── get_base_paths() - Docker/로컬 경로 자동 감지
│   └── 캐시 폴더 생성
│
├── [91-120] 공통 유틸리티 함수
│   ├── get_api_key() - API 키 관리 (env + session_state)
│   ├── llm_available() - LLM 사용 가능 여부
│   └── get_openai_client() - OpenAI 클라이언트 생성
│
├── [121-190] 데이터 로드 함수
│   ├── load_data() - ZIP에서 CSV 로드 및 병합
│   ├── load_code_map() - 사건코드 매핑 테이블
│   └── build_stopwords() - 불용어 자동 생성
│
├── [191-320] 탭1 전용 함수 (사건 검색)
│   ├── llm_classify_intent() - LLM 의도 분류
│   ├── konlpy_extract_keywords_or_fallback() - 키워드 추출
│   ├── explain_law_article() - 조문 설명 생성
│   └── 검색 로직 (LLM → KoNLPy → simple 폴백)
│
├── [321-480] 탭2 전용 함수 (RAG 챗봇)
│   ├── create_retriever_from_df() - DataFrame → FAISS Retriever
│   ├── create_retriever_from_pdf() - PDF → FAISS Retriever
│   └── create_rag_chain() - LangChain RAG 파이프라인
│
├── [481-710] 탭1 UI: 사건 검색 시뮬레이션
│   ├── 입력 폼 (입장, 합의여부, 피해금액, 사건내용)
│   ├── 승률 계산 및 표시
│   ├── 페이지네이션 결과 표시
│   └── 통계 차트
│
├── [711-800] 탭2 UI: RAG 기반 상담 챗봇
│   ├── PDF 업로드 옵션
│   ├── 채팅 인터페이스
│   └── 스트리밍 응답
│
└── 푸터
```

### 주요 함수 설명

#### `get_api_key()`
- OpenAI API 키를 환경변수 또는 세션에서 가져옴
- 없으면 사이드바에서 입력 받음

#### `load_data()`
- 3개의 ZIP 파일에서 CSV 추출 및 병합
- `@st.cache_data`로 캐싱하여 성능 최적화

#### `llm_classify_intent()`
- GPT-4o-mini로 사건 도메인 자동 분류
- 검색 키워드 8~15개 추출
- JSON 파싱 실패시 정규식으로 재시도

#### `create_retriever_from_df()`
- DataFrame을 FAISS 벡터 DB로 변환
- 해시 기반 캐싱으로 재생성 방지

#### `create_rag_chain()`
- LangChain의 RunnableMap으로 RAG 파이프라인 구성
- 스트리밍 응답 지원

---

## 첫 실행 시 주의사항

### 임베딩 생성
- 첫 실행 시 OpenAI Embeddings API를 사용하여 벡터 DB 생성
- **시간 소요**: 데이터 양에 따라 수 분 ~ 수십 분
- **비용 발생**: OpenAI API 사용료

### 캐시 공유
임베딩 비용을 줄이려면 `mycache/` 폴더를 팀원과 공유:
- `mycache/embedding/` - 임베딩 캐시 (~130MB)
- `mycache/vectorstore/` - FAISS 인덱스 (~27MB)

> GitHub 100MB 제한으로 인해 Git LFS 또는 별도 공유 권장
