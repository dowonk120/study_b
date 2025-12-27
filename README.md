# 법률 문서 요약 및 상담 챗봇 프로젝트

ML + DL + LLM 융합 법률 서비스 플랫폼

---

## 목차

1. [프로젝트 개요](#프로젝트-개요)
2. [주요 기능](#주요-기능)
3. [기술 스택](#기술-스택)
4. [프로젝트 구조](#프로젝트-구조)
5. [설치 및 실행](#설치-및-실행)
6. [Docker 실행](#docker-실행)
7. [코드 최적화 내역](#코드-최적화-내역)
8. [전체 코드 설명](#전체-코드-설명)
9. [데이터 출처](#데이터-출처)

---

## 프로젝트 개요

### 문제 인식
- 일반 사용자들은 계약서나 판례 같은 법률 문서를 읽고 이해하기 어려움
- 법률 용어는 어렵고, 문서 길이도 길어 핵심 내용 파악이 어려움

### 목표
- 법률 문서를 자동 분류 → 요약 → 질의응답 형태로 처리
- 사용자가 쉽게 이해하고 활용할 수 있는 챗봇 형태의 서비스 구현

---

## 주요 기능

| 기능 | 설명 |
|------|------|
| 사건 검색 시뮬레이션 | 사건 내용 입력 시 유사 판례 검색 및 승소 확률 예측 |
| RAG 기반 상담 챗봇 | 판례 데이터 기반 법률 질의응답 |
| LLM 키워드 추출 | GPT-4o-mini를 활용한 사건 도메인/쟁점 자동 분류 |
| PDF 업로드 지원 | 사용자 문서 업로드 후 RAG 기반 상담 가능 |
| 참조조문 설명 | LLM을 통한 법률 조문 쉬운 설명 제공 |

---

## 기술 스택

| 범주 | 사용 기술 |
|------|----------|
| Frontend | Streamlit |
| LLM | OpenAI GPT-4o-mini |
| Vector DB | FAISS |
| Embedding | OpenAI Embeddings |
| Framework | LangChain |
| NLP (폴백) | KoNLPy (Okt) |
| 데이터 처리 | Pandas |

---

## 프로젝트 구조

```
study_b/
├── README.md                # 프로젝트 문서
├── .gitignore               # Git 제외 파일
├── .env.example             # 환경변수 예시
│
├── docker/                  # Docker 관련 파일
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── requirements.txt
│   └── entrypoint.sh
│
├── scripts/                 # 소스 코드
│   ├── law_page.py          # 메인 앱 (통합 버전)
│   ├── law_str_llm.py       # 검색 전용 앱
│   ├── law_str.py           # 기본 검색 앱
│   ├── my_rag_chatbot.py    # RAG 챗봇
│   └── preprocessing.py     # 데이터 전처리
│
├── data/                    # 데이터 파일
│   ├── result_1_merged.zip  # 판례 데이터 1
│   ├── result_2_merged.zip  # 판례 데이터 2
│   ├── result_3_merged.zip  # 판례 데이터 3
│   ├── result.zip           # 원본 데이터
│   └── code_map.csv         # 사건코드 매핑
│
└── mycache/                 # 자동 생성 (Git 제외)
    ├── files/               # 업로드 파일
    ├── embedding/           # 임베딩 캐시
    └── vectorstore/         # FAISS 인덱스
```

---

## 설치 및 실행 (로컬)

### 1. 의존성 설치

```bash
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

브라우저에서 `http://localhost:8501` 접속

### 4. 옵션: KoNLPy 설치 (폴백용)

```bash
# Java 설치 필요
sudo apt-get install openjdk-11-jdk
pip install konlpy JPype1
```

> KoNLPy가 없어도 앱은 정상 동작합니다 (simple 폴백 사용)

---

## Docker 실행

### 1. 환경 변수 설정

```bash
cp .env.example .env
# .env 파일에서 OPENAI_API_KEY 설정
```

### 2. Docker Compose로 실행

```bash
cd docker
docker-compose up -d
```

### 3. 접속

브라우저에서 `http://localhost:8501` 접속

### 4. 로그 확인

```bash
docker-compose logs -f
```

### 5. 종료

```bash
docker-compose down
```

### Docker 구성 요약

| 항목 | 값 |
|------|-----|
| 이미지 | law-service:latest |
| 포트 | 8501 |
| 데이터 | ./data → /app/data (읽기전용) |
| 캐시 | ./mycache → /app/mycache |

---

## 코드 최적화 내역

기존 코드들의 장점을 통합하여 최적화했습니다.

### 1. API Key 관리 통합

**기존 (law_page.py)**
```python
def get_api_key():
    key = os.getenv("OPENAI_API_KEY")
    if key:
        return key
    # session_state 미사용
```

**최적화 후**
```python
def get_api_key():
    key = os.getenv("OPENAI_API_KEY") or st.session_state.get("OPENAI_API_KEY")
    if not key:
        with st.sidebar:
            k = st.text_input("OPENAI_API_KEY 입력", type="password")
            if k:
                st.session_state["OPENAI_API_KEY"] = k
                key = k
    return key
```

> env와 session_state 둘 다 확인하여 유연성 향상

---

### 2. LLM 예외 처리 및 폴백

**기존 (law_page.py)**
```python
# 예외 처리 없음 - LLM 실패시 앱 크래시
resp = client.chat.completions.create(...)
return json.loads(resp.choices[0].message.content)
```

**최적화 후**
```python
def llm_classify_intent(text, api_key):
    try:
        resp = client.chat.completions.create(...)
        try:
            return json.loads(txt)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", txt, re.S)
            return json.loads(m.group(0)) if m else {}
    except Exception as e:
        st.info(f"LLM 분석 실패 (자동 폴백): {e}")
        return {}
```

> API 오류, JSON 파싱 오류 모두 안전하게 처리

---

### 3. KoNLPy 폴백 로직

**기존 (law_page.py)**
```python
# KoNLPy 실패시 에러
from konlpy.tag import Okt
okt = Okt()  # Java 없으면 크래시
```

**최적화 후**
```python
def konlpy_extract_keywords_or_fallback(situation):
    def simple_keywords(text):
        priority = ['마약', '성폭행', '음주운전', '사기', ...]
        found = [w for w in priority if w in text]
        return found, '|'.join(found) if found else ".*"

    try:
        from konlpy.tag import Okt
        okt = Okt()
        nouns = okt.nouns(situation)
        ...
    except Exception as e:
        st.info(f"KoNLPy 불가 (자동 폴백): {e}")
        return simple_keywords(situation)
```

> Java 미설치 환경에서도 기본 키워드 매칭으로 동작

---

### 4. st.set_page_config 중복 제거

**기존 (law_page.py)**
```python
st.set_page_config(...)  # 최상단
...
with tab1:
    st.set_page_config(...)  # 중복 호출 -> 에러!
```

**최적화 후**
```python
# 최상단에서 한 번만 호출
st.set_page_config(page_title="법률 서비스 통합", layout="wide")
```

---

### 5. Import 통합 및 변수명 충돌 해결

**기존**
```python
# 탭1과 탭2에서 각각 import
with tab1:
    import pandas as pd
    df = load_data()  # 변수명 충돌

with tab2:
    import pandas as pd
    df = load_data()  # 같은 이름!
```

**최적화 후**
```python
# 상단에서 한 번만 import
import pandas as pd
...
with tab1:
    df_search = load_data()  # 명확한 이름

with tab2:
    df_chat = load_data()    # 명확한 이름
```

---

### 6. RAG 프롬프트 개선

**기존 (law_page.py)**
```python
"Format the response in Markdown..."
"답변후에 혹시 이 주제에 대해 더 궁금한 것이 있으신가요?"
```

**최적화 후 (chatbot.py에서 가져옴)**
```python
"""
⛔ 반드시 유의하세요: context에 없는 내용은 절대 생성하지 마세요.
...
4️⃣ 판례나 사건번호가 있다면, 반드시 아래 형식을 따르세요:
   ### ✅ 판례 예시
   - 사건번호: [사건번호]
   - 판결 내용 요약: [간략한 설명]
...
9️⃣ 출처가 context 기반인지, 모델의 사전 지식인지 반드시 명시하세요.
"""
```

> 더 상세한 지침으로 답변 품질 향상

---

### 7. Document 생성 개선

**기존**
```python
docs = [Document(page_content=str(row.to_dict())) for _, row in df.iterrows()]
```

**최적화 후**
```python
for i, row in sel_df.iterrows():
    row_hash = hashlib.md5(str(row_dict).encode()).hexdigest()

    # 참조조문별로 Document 분리
    for 조문 in 참조조문.split(','):
        content = f"""참조조문: {조문}
배상책임: {row.get('배상책임', '')}
주제: {row.get('주제', '')}
..."""
        docs.append(Document(
            page_content=content,
            metadata={"row_id": row_hash, "조문": 조문}
        ))
```

> 조문별 분리 + 메타데이터 추가로 검색 정확도 향상

---

### 8. Stopwords 자동 생성

**기존**
```python
# 수동으로 불용어 정의
stopwords = {"어제", "먹고", "쳤어", ...}
```

**최적화 후**
```python
@st.cache_data
def build_stopwords(df, top_n=100):
    all_words = []
    target_cols = ["판시사항", "판결요지", ...]
    for col in target_cols:
        texts = df[col].dropna().astype(str)
        for text in texts:
            all_words.extend(re.findall(r'[가-힣a-zA-Z0-9]{2,}', text))

    common_words = [w for w, _ in Counter(all_words).most_common(top_n)]
    extra_stopwords = {"어제", "먹고", ...}
    return set(common_words).union(extra_stopwords)
```

> 데이터 기반 자동 생성 + 수동 추가 병합

---

## 전체 코드 설명

### 코드 구조

```
law_page.py (772줄)
│
├── [1-45] 공통 Import
│   └── 모든 필요한 라이브러리를 상단에서 한 번에 import
│
├── [46-60] 환경 설정
│   ├── load_dotenv() - .env 파일 로드
│   ├── st.set_page_config() - 페이지 설정 (한 번만)
│   └── 캐시 폴더 생성
│
├── [62-90] 공통 유틸리티 함수
│   ├── get_api_key() - API 키 관리
│   ├── llm_available() - LLM 사용 가능 여부
│   └── get_openai_client() - OpenAI 클라이언트 생성
│
├── [92-158] 데이터 로드 함수
│   ├── load_data() - ZIP에서 CSV 로드 및 병합
│   ├── load_code_map() - 사건코드 매핑 테이블
│   └── build_stopwords() - 불용어 자동 생성
│
├── [160-291] 탭1 전용 함수
│   ├── llm_classify_intent() - LLM 의도 분류
│   ├── konlpy_extract_keywords_or_fallback() - 키워드 추출
│   ├── explain_law_article() - 조문 설명 생성
│   ├── extract_abbr() - 사건번호 약어 추출
│   └── infer_domain_from_codes() - 도메인 추론
│
├── [293-445] 탭2 전용 함수
│   ├── create_retriever_from_df() - DataFrame → Retriever
│   ├── create_retriever_from_pdf() - PDF → Retriever
│   └── create_rag_chain() - RAG 체인 생성
│
├── [447-462] 탭 스타일링 및 구성
│
├── [464-682] 탭1: 사건 검색 시뮬레이션
│   ├── 입력 폼 (입장, 합의여부, 피해금액, 사건내용)
│   ├── 검색 로직 (LLM → KoNLPy → simple 폴백)
│   ├── 승률 계산 및 표시
│   ├── 페이지네이션 결과 표시
│   └── 통계 차트
│
├── [684-766] 탭2: RAG 기반 상담 챗봇
│   ├── PDF 업로드 옵션
│   ├── Retriever/Chain 생성
│   ├── 채팅 인터페이스
│   └── 질문 확장 및 응답 스트리밍
│
└── [768-772] 푸터
```

### 주요 함수 설명

#### `get_api_key()` (Line 66-76)
- OpenAI API 키를 환경변수 또는 세션에서 가져옴
- 없으면 사이드바에서 입력 받음

#### `load_data()` (Line 96-130)
- 3개의 ZIP 파일에서 CSV 추출 및 병합
- 불필요한 컬럼 제거, 결측치 처리
- `@st.cache_data`로 캐싱하여 성능 최적화

#### `llm_classify_intent()` (Line 164-215)
- GPT-4o-mini로 사건 도메인(형사/민사/가사 등) 자동 분류
- 검색 키워드 8~15개 추출
- JSON 파싱 실패시 정규식으로 재시도

#### `konlpy_extract_keywords_or_fallback()` (Line 218-248)
- KoNLPy의 Okt로 명사 추출
- Java 미설치시 simple_keywords() 폴백
- 우선순위 키워드 매칭

#### `create_retriever_from_df()` (Line 297-348)
- DataFrame을 FAISS 벡터 DB로 변환
- 해시 기반 캐싱으로 재생성 방지
- 조문별 Document 분리로 검색 정확도 향상

#### `create_rag_chain()` (Line 387-444)
- LangChain의 RunnableMap으로 RAG 파이프라인 구성
- 개선된 프롬프트로 답변 품질 향상
- 스트리밍 응답 지원

---

## 데이터 출처

- 법원 사이트: https://portal.scourt.go.kr/pgp/index.on?m=PGP1011M01&l=N&c=900
- 공공데이터 API: https://www.data.go.kr/data/15057123/openapi.do

---

## 주의사항

본 서비스는 참고용이며, 실제 법률 자문이 아닙니다.
정확한 법률 상담은 전문 변호사에게 문의하세요.
