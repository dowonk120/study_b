"""
법률 서비스 통합 페이지 (최적화 버전)
- 탭1: 사건 검색 시뮬레이션 (LLM + KoNLPy 폴백)
- 탭2: RAG 기반 법률 상담 챗봇

최적화 내용:
1. API Key 관리 통합 (env + session_state)
2. LLM/KoNLPy 예외 처리 및 폴백 로직
3. 중복 import 제거 및 상단 통합
4. st.set_page_config 중복 호출 제거
5. 변수명 충돌 해결
6. RAG 프롬프트 개선 (출처 명시, 판례 형식)
7. Document 생성 시 메타데이터 추가
8. Stopwords 자동 생성
"""

# =============================================================================
# 공통 Import (상단 통합)
# =============================================================================
import os
import re
import io
import json
import zipfile
import hashlib
from collections import Counter
from operator import itemgetter

import pandas as pd
import streamlit as st
import altair as alt
from dotenv import load_dotenv

from openai import OpenAI
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import loading
from langchain_core.runnables import RunnableLambda, RunnableMap
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# =============================================================================
# 환경 설정
# =============================================================================
load_dotenv()

# 페이지 설정 (한 번만 호출)
st.set_page_config(page_title="⚖️ 법률 서비스 통합", layout="wide")
st.title("⚖️ 법률 서비스 통합 페이지")

# =============================================================================
# 경로 설정 (Docker / 로컬 자동 감지)
# =============================================================================
def get_base_paths():
    """Docker 환경과 로컬 환경에 따라 경로 자동 설정"""
    # Docker 환경: /app/data, /app/mycache
    # 로컬 환경: ../data, ./mycache (scripts 폴더 기준)
    if os.path.exists("/app/data"):
        # Docker 환경
        return {
            "data": "/app/data",
            "cache": "/app/mycache"
        }
    else:
        # 로컬 환경 (scripts 폴더에서 실행)
        return {
            "data": "../data",
            "cache": "./mycache"
        }

PATHS = get_base_paths()
DATA_DIR = PATHS["data"]
CACHE_DIR = PATHS["cache"]

# 캐시 폴더 생성
CACHE_DIRS = [
    CACHE_DIR,
    f"{CACHE_DIR}/files",
    f"{CACHE_DIR}/embedding",
    f"{CACHE_DIR}/vectorstore"
]
for path in CACHE_DIRS:
    os.makedirs(path, exist_ok=True)

store = LocalFileStore(f"{CACHE_DIR}/embedding")

# =============================================================================
# 공통 유틸리티 함수
# =============================================================================

def get_api_key():
    """OpenAI API 키 로더 (env + session_state 통합)"""
    key = os.getenv("OPENAI_API_KEY") or st.session_state.get("OPENAI_API_KEY")
    if not key:
        with st.sidebar:
            st.warning("⚠️ OpenAI API 키가 없으면 LLM 기능이 제한됩니다.")
            k = st.text_input("OPENAI_API_KEY 입력", type="password", key="api_key_input")
            if k:
                st.session_state["OPENAI_API_KEY"] = k
                key = k
    return key


def llm_available():
    """LLM 사용 가능 여부 확인"""
    return bool(get_api_key())


def get_openai_client():
    """OpenAI 클라이언트 생성 (API 키 확인 후)"""
    api_key = get_api_key()
    if api_key:
        return OpenAI(api_key=api_key)
    return None


# =============================================================================
# 데이터 로드 함수 (공통)
# =============================================================================

@st.cache_data
def load_data():
    """ZIP 파일에서 CSV 데이터 로드 및 병합"""
    zip_files = [
        f"{DATA_DIR}/result_1_merged.zip",
        f"{DATA_DIR}/result_2_merged.zip",
        f"{DATA_DIR}/result_3_merged.zip"
    ]
    all_dfs = []

    for zip_file in zip_files:
        try:
            with zipfile.ZipFile(zip_file) as z:
                for filename in z.namelist():
                    if filename.endswith(".csv"):
                        with z.open(filename) as f:
                            df = pd.read_csv(io.StringIO(f.read().decode("utf-8")))
                            all_dfs.append(df)
        except FileNotFoundError:
            st.warning(f"파일을 찾을 수 없습니다: {zip_file}")

    if not all_dfs:
        return pd.DataFrame()

    merged_df = pd.concat(all_dfs, ignore_index=True)

    # 불필요한 컬럼 제거
    if "Unnamed: 0" in merged_df.columns:
        merged_df.drop("Unnamed: 0", axis=1, inplace=True)

    # 참조조문이 없는 행 제거
    merged_df.dropna(subset=["참조조문"], inplace=True)
    merged_df.reset_index(drop=True, inplace=True)

    return merged_df


@st.cache_data
def load_code_map():
    """사건 코드 매핑 테이블 로드"""
    try:
        return pd.read_csv(f"{DATA_DIR}/code_map.csv", quoting=1)
    except FileNotFoundError:
        return pd.DataFrame()


@st.cache_data
def build_stopwords(df, top_n=100):
    """데이터에서 자동으로 불용어 생성"""
    all_words = []
    target_cols = ["판시사항", "판결요지", "참조조문", "사건명", "요약", "판례내용", "쟁점", "재판부_판단", "결과"]

    for col in target_cols:
        if col in df.columns:
            texts = df[col].dropna().astype(str)
            for text in texts:
                all_words.extend(re.findall(r'[가-힣a-zA-Z0-9]{2,}', text))

    common_words = [w for w, _ in Counter(all_words).most_common(top_n)]
    extra_stopwords = {"어제", "먹고", "쳤어", "어떻게", "되는거야", "했는데", "인데", "있어", "없어", "하다가", "넘어서", "같아", "사람을"}

    return set(common_words).union(extra_stopwords)


# =============================================================================
# 탭1 전용 함수: 사건 검색 시뮬레이션
# =============================================================================

def llm_classify_intent(text: str, api_key: str) -> dict:
    """LLM을 사용한 사건 의도/키워드 추출 (예외 처리 포함)"""
    try:
        client = OpenAI(api_key=api_key)

        prompt = f"""
사용자의 법률 사건 설명을 보고, 아래 기준에 따라 도메인을 반드시 올바르게 분류하라:

- "살인, 폭행, 성폭행, 가정폭력, 상해, 사망, 강도, 마약, 절도, 음주운전" 등이 포함되면 반드시 형사.
- "이혼, 양육권, 상속, 친권, 입양" 등 가정 내 분쟁이면 가사.
- "세금, 과세, 신고, 행정처분" 등은 행정.
- "계약, 손해배상, 채무, 임대차, 금전" 등은 민사.
- "특허, 상표, 지적재산"은 특허.
- "선거" 관련은 선거.
- 그 외에는 가장 적합한 분야를 선택.

추가로:
1. 주요 쟁점 키워드(issue_tags) 2~5개
2. 검색 recall을 높일 수 있는 키워드 8~15개
3. 위 키워드를 OR로 묶은 regex

출력은 JSON만:
{{
  "domain": "형사",
  "issue_tags": ["폭행", "상해"],
  "search_keywords": ["폭행", "상해", "폭력", "구타", "신체", "피해"],
  "regex": "(폭행|상해|폭력|구타|신체|피해)"
}}

사용자 입력: {text}
"""

        resp = client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            messages=[
                {"role": "system", "content": "항상 JSON만 출력"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )

        txt = resp.choices[0].message.content
        try:
            return json.loads(txt)
        except json.JSONDecodeError:
            # JSON 추출 시도
            m = re.search(r"\{.*\}", txt, re.S)
            return json.loads(m.group(0)) if m else {}

    except Exception as e:
        st.info(f"🔄 LLM 분석 실패 (자동 폴백): {e}")
        return {}


def konlpy_extract_keywords_or_fallback(situation: str) -> tuple:
    """KoNLPy로 키워드 추출 (Java 없으면 자동 폴백)"""

    def simple_keywords(text):
        """간단한 키워드 매칭 (폴백용)"""
        priority = [
            '마약', '성폭행', '음주운전', '사기', '살인', '폭행', '횡령', '배임',
            '상해', '협박', '방화', '절도', '성매매', '이혼', '상속', '임대차',
            '손해배상', '계약', '채무', '명예훼손'
        ]
        found = [w for w in priority if w in text]
        regex = '|'.join(found) if found else ".*"
        return found, regex

    try:
        from konlpy.tag import Okt
        okt = Okt()
        nouns = okt.nouns(situation)

        priority_words = [
            '마약', '성폭행', '음주운전', '사기', '살인', '폭행', '횡령', '배임',
            '상해', '협박', '방화', '절도', '성매매', '이혼', '상속', '임대차'
        ]
        keywords = [w for w in nouns if w in priority_words] or nouns[:5]
        query_regex = '|'.join(keywords) if keywords else ".*"

        return keywords, query_regex

    except Exception as e:
        st.info(f"🔄 KoNLPy 불가 (자동 폴백): {e}")
        return simple_keywords(situation)


def explain_law_article(law: str, api_key: str = None) -> str:
    """참조조문 설명 생성"""
    if not api_key:
        return "설명 없음"

    try:
        client = OpenAI(api_key=api_key)
        prompt = f"'{law}' 이 조문이 법률 사건에서 자주 등장하는 이유를 간단히 설명해줘."

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "간단한 이유만 출력"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )

        explanation = resp.choices[0].message.content.strip()
        return explanation if explanation else "설명 없음"

    except Exception:
        return "설명 없음"


def extract_abbr(case_no: str) -> str:
    """사건번호에서 약어 추출"""
    m = re.search(r"[가-힣A-Za-z]{1,3}", str(case_no))
    return m.group(0) if m else None


def infer_domain_from_codes(results_df) -> str:
    """검색 결과에서 도메인 추론"""
    if results_df.empty:
        return None
    if "domain" in results_df.columns:
        counts = results_df["domain"].value_counts()
        if not counts.empty:
            return counts.idxmax()
    return None


# =============================================================================
# 탭2 전용 함수: RAG 챗봇
# =============================================================================

@st.cache_resource(show_spinner="📚 벡터 DB 준비 중...")
def create_retriever_from_df(sel_df):
    """DataFrame 기반 retriever 생성 (임베딩 캐싱 + FAISS 재사용)"""

    df_hash = hashlib.md5(
        pd.util.hash_pandas_object(sel_df, index=True).values
    ).hexdigest()
    faiss_path = f"{CACHE_DIR}/vectorstore/csv_{df_hash}"

    if os.path.exists(faiss_path):
        vectorstore = FAISS.load_local(
            faiss_path, OpenAIEmbeddings(), allow_dangerous_deserialization=True
        )
    else:
        # Document 리스트 생성 (조문별 분리 + 메타데이터)
        docs = []
        for i, row in sel_df.iterrows():
            row_dict = row.to_dict()
            row_hash = hashlib.md5(str(row_dict).encode()).hexdigest()

            # 참조조문별로 Document 분리
            참조조문 = str(row.get('참조조문', ''))
            for 조문 in 참조조문.split(','):
                조문 = 조문.strip()
                if not 조문:
                    continue

                content = f"""참조조문: {조문}
배상책임: {row.get('배상책임', '')}
주제: {row.get('주제', '')}
재판부_판단: {row.get('재판부_판단', '')}
결과: {row.get('결과', '')}
요약: {row.get('요약', '')}
사건번호: {row.get('사건번호', '')}"""

                docs.append(Document(
                    page_content=content,
                    metadata={"row_id": row_hash, "조문": 조문}
                ))

        # 임베딩 + 캐시
        embedding = OpenAIEmbeddings()
        cached_embedder = CacheBackedEmbeddings.from_bytes_store(embedding, store)

        vectorstore = FAISS.from_documents(docs, embedding=cached_embedder)
        vectorstore.save_local(faiss_path)

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 30, "lambda_mult": 0.5}
    )
    return retriever


@st.cache_resource(show_spinner="📄 PDF 처리 중...")
def create_retriever_from_pdf(uploaded_file):
    """PDF 파일 기반 retriever 생성"""

    # 임시 파일 저장
    temp_path = f"{CACHE_DIR}/files/{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    loader = PDFPlumberLoader(temp_path)
    pdf_docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(pdf_docs)

    embedding = OpenAIEmbeddings()
    cached_embedder = CacheBackedEmbeddings.from_bytes_store(embedding, store)

    file_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()
    faiss_path = f"{CACHE_DIR}/vectorstore/pdf_{file_hash}"

    if os.path.exists(faiss_path):
        vectorstore = FAISS.load_local(
            faiss_path, cached_embedder, allow_dangerous_deserialization=True
        )
    else:
        vectorstore = FAISS.from_documents(chunks, embedding=cached_embedder)
        vectorstore.save_local(faiss_path)

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 30, "lambda_mult": 0.5}
    )
    return retriever


def create_rag_chain(retriever):
    """RAG 체인 생성 (개선된 프롬프트)"""

    join_docs = RunnableLambda(lambda docs: "\n".join(doc.page_content for doc in docs))

    prompt_text = {
        "_type": "prompt",
        "template": """당신은 한국어로 법률 상담을 제공하는 전문 AI 어시스턴트입니다.

⛔ 반드시 유의하세요: context에 없는 내용은 절대 생성하지 마세요.
모든 답변은 반드시 제공된 context에 기반해야 하며, 부족한 경우에는 '정확히 알 수 없습니다'라고 명확히 밝혀야 합니다.

🧭 다음의 절차와 형식을 따르세요:
1️⃣ 질문의 유형을 분류하세요: 정보 탐색 / 사건 상담 / 법 해석 / 예방 조언 / 기타
2️⃣ 질문자의 상황 및 맥락(chat_history)을 반영해 맞춤형으로 답변하세요.
3️⃣ context에서 관련 정보를 최대한 활용해 실무적인 답변을 구성하세요.
4️⃣ 판례나 사건번호가 있다면, 반드시 아래 형식을 따르세요:
   ### ✅ 판례 예시
   - 사건번호: [사건번호]
   - 판결 내용 요약: [간략한 설명]
   👉 더 많은 판례 확인: https://portal.scourt.go.kr/pgp/index.on?m=PGP1011M01&l=N&c=900
   📌 판례나 사건번호가 포함된 정보가 context에 없다면, 해당 섹션은 생략하세요.
5️⃣ 마크다운 형식으로 이모지와 헤더를 활용해 시각적으로 구성하세요.
6️⃣ 민감하거나 감정적인 사안은 공감과 배려를 포함한 언어로 표현하세요.
7️⃣ 복잡한 질문에는 답변 마지막에 간결한 요약을 포함하세요.
8️⃣ 마지막에 사용자가 후속 질문을 할 수 있도록 유도하세요.
9️⃣ 출처가 context 기반인지, 모델의 사전 지식인지 반드시 명시하세요.

<information>
{context}
</information>

#질문:
{question}

#이전 대화 내용:
{chat_history}

#답변:
""",
        "input_variables": ["question", "context", "chat_history"],
    }

    prompt = loading.load_prompt_from_config(prompt_text)
    llm = ChatOpenAI(model="gpt-4o-mini-2024-07-18", temperature=0)

    chain = (
        RunnableMap({
            "context": itemgetter("question") | retriever | join_docs,
            "question": itemgetter("question"),
            "chat_history": itemgetter("chat_history"),
        })
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


# =============================================================================
# 탭 스타일링
# =============================================================================
st.markdown("""
<style>
.stTabs [data-baseweb="tab"] p {
    font-size: 20px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 메인 탭 구성
# =============================================================================
tab1, tab2 = st.tabs(["🔍 사건 검색 시뮬레이션", "💬 RAG 기반 상담 챗봇"])

# =============================================================================
# 탭1: 사건 검색 시뮬레이션
# =============================================================================
with tab1:
    # 데이터 로드
    df_search = load_data()

    if df_search.empty:
        st.error("❌ 데이터가 없습니다. data 폴더에 ZIP 파일을 확인하세요.")
        st.stop()

    # code_map 병합
    code_map = load_code_map()
    if not code_map.empty:
        df_search["사건코드약어"] = df_search["사건번호"].apply(extract_abbr)
        df_search = df_search.merge(code_map, left_on="사건코드약어", right_on="abbr", how="left")

    # Stopwords 생성
    stopwords = build_stopwords(df_search)

    # 레이아웃
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("📝 본인 입장 및 사건 정보 입력")

        position = st.selectbox("본인 입장", ["원고", "피고", "검사", "기타"], key="position")
        settlement = st.radio("합의 여부", ["합의 안 함", "합의 함"], key="settlement", horizontal=True)
        damage = st.number_input("피해금액 (만원 단위)", min_value=0, step=100, key="damage")
        situation = st.text_area("사건 내용을 구체적으로 입력하세요", height=150, key="situation")

        if st.button("🔍 검색 및 예상 결과", type="primary"):
            if not situation.strip():
                st.warning("⚠️ 사건 내용을 입력해 주세요.")
            else:
                api_key = get_api_key()
                expanded_terms, query_regex, llm_domain = [], ".*", None

                # 1) LLM 시도 (실패시 자동 폴백)
                if llm_available():
                    intent = llm_classify_intent(situation, api_key)
                    if intent:
                        expanded_terms = intent.get("search_keywords", [])
                        query_regex = intent.get("regex", ".*")
                        llm_domain = intent.get("domain")
                        st.success(f"🤖 LLM 감지 도메인: **{llm_domain}** / 이슈: {intent.get('issue_tags')}")
                        st.write(f"🔍 검색 키워드: {expanded_terms[:10]}")

                # 2) LLM 실패시 KoNLPy 또는 simple 폴백
                if not expanded_terms:
                    keywords, query_regex = konlpy_extract_keywords_or_fallback(situation)
                    st.info(f"📝 추출 키워드: {keywords[:10]}")

                # 3) 검색 수행 (AND → OR 폴백)
                df_search['combined_text'] = df_search[['요약', '판결요지', '쟁점']].fillna('').agg(' '.join, axis=1)

                if expanded_terms:
                    # AND 검색 우선
                    top_keywords = expanded_terms[:3]
                    keywords_regex = '(?=.*' + ')(?=.*'.join([re.escape(k) for k in top_keywords]) + ')'
                    search_results = df_search[df_search['combined_text'].str.contains(keywords_regex, na=False, case=False)]

                    # AND 실패시 OR 폴백
                    if search_results.empty:
                        fallback_regex = '|'.join([re.escape(k) for k in expanded_terms])
                        search_results = df_search[df_search['combined_text'].str.contains(fallback_regex, na=False, case=False)]
                        if not search_results.empty:
                            st.info(f"🔄 정확히 일치하는 사건은 없지만, 유사 사건 {len(search_results)}건을 찾았습니다.")
                else:
                    search_results = df_search[df_search['combined_text'].str.contains(query_regex, na=False, case=False)]

                st.write(f"📊 검색 결과: **{len(search_results)}건**")

                # 도메인 보정
                inferred_domain = infer_domain_from_codes(search_results)
                final_domain = inferred_domain if inferred_domain else llm_domain
                if final_domain:
                    st.write(f"📂 최종 도메인: **{final_domain}**")

                # 세션에 저장
                st.session_state.search_results = search_results
                st.session_state.final_domain = final_domain
                st.session_state.search_position = position
                st.session_state.search_settlement = settlement
                st.session_state.search_damage = damage

    # 검색 결과 표시
    if "search_results" in st.session_state:
        search_results = st.session_state.search_results

        if search_results.empty:
            st.warning("🔍 검색된 판례가 없습니다. 키워드를 조금 더 일반화해보세요.")
        else:
            # 승률 계산
            position = st.session_state.get("search_position", "원고")
            settlement = st.session_state.get("search_settlement", "합의 안 함")
            damage = st.session_state.get("search_damage", 0)

            if position in ["원고", "피고"]:
                relevant = search_results[search_results["결과"].str.contains(position, na=False)]
                base_win_rate = len(relevant) / len(search_results) if len(search_results) > 0 else 0.5
            else:
                base_win_rate = 0.5

            rate = base_win_rate
            if position == "원고" and settlement == "합의 함":
                rate += 0.1
            elif position == "피고" and settlement == "합의 안 함":
                rate -= 0.1
            rate -= damage / 10000
            rate = max(0, min(1, rate))

            # 결과 표시
            st.markdown("---")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("🎯 기본 승률", f"{base_win_rate:.1%}")
            with col_b:
                st.metric("⚖️ 예상 승소 확률", f"{rate:.1%}")
            with col_c:
                st.metric("💰 예상 배상액", f"{damage*0.5:.0f} ~ {damage*1.5:.0f} 만원")

            st.progress(rate)

            if rate < 0.3:
                st.error("❌ 실형 가능성이 높습니다!")
            elif rate < 0.6:
                st.warning("⚠️ 위험도가 중간입니다.")
            else:
                st.success("✅ 승소 가능성이 높습니다.")

            # 사건 상세보기 (페이지네이션)
            st.markdown("---")
            st.subheader("📄 사건 상세보기")

            PAGE_SIZE = 10
            total = len(search_results)
            total_pages = (total + PAGE_SIZE - 1) // PAGE_SIZE

            if "detail_page" not in st.session_state:
                st.session_state.detail_page = 1

            # 페이지네이션 컨트롤
            col_nav = st.columns([1, 1, 3, 1, 1])
            with col_nav[0]:
                if st.button("⏮️", disabled=st.session_state.detail_page == 1, key="first"):
                    st.session_state.detail_page = 1
                    st.rerun()
            with col_nav[1]:
                if st.button("◀️", disabled=st.session_state.detail_page == 1, key="prev"):
                    st.session_state.detail_page -= 1
                    st.rerun()
            with col_nav[2]:
                st.markdown(f"<div style='text-align:center'>페이지 {st.session_state.detail_page}/{total_pages} (총 {total}건)</div>", unsafe_allow_html=True)
            with col_nav[3]:
                if st.button("▶️", disabled=st.session_state.detail_page >= total_pages, key="next"):
                    st.session_state.detail_page += 1
                    st.rerun()
            with col_nav[4]:
                if st.button("⏭️", disabled=st.session_state.detail_page >= total_pages, key="last"):
                    st.session_state.detail_page = total_pages
                    st.rerun()

            # 현재 페이지 데이터
            start = (st.session_state.detail_page - 1) * PAGE_SIZE
            end = min(start + PAGE_SIZE, total)
            page_df = search_results.iloc[start:end]

            def getv(r, col):
                return r[col] if col in r and pd.notna(r[col]) else "-"

            for idx, row in page_df.iterrows():
                title = f"{getv(row, '사건번호')} | {getv(row, '사건명')} | {getv(row, '결과')}"
                with st.expander(title):
                    col_d1, col_d2 = st.columns(2)
                    with col_d1:
                        st.write(f"**사건 코드**: {getv(row, '사건코드약어')} ({getv(row, 'domain')})")
                        st.write(f"**사건 유형**: {getv(row, 'type')}")
                        st.write(f"**사건 종류**: {getv(row, '사건종류명')}")
                    with col_d2:
                        st.write(f"**판결 유형**: {getv(row, '판결유형')}")
                        st.write(f"**배상 책임**: {getv(row, '배상책임')}")
                        st.write(f"**결과**: {getv(row, '결과')}")

                    st.write(f"**주요 쟁점**: {getv(row, '쟁점')}")
                    st.write(f"**요약**: {getv(row, '요약')}")

                    if getv(row, "판시사항") != "-":
                        with st.expander("📜 판시사항 자세히 보기"):
                            st.write(getv(row, "판시사항"))

                    if getv(row, "참조조문") != "-":
                        with st.expander("📖 참조조문 보기"):
                            api_key = get_api_key()
                            for law in str(row["참조조문"]).split(",")[:5]:
                                law = law.strip()
                                if law:
                                    explanation = explain_law_article(law, api_key) if api_key else "설명 없음"
                                    st.write(f"- **{law}**: {explanation}")

    # 오른쪽 통계
    with col_right:
        st.subheader("📊 판례 통계")

        if "사건종류명" in df_search.columns:
            st.markdown("**사건 종류 분포**")
            case_type_counts = df_search["사건종류명"].value_counts().head(10)
            st.bar_chart(case_type_counts)

        st.markdown("**참조조문 Top 5**")
        all_laws = []
        for c in df_search["참조조문"].dropna():
            all_laws.extend([x.strip() for x in str(c).split(",")])
        law_counts = Counter(all_laws)
        law_df = pd.DataFrame(
            law_counts.items(), columns=["조문", "횟수"]
        ).sort_values(by="횟수", ascending=False).head(5)
        st.table(law_df)

# =============================================================================
# 탭2: RAG 기반 상담 챗봇
# =============================================================================
with tab2:
    # 세션 초기화
    if "chat_chain" not in st.session_state:
        st.session_state["chat_chain"] = None
    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = []

    def add_chat_message(role, message):
        st.session_state["chat_messages"].append({"role": role, "content": message})

    st.subheader("💬 RAG 기반 법률 상담 챗봇")

    # PDF 업로드 옵션
    uploaded_pdf = st.file_uploader("📄 PDF 문서 업로드 (선택사항)", type="pdf", key="pdf_upload")

    # 데이터 로드 및 retriever 생성
    df_chat = load_data()

    if df_chat.empty and not uploaded_pdf:
        st.error("❌ 데이터가 없습니다. data 폴더를 확인하거나 PDF를 업로드하세요.")
    else:
        # Retriever 생성
        if uploaded_pdf:
            retriever = create_retriever_from_pdf(uploaded_pdf)
            st.success(f"✅ PDF '{uploaded_pdf.name}' 로드 완료!")
        else:
            sel_df = df_chat[['참조조문', '배상책임', '주제', '재판부_판단', '결과', '요약', '사건번호']].head(1000)
            retriever = create_retriever_from_df(sel_df)

        # Chain 생성
        chain = create_rag_chain(retriever)
        st.session_state["chat_chain"] = chain

        # 채팅 기록 표시
        for chat_message in st.session_state["chat_messages"]:
            st.chat_message(chat_message["role"]).write(chat_message["content"])

        # 사용자 입력
        user_input = st.chat_input("💬 법률 관련 질문을 입력하세요")

        if user_input:
            st.session_state["chat_messages"].append({"role": "user", "content": user_input})

            # 히스토리 생성
            history_str = "\n".join(
                f"{m['role']}: {m['content']}" for m in st.session_state["chat_messages"]
            )

            # 질문 확장 (LLM)
            client = get_openai_client()
            if client:
                try:
                    expand_prompt = f"""
이전 대화를 참고해서 "{user_input}"라는 질문을
구체적이고 다른 관점의 질문으로 확장해줘.
단, 이전 답변을 반복하지 말고 새로운 법적 쟁점을 포함해야 한다.
"""
                    expanded_response = client.chat.completions.create(
                        model="gpt-4o-mini-2024-07-18",
                        messages=[{"role": "system", "content": expand_prompt}]
                    )
                    expanded_question = expanded_response.choices[0].message.content
                except Exception:
                    expanded_question = user_input
            else:
                expanded_question = user_input

            # Chain 실행
            payload = {"question": expanded_question, "chat_history": history_str}
            response = chain.stream(payload)

            st.chat_message("user").write(user_input)

            with st.chat_message("assistant"):
                container = st.empty()
                ai_answer = ""
                for token in response:
                    ai_answer += token
                    container.write(ai_answer)
                add_chat_message("assistant", ai_answer)

# =============================================================================
# 푸터
# =============================================================================
st.markdown("---")
st.caption("⚠️ 본 서비스는 참고용이며, 실제 법률 자문이 아닙니다. 정확한 법률 상담은 전문 변호사에게 문의하세요.")
