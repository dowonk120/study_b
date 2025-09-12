import streamlit as st

# 페이지 기본 설정
st.set_page_config(page_title="⚖️ 사건 검색 시뮬레이션", layout="wide")
st.title("⚖️ 법률 서비스 통합 페이지")

# 탭 UI 글씨 커스터마이징
st.markdown(
    """
    <style>
    .stTabs [data-baseweb="tab"] p {
        font-size: 25px;  /* 글씨 크기 */
        font-weight: 600; /* 글씨 두께 */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 탭 생성
tab1, tab2 = st.tabs(["사건 검색 시뮬레이션", "RAG 기반 상담 챗봇"])

# -------------------------
# 탭 1: 사건 검색 & 시뮬
# -------------------------
with tab1:
    import os, re, io, json, zipfile
    import pandas as pd
    import streamlit as st
    from collections import Counter
    from dotenv import load_dotenv
    import altair as alt

    load_dotenv()

    st.set_page_config(page_title="⚖️ 사건 검색 시뮬레이션", layout="wide")
    # st.title("⚖️ 법률 사건 검색 및 예상 결과 (LLM 보강 버전)")

    # --------------------
    # API Key 로더
    # --------------------
    def get_api_key():
        key = os.getenv("OPENAI_API_KEY")
        if key:
            return key
        with st.sidebar:
            st.warning("OpenAI API 키가 없으면 LLM 기능이 제한됩니다.")
            k = st.text_input("OPENAI_API_KEY 입력 (선택)", type="password")
            if k:
                st.session_state["OPENAI_API_KEY"] = k
                return k
        return None

    # --------------------
    # CSV 로드
    # --------------------
    @st.cache_data
    def load_data():
        zip_files = ["data/result_1_merged.zip", "data/result_2_merged.zip", "data/result_3_merged.zip"]
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
        merged_df.dropna(how="all", inplace=True)
        merged_df.reset_index(drop=True, inplace=True)
        return merged_df

    df = load_data()
    if df.empty:
        st.error("데이터가 없습니다.")
        st.stop()

    # --------------------
    # code_map.csv 로드
    # --------------------
    @st.cache_data
    def load_code_map():
        return pd.read_csv("data/code_map.csv", quoting=1)

    code_map = load_code_map()

    def extract_abbr(case_no: str):
        m = re.search(r"[가-힣A-Za-z]{1,3}", str(case_no))
        return m.group(0) if m else None

    df["사건코드약어"] = df["사건번호"].apply(extract_abbr)
    df = df.merge(code_map, left_on="사건코드약어", right_on="abbr", how="left")

    # --------------------
    # LLM 의도/키워드 추출
    # --------------------
    def llm_classify_intent(text, api_key):
        from openai import OpenAI
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

        출력 형식(JSON):
        {{
        "domain": "...",
        "issue_tags": [...],
        "search_keywords": [...],
        "regex": "..."
        }}
        """

        resp = client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            messages=[{"role":"system","content":"항상 JSON만 출력"}, {"role":"user","content":prompt + "\n\n사건: " + text}],
            temperature=0.2
        )
        txt = resp.choices[0].message.content
        try:
            return json.loads(txt)
        except:
            m = re.search(r"\{.*\}", txt, re.S)
            return json.loads(m.group(0)) if m else {}

    # --------------------
    # 참조조문 설명 함수
    # --------------------
    def explain_law_article(law, api_key=None):
        if not api_key:
            return "설명 없음"
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            prompt = f"""'{law}' 이 조문이 법률 사건에서 자주 등장하는 이유를 간단히 설명해줘."""
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": "간단한 이유만 출력"}, {"role": "user", "content": prompt}],
                temperature=0.2
            )
            explanation = resp.choices[0].message.content.strip()
            return explanation if explanation else "설명 없음"
        except:
            return "설명 없음"

    # --------------------
    # domain 보정: code_map 기반
    # --------------------
    def infer_domain_from_codes(results_df):
        if results_df.empty:
            return None
        counts = results_df["domain"].value_counts()
        if not counts.empty:
            return counts.idxmax()
        return None

    # --------------------
    # 화면 레이아웃
    # --------------------
    col_left, col_right = st.columns([2,1])

    with col_left:
        st.subheader("본인 입장 및 사건 정보 입력")
        position = st.selectbox("본인 입장", ["원고","피고","검사","기타"])
        settlement = st.radio("합의 여부", ["합의 안 함", "합의 함"])
        damage_input = st.text_input("피해금액 (만원 단위, 없으면 0 입력)", "0")
        try:
            damage = float(damage_input)
        except:
            damage = 0.0

        situation = st.text_area("사건 내용을 입력하세요", height=150)
        go = st.button("검색하기")

        if go:
            if not situation.strip():
                st.warning("사건 내용을 입력해 주세요.")
                st.stop()

            api_key = get_api_key()
            expanded_terms, query_regex, llm_domain = [], ".*", None

            if api_key:
                try:
                    intent = llm_classify_intent(situation, api_key)
                    expanded_terms = intent.get("search_keywords", [])
                    query_regex = intent.get("regex", ".*")
                    llm_domain = intent.get("domain")
                    st.write(f"감지된 도메인 (LLM): {llm_domain} / 이슈: {intent.get('issue_tags')}")
                    st.write(f"LLM 키워드: {expanded_terms[:10]}")
                except Exception as e:
                    st.info(f"LLM 분석 실패: {e}")

            # 사건 검색
            df['combined_text'] = df[['요약','판결요지','쟁점']].fillna('').agg(' '.join, axis=1)

            if expanded_terms:
                # 우선 AND 검색
                top_keywords = expanded_terms[:3]
                keywords_regex = '(?=.*' + ')(?=.*'.join([re.escape(k) for k in top_keywords]) + ')'
                search_results = df[df['combined_text'].str.contains(keywords_regex, na=False, case=False)]
            else:
                search_results = df[df['combined_text'].str.contains(query_regex, na=False, case=False)]

            # fallback OR 검색
            if search_results.empty and expanded_terms:
                fallback_regex = '|'.join([re.escape(k) for k in expanded_terms])
                search_results = df[df['combined_text'].str.contains(fallback_regex, na=False, case=False)]
                if not search_results.empty:
                    st.info(f"정확히 일치하는 사건은 없지만, 유사 사건 {len(search_results)}건을 찾았습니다.")

            st.write(f"검색 결과 {len(search_results)}건")

            # --- domain 보정 ---
            inferred_domain = infer_domain_from_codes(search_results)
            final_domain = inferred_domain if inferred_domain else llm_domain
            st.write(f"최종 도메인: {final_domain}")

            # 세션에 검색 결과 저장
            st.session_state.search_results = search_results
            st.session_state.final_domain = final_domain

    # --------------------
    # 상세보기 & 통계 출력
    # --------------------
    if "search_results" in st.session_state:
        search_results = st.session_state.search_results
        final_domain = st.session_state.get("final_domain", None)

        if search_results.empty:
            st.warning("검색된 판례가 없습니다. 키워드를 조금 더 일반화하거나 다시 입력해보세요.")
        else:
            # 기본 승률 계산
            position = st.session_state.get("position", "원고")
            settlement = st.session_state.get("settlement", "합의 안 함")
            damage = st.session_state.get("damage", 0.0)

            if position in ["원고", "피고"]:
                relevant = search_results[search_results["결과"].str.contains(position, na=False)]
                base_win_rate = len(relevant) / len(search_results) if len(search_results) > 0 else 0.5
            else:
                base_win_rate = 0.5

            st.write(f"검색 기반 기본 승률: {base_win_rate:.1%}")

            rate = base_win_rate
            if position == "원고" and settlement == "합의 함":
                rate += 0.1
            elif position == "피고" and settlement == "합의 안 함":
                rate -= 0.1
            rate -= damage / 10000
            rate = max(0, min(1, rate))

            st.write(f"예상 승소 확률: {rate:.1%}")
            st.write(f"예상 배상액 범위: {damage*0.5:.0f} ~ {damage*1.5:.0f} 만원")
            st.progress(rate)

            if rate < 0.3:
                st.error("실형 가능성이 높습니다!")
            elif rate < 0.6:
                st.warning("위험도가 중간입니다.")
            else:
                st.success("승소 가능성이 높습니다.")

            # 📄 사건 상세보기
            st.markdown("---")
            st.subheader("📄 사건 상세보기")

            PAGE_SIZE = 10
            total = len(search_results)
            total_pages = (total + PAGE_SIZE - 1) // PAGE_SIZE

            if "detail_page" not in st.session_state:
                st.session_state.detail_page = 1

            start = (st.session_state.detail_page-1)*PAGE_SIZE
            end = min(start+PAGE_SIZE, total)
            page_df = search_results.iloc[start:end]

            def getv(r, col): 
                return r[col] if col in r and pd.notna(r[col]) else "-"

            for idx, row in page_df.iterrows():
                title = f"{getv(row,'사건번호')} | {getv(row,'사건명')} | {getv(row,'결과')}"
                with st.expander(title):
                    st.write(f"**사건 코드**: {getv(row,'사건코드약어')} ({getv(row,'domain')})")
                    st.write(f"**사건유형**: {getv(row,'type')}")
                    st.write(f"**주요 쟁점**: {getv(row,'쟁점')}")
                    st.write(f"**결과**: {getv(row,'결과')}")
                    st.write(f"**요약**: {getv(row,'요약')}")

                    if getv(row,"판시사항") != "-":
                        with st.expander("판시사항 자세히 보기"):
                            st.write(getv(row,"판시사항"))

                    if getv(row,"참조조문") != "-":
                        with st.expander("참조조문 보기"):
                            for law in str(row["참조조문"]).split(","):
                                law = law.strip()
                                if law:
                                    st.write(f"- {law}")

            # 페이지네이션 버튼
            st.markdown("---")
            cols = st.columns(7)
            if cols[0].button("<<", disabled=st.session_state.detail_page == 1):
                st.session_state.detail_page = 1
            if cols[1].button("<", disabled=st.session_state.detail_page == 1):
                st.session_state.detail_page -= 1
            if cols[5].button(">", disabled=st.session_state.detail_page == total_pages):
                st.session_state.detail_page += 1
            if cols[6].button(">>", disabled=st.session_state.detail_page == total_pages):
                st.session_state.detail_page = total_pages

            # 📊 오른쪽 통계
            with col_right:
                st.subheader("📊 검색 결과 통계")

                if "사건종류명" in search_results.columns:
                    st.markdown("**사건 종류 분포**")
                    case_type_counts = (
                        search_results["사건종류명"]
                        .value_counts()
                        .rename_axis("사건종류")
                        .reset_index(name="건수")
                    )
                    st.bar_chart(case_type_counts.set_index("사건종류")["건수"])

                if "참조조문" in search_results.columns:
                    st.markdown("**참조조문 Top 5**")
                    all_laws = []
                    for c in search_results["참조조문"].dropna():
                        all_laws.extend([x.strip() for x in str(c).split(",")])
                    law_counts = Counter(all_laws)
                    law_df = (
                        pd.DataFrame(law_counts.items(), columns=["조문","횟수"])
                        .sort_values(by="횟수", ascending=False)
                        .head(5)
                    )
                    api_key = get_api_key()
                    if not law_df.empty:
                        law_df["이유"] = law_df["조문"].apply(lambda x: explain_law_article(x, api_key))
                        st.table(law_df)

    st.caption("※ 본 서비스는 참고용이며, 실제 법률 자문이 아닙니다.")



# -------------------------
# 탭 2: RAG 기반 상담 챗봇
# -------------------------
with tab2:
    from openai import OpenAI
    import streamlit as st
    from langchain_core.messages.chat import ChatMessage
    from langchain_openai import ChatOpenAI
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnableLambda, RunnableMap
    import os
    import io
    import zipfile
    import pandas as pd
    import hashlib
    from operator import itemgetter
    from langchain.vectorstores import FAISS
    from langchain.embeddings import OpenAIEmbeddings
    from langchain.embeddings.cache import CacheBackedEmbeddings
    from langchain.docstore.document import Document
    from langchain.storage import LocalFileStore
    from langchain_core.prompts import loading
    from langchain_community.document_loaders import PDFPlumberLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    # --------------------
    # 캐시/디렉토리 생성
    # --------------------
    for path in ["./mycache", "./mycache/files", "./mycache/embedding", "./mycache/vectorstore"]:
        if not os.path.isdir(path):
            os.mkdir(path)

    store = LocalFileStore("./mycache/embedding")

    # --------------------
    # CSV 로드
    # --------------------
    @st.cache_data
    def load_data():
        zip_files = ["data/result_1_merged.zip", "data/result_2_merged.zip", "data/result_3_merged.zip"]
        all_dfs = []
        for zip_file in zip_files:
            with zipfile.ZipFile(zip_file) as z:
                for filename in z.namelist():
                    if filename.endswith(".csv"):
                        with z.open(filename) as f:
                            df = pd.read_csv(io.StringIO(f.read().decode("utf-8")))
                            all_dfs.append(df)
        merged_df = pd.concat(all_dfs, ignore_index=True)
        merged_df.dropna(subset=["참조조문"], inplace=True)
        merged_df.reset_index(drop=True, inplace=True)
        return merged_df

    df = load_data()
    sel_df = df[['참조조문','배상책임', '주제','재판부_판단', '결과','요약']].head(10)

    client = OpenAI()

    if "chain" not in st.session_state:
        st.session_state["chain"] = None
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    def add_message(role, message):
        st.session_state["messages"].append({"role": role, "content": message})

    st.title("RAG 기반 법률 챗봇")

    # --------------------
    # CSV/PDF retriever 생성
    # --------------------
    @st.cache_resource(show_spinner="업로드 파일 처리중 기다리세요")
    def create_retriever_from_csv(sel_df):
        df_hash = hashlib.md5(pd.util.hash_pandas_object(sel_df, index=True).values).hexdigest()
        faiss_path = f"./mycache/vectorstore/csv_{df_hash}"

        if os.path.exists(faiss_path):
            vectorstore = FAISS.load_local(faiss_path, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
        else:
            docs = []
            for _, row in sel_df.iterrows():
                for 조문 in str(row['참조조문']).split(','):
                    조문 = 조문.strip()
                    if not 조문: continue
                    content = f"{조문}\n배상책임: {row['배상책임']}\n주제: {row['주제']}\n재판부_판단: {row['재판부_판단']}\n결과: {row['결과']}\n요약: {row['요약']}"
                    docs.append(Document(page_content=content))
            embedding = OpenAIEmbeddings()
            cached_embedder = CacheBackedEmbeddings.from_bytes_store(embedding, store)
            vectorstore = FAISS.from_documents(docs, embedding=cached_embedder)
            vectorstore.save_local(faiss_path)

        retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k":5, "fetch_k":30, "lambda_mult":0.5})
        return retriever

    @st.cache_resource(show_spinner="PDF 처리중 기다리세요")
    def create_retriever_from_pdf(uploaded_file):
        loader = PDFPlumberLoader(uploaded_file)
        pdf_docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(pdf_docs)

        embedding = OpenAIEmbeddings()
        cached_embedder = CacheBackedEmbeddings.from_bytes_store(embedding, store)

        df_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()
        faiss_path = f"./mycache/vectorstore/pdf_{df_hash}"
        
        if os.path.exists(faiss_path):
            vectorstore = FAISS.load_local(faiss_path, cached_embedder, allow_dangerous_deserialization=True)
        else:
            vectorstore = FAISS.from_documents(chunks, embedding=cached_embedder)
            vectorstore.save_local(faiss_path)

        retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k":5, "fetch_k":30, "lambda_mult":0.5})
        return retriever

    # --------------------
    # Chain 생성
    # --------------------
    join_docs = RunnableLambda(lambda docs: "\n".join(doc.page_content for doc in docs))

    def create_chain(retriever):
        prompt_text = {
            "_type": "prompt",
            "template": (
                "You are a professional legal assistant that answers questions in Korean.\n"
                "Use the retrieved legal information to provide accurate and clear answers.\n"
                "Always include relevant legal references in a separate section.\n\n"
                "If you don't know the answer based on the information, say you don't know.\n\n"

                "Format the response in clean, user-friendly **Markdown** with emojis.\n"
                "Structure:\n"
                "1️⃣ 답변은 한 문단으로 시작하고 필요 시 bullet point로 정리합니다.\n"
                "2️⃣ 참조 조문은 '📜 **참조 조문**' 섹션에서 bullet point로 나열합니다.\n"
                "3️⃣ 마지막에 '혹시 이 주제에 대해 더 궁금한 것이 있으신가요?'를 추가합니다.\n\n"

                "<information>\n{context}\n</information>\n\n"
                "#Question:\n{question}\n\n"
                "#Answer:\n#chat_history:\n{chat_history} 답변은 chat_history를 참고해서 답변해\n\n"

                "⚖️ **답변**\n"
                "- (여기에 질문에 대한 명확하고 간결한 답변을 작성)\n\n"
                "📜 **참조 조문**\n"
                "- 각 조문 번호와 제목을 먼저 출력하고, **그 다음 줄에 해당 조문 내용을 출력**하세요.\n\n"
                " 동일한 키워드가 다시 등장하더라도 이전 답변을 반복하지 말고, 사건의 다른 법적 쟁점(결과, 고의성, 책임 경중 등)에 집중하세요.\n"
                "답변후에 혹시 이 주제에 대해 더 궁금한 것이 있으신가요?라고 물어봐줘"
            ),
            "input_variables": ["question", "context", "chat_history"],
        }

        prompt = loading.load_prompt_from_config(prompt_text)
        llm = ChatOpenAI(model="gpt-4o-mini-2024-07-18", temperature=0)

        chain = (
            RunnableMap(
                {
                    "context": itemgetter("question") | retriever | join_docs,
                    "question": itemgetter("question"),
                    "chat_history": itemgetter("chat_history"),
                }
            )
            | prompt
            | llm
            | StrOutputParser()
        )
        return chain

    # --------------------
    # PDF 업로드 UI
    # --------------------
    uploaded_pdf = st.file_uploader("보고서 업로드 (PDF)", type="pdf")

    if uploaded_pdf:
        retriever = create_retriever_from_pdf(uploaded_pdf)
    else:
        retriever = create_retriever_from_csv(sel_df)

    chain = create_chain(retriever)
    st.session_state["chain"] = chain

    # --------------------
    # Streamlit 채팅
    # --------------------
    def print_messages():
        for chat_message in st.session_state["messages"]:
            st.chat_message(chat_message["role"]).write(chat_message["content"])

    print_messages()
    
    user_input = st.chat_input("질문을 하세요")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        history_str = "\n".join(f"{m['role']} : {m['content']}" for m in st.session_state.messages)

        # 질문 확장
        expand_prompt = f"""
        이전 대화를 참고해서 "{user_input}"라는 질문을
        구체적이고 다른 관점의 질문으로 확장해줘.
        단, 이전 답변을 반복하지 말고 새로운 법적 쟁점을 포함해야 한다.
        """
        expanded_question = client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            messages=[{"role": "system", "content": expand_prompt}]
        ).choices[0].message.content

        payload = {"question": expanded_question, "chat_history": history_str}
        response = chain.stream(payload)

        st.chat_message("user").write(user_input)
        with st.chat_message("assistant"):
            container = st.empty()
            ai_answer = ""
            for token in response:
                    ai_answer += token
                    container.write(ai_answer)   # ✅ write로 출력
            add_message("assistant", ai_answer)