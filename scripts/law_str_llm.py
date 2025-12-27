import os, re, io, json, zipfile
import pandas as pd
import streamlit as st
from collections import Counter
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="⚖️ 법률 사건 검색+시뮬", layout="wide")
st.title("⚖️ 법률 사건 검색 및 예상 결과 (LLM 보강 버전)")

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
    zip_files = ["../data/result_1_merged.zip", "../data/result_2_merged.zip", "../data/result_3_merged.zip"]
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
    return pd.read_csv("../data/code_map.csv", quoting=1)  # quoting=1 → QUOTE_ALL 대응

code_map = load_code_map()

def extract_abbr(case_no: str):
    m = re.search(r"[가-힣A-Za-z]{1,3}", str(case_no))
    return m.group(0) if m else None

df["사건코드약어"] = df["사건번호"].apply(extract_abbr)
df = df.merge(code_map, left_on="사건코드약어", right_on="abbr", how="left")

# --------------------
# Stopwords
# --------------------
@st.cache_data
def build_stopwords(df, top_n=100):
    words = []
    for col in df.columns:
        texts = df[col].dropna().astype(str)
        for text in texts:
            words.extend(re.findall(r'[가-힣a-zA-Z0-9]{2,}', text))
    common_words = [w for w, _ in Counter(words).most_common(top_n)]
    return set(common_words)

stopwords = build_stopwords(df).union({"하다가","넘어서","같아","사람을"})

# --------------------
# LLM 의도/키워드 추출
# --------------------
def llm_classify_intent(text, api_key):
    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    prompt = f"""
    사용자의 법률 사건 설명을 보고:
    1. 사건 도메인 (민사/형사/가사/행정/특허/보호/선거/비송도산/집행/감치/신청/기타)
    2. 주요 쟁점 키워드(issue_tags) 2~5개
    3. 검색 recall을 높일 수 있는 키워드 8~15개
    4. 위 키워드를 OR로 묶은 regex

    출력은 JSON만:
    {{
      "domain": "가사",
      "issue_tags": ["외도","위자료","시어머니"],
      "search_keywords": ["외도","부정행위","상간","위자료","시어머니"],
      "regex": "(외도|부정행위|상간|위자료|시어머니)"
    }}
    """

    resp = client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=[{"role":"system","content":"항상 JSON만 출력"}, {"role":"user","content":prompt}],
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
        prompt = f"'{law}' 이 조문이 법률 사건에서 자주 등장하는 이유를 간단히 설명해줘."
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
# 입력 폼
# --------------------
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
    expanded_terms, query_regex = [], ".*"

    if api_key:
        try:
            intent = llm_classify_intent(situation, api_key)
            expanded_terms = intent.get("search_keywords", [])
            query_regex = intent.get("regex", ".*")
            st.write(f"감지된 도메인: {intent.get('domain')} / 이슈: {intent.get('issue_tags')}")
            st.write(f"LLM 키워드: {expanded_terms[:10]}")
        except Exception as e:
            st.info(f"LLM 분석 실패: {e}")

    # --------------------
    # 사건 검색
    # --------------------
    cond = False
    for col in ["판시사항","사건명","판례내용","요약","쟁점","재판부_판단","결과"]:
        if col in df.columns:
            mask = df[col].astype(str).str.contains(query_regex, na=False)
            cond = mask if cond is False else (cond | mask)

    search_results = df[cond] if isinstance(cond, pd.Series) else df.iloc[0:0]
    st.write(f"검색 결과 {len(search_results)}건")

    # --------------------
    # 기본 승률 계산
    # --------------------
    if position in ["원고", "피고"]:
        relevant = search_results[search_results["결과"].str.contains(position, na=False)]
        base_win_rate = len(relevant) / len(search_results) if len(search_results) > 0 else 0.5
    else:
        base_win_rate = 0.5

    st.write(f"검색 기반 기본 승률: {base_win_rate:.1%}")

    # 승률 조정
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

    # --------------------
    # 사건 상세보기 (페이지네이션 개선)
    # --------------------
    if not search_results.empty:
        st.markdown("---")
        st.subheader("📄 사건 상세보기")

        PAGE_SIZE = 10
        total = len(search_results)
        total_pages = (total + PAGE_SIZE - 1) // PAGE_SIZE

        if "detail_page" not in st.session_state:
            st.session_state.detail_page = 1

        col_prev, col_info, col_next = st.columns([1,4,1])
        with col_prev:
            if st.button("⬅ 이전", disabled=st.session_state.detail_page <= 1):
                st.session_state.detail_page -= 1
                st.rerun()

        with col_info:
            detail_page = st.number_input(
                "페이지 이동",
                min_value=1, max_value=total_pages,
                value=st.session_state.detail_page,
                step=1,
                label_visibility="collapsed"
            )
            if detail_page != st.session_state.detail_page:
                st.session_state.detail_page = detail_page
                st.rerun()
            st.markdown(
                f"<div style='text-align:center'>페이지 {detail_page}/{total_pages} (총 {total}건)</div>",
                unsafe_allow_html=True
            )

        with col_next:
            if st.button("다음 ➡", disabled=st.session_state.detail_page >= total_pages):
                st.session_state.detail_page += 1
                st.rerun()

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
                            st.write(f"- {law.strip()} : (풀이) 쉽게 풀이한 설명")

    # --------------------
    # 통계/그래프/Top 조문
    # --------------------
    if not search_results.empty:
        import altair as alt

        st.markdown("---")
        st.subheader("📊 검색 결과 통계")

        col1, col2 = st.columns(2)

        # (1) 승패 비율
        with col1:
            if "결과" in search_results.columns:
                win_count = search_results["결과"].str.contains("원고", na=False).sum()
                lose_count = search_results["결과"].str.contains("피고", na=False).sum()
                total_cases = max(win_count + lose_count, 1)
                chart_df = pd.DataFrame({
                    "결과": ["원고 승", "피고 승"],
                    "비율(%)": [win_count/total_cases*100, lose_count/total_cases*100]
                })
                chart = (
                    alt.Chart(chart_df)
                    .mark_bar()
                    .encode(
                        x=alt.X("결과", sort=None),
                        y="비율(%)",
                        tooltip=["결과", "비율(%)"]
                    )
                    .properties(height=240)
                )
                st.altair_chart(chart, use_container_width=True)

        # (2) 사건 종류 분포
        with col2:
            if "사건종류명" in search_results.columns:
                case_type_counts = (
                    search_results["사건종류명"]
                    .value_counts()
                    .rename_axis("사건종류")
                    .reset_index(name="건수")
                )
                chart2 = (
                    alt.Chart(case_type_counts)
                    .mark_bar()
                    .encode(
                        x=alt.X("사건종류", sort=None),
                        y="건수",
                        tooltip=["사건종류", "건수"]
                    )
                    .properties(height=240)
                )
                st.altair_chart(chart2, use_container_width=True)

        # (3) Top 조문 + LLM 설명
        if "참조조문" in search_results.columns:
            st.markdown("**참조조문 Top 5 (검색 결과 기준)**")
            all_laws = []
            for c in search_results["참조조문"].dropna():
                all_laws.extend([x.strip() for x in str(c).split(",")])
            law_counts = Counter(all_laws)
            law_df = (
                pd.DataFrame(law_counts.items(), columns=["조문", "횟수"])
                .sort_values(by="횟수", ascending=False)
                .head(5)
            )
            if not law_df.empty:
                law_df["이유"] = law_df["조문"].apply(lambda x: explain_law_article(x, api_key))
                st.table(law_df)

st.caption("※ 본 서비스는 참고용이며, 실제 법률 자문이 아닙니다.")
