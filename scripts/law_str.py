import streamlit as st
import pandas as pd
import zipfile, io, re
from collections import Counter


st.set_page_config(layout="wide")
st.title("⚖️ 법률 사건 검색 및 예상 결과 시뮬레이션")

# --------------------
# 1. CSV 로드
# --------------------
@st.cache_data
def load_data():
    zip_files = ["../data/result_1_merged.zip", "../data/result_2_merged.zip", "../data/result_3_merged.zip"]
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

# --------------------
# 2. Stopwords 자동 생성
# --------------------
@st.cache_data
def build_stopwords(df, top_n=100):
    all_words = []
    target_cols = ["판시사항","판결요지","참조조문","사건명","요약","판례내용","쟁점","재판부_판단","결과"]
    for col in target_cols:
        texts = df[col].dropna().astype(str)
        for text in texts:
            all_words.extend(re.findall(r'[가-힣a-zA-Z0-9]{2,}', text))
    common_words = [w for w, _ in Counter(all_words).most_common(top_n)]
    return set(common_words)

auto_stopwords = build_stopwords(df)
extra_stopwords = {"어제","먹고","쳤어","어떻게","되는거야","했는데","인데","있어","없어", "하다가","넘어서","같아","사람을"}
stopwords = auto_stopwords.union(extra_stopwords)

# --------------------
# 3. 페이지 레이아웃
# --------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("본인 입장 및 사건 정보 입력")

    # 입력
    position = st.selectbox("본인 입장 선택", ["원고", "피고", "검사", "기타"])
    situation = st.text_area("사건 내용을 구체적으로 입력하세요", height=150)
    damage = st.number_input("피해 금액 입력 (만원 단위)", min_value=0, step=100)
    settlement = st.radio("합의 여부", ["합의 안 함", "합의 함"])

    # 검색 + 시뮬레이션 버튼
    if st.button("검색 및 예상 결과"):
        if situation.strip() == "":
            st.warning("사건 내용을 입력해 주세요.")
        else:
            # --------------------
            # 키워드 추출 (Stopwords 적용)
            # --------------------
            from konlpy.tag import Okt
            okt = Okt()
            nouns = okt.nouns(situation)  # 사건 내용에서 핵심 명사만 추출
            priority_words = ['마약', '성폭행', '음주운전', '사기', '살인', '폭행']  # 사건 종류별 예시
            nouns = okt.nouns(situation)
            keywords = [w for w in nouns if w in priority_words]            
            query_regex = '|'.join(keywords) if keywords else ".*"
            st.write(f"검색 키워드: {keywords}")

            # --------------------
            # 사건 검색
            # --------------------
            search_results = df[
                df["판시사항"].str.contains(query_regex, na=False) |
                df["사건종류명"].str.contains(query_regex, na=False) |
                df["판결요지"].str.contains(query_regex, na=False) |
                df["사건명"].str.contains(query_regex, na=False) |
                df["판례내용"].str.contains(query_regex, na=False) |
                df["판결유형"].str.contains(query_regex, na=False) |
                df["주제"].str.contains(query_regex, na=False) |
                df["쟁점"].str.contains(query_regex, na=False) |
                df["당사자1_역할"].str.contains(query_regex, na=False) |
                df["당사자1_주장"].str.contains(query_regex, na=False) |
                df["당사자2_역할"].str.contains(query_regex, na=False) |
                df["당사자2_주장"].str.contains(query_regex, na=False) |
                df["재판부_판단"].str.contains(query_regex, na=False) |
                df["결과"].str.contains(query_regex, na=False) |
                df["요약"].str.contains(query_regex, na=False)
            ]
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

            # 승률 조정 (피해금액 + 합의 여부)
            rate = base_win_rate
            if position == "원고" and settlement == "합의 함":
                rate += 0.1
            elif position == "피고" and settlement == "합의 안 함":
                rate -= 0.1
            rate -= damage / 10000  # 피해금액 영향
            rate = max(0, min(1, rate))

            st.write(f"예상 승소 확률: {rate:.1%}")
            st.write(f"예상 배상액 범위: {damage*0.5:.0f} ~ {damage*1.5:.0f} 만원")
            st.progress(rate)

            # 위험도 표시
            if rate < 0.3:
                st.error("실형 가능성이 높습니다!")
            elif rate < 0.6:
                st.warning("위험도가 중간입니다.")
            else:
                st.success("승소 가능성이 높습니다.")

            # 사건 상세보기
            for idx, row in search_results.iterrows():
                with st.expander(f"{row['사건번호']} | {row['사건명']} | {row['결과']}"):
                    st.write(f"사건 종류: {row['사건종류명']}")
                    st.write(f"판시사항: {row['판시사항']}")
                    st.write(f"주요 쟁점: {row['쟁점']}")
                    st.write(f"판결유형: {row['판결유형']}")
                    st.write(f"배상책임: {row['배상책임']}")
                    st.write(f"요약: {row['요약']}")
                    if pd.notna(row["참조조문"]):
                        laws = [x.strip() for x in row["참조조문"].split(",")]
                        st.write("참조조문:")
                        for law in laws:
                            st.write(f"- {law}: (원문) {law} 조문 내용 / (풀이) 쉽게 풀이한 설명")

# --------------------
# 오른쪽: 통계 및 Top 조문
# --------------------
with col2:
    st.subheader("판례 통계")
    case_counts = df["사건종류명"].value_counts()
    st.markdown("**사건 종류 분포**")
    st.bar_chart(case_counts)

    st.markdown("**참조조문 Top 5**")
    all_laws = []
    for c in df["참조조문"].dropna():
        all_laws.extend([x.strip() for x in c.split(",")])
    law_counts = Counter(all_laws)
    law_df = pd.DataFrame(law_counts.items(), columns=["조문", "횟수"]).sort_values(by="횟수", ascending=False).head(5)
    st.table(law_df)
