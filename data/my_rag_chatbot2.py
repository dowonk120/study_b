# 0811 도원 수정 : 전체 파일 합친 후 임베딩하도록 변경, 프롬프트 변경

from openai import OpenAI
import streamlit as st
from langchain_core.messages.chat import ChatMessage
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableMap
import os

from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings
from langchain_core.prompts import loading
from langchain_core.prompts import ChatPromptTemplate
import base64
from PIL import Image
import io
from langchain_experimental.text_splitter import SemanticChunker
from operator import itemgetter
from langchain_community.retrievers import TavilySearchAPIRetriever
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv
import zipfile
import pandas as pd
from langchain_core.documents import Document
from enum import Enum
from pydantic import BaseModel, Field


# 캐시 폴더 지정
if os.path.isdir("./mycache") == False:
    os.mkdir("./mycache")

if os.path.isdir("./mycache/files") == False:
    os.mkdir("./mycache/files")

if os.path.isdir("./mycache/embedding") == False:
    os.mkdir("./mycache/embedding")

store = LocalFileStore("./mycache/embedding")


import os

os.listdir("./mycache/embedding").__len__()
# 합쳐질 DataFrame들을 담을 리스트
all_dfs = []

# 처리할 ZIP 파일 목록
zip_files = ["result_1_merged.zip", "result_2_merged.zip", "result_3_merged.zip"]


# 세개 다 풀어서 합치고 df를 저장해서 바로 불러오는 걸로 변경
for zip_file in zip_files:
    with zipfile.ZipFile(zip_file) as z:
        for filename in z.namelist():
            if filename.endswith(".csv"):
                with z.open(filename) as f:
                    df = pd.read_csv(io.StringIO(f.read().decode("utf-8")))
                    all_dfs.append(df)

# 하나의 DataFrame으로 병합
merged_df = pd.concat(all_dfs, ignore_index=True)
merged_df.drop("Unnamed: 0", axis=1, inplace=True)
merged_df = merged_df.dropna(subset=["참조조문"])
merged_df.reset_index(drop=True, inplace=True)
# merged_df.columns
# path_df = "./result1_df.csv"
# df = pd.read_csv(path_df, encoding="utf-8")
sel_df = merged_df[
    ["참조조문", "배상책임", "주제", "재판부_판단", "결과", "요약", "사건번호"]
]

client = OpenAI()


if "chain" not in st.session_state:
    st.session_state["chain"] = None

if "messages" not in st.session_state:
    st.session_state["messages"] = []


def add_message(role, message):
    st.session_state["messages"].append({"role": role, "content": message})


st.title("RAG 기반 법률 상담 챗봇")

import hashlib
import os
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings.cache import CacheBackedEmbeddings
from langchain.docstore.document import Document
from langchain.storage import LocalFileStore


@st.cache_resource(show_spinner="업로드 파일 처리 중 기다리세요")
def processing(sel_df):
    """업로드된 CSV 기반 retriever 생성 (임베딩 캐싱 + FAISS 재사용)"""

    # 👉 DataFrame 전체 해시로 캐시 경로 결정
    df_hash = hashlib.md5(
        pd.util.hash_pandas_object(sel_df, index=True).values
    ).hexdigest()
    faiss_path = f"./mycache/vectorstore/{df_hash}"

    # 👉 FAISS 벡터 DB가 이미 있으면 로드
    if os.path.exists(faiss_path):
        vectorstore = FAISS.load_local(
            faiss_path, OpenAIEmbeddings(), allow_dangerous_deserialization=True
        )
    else:
        # 👉 문서 리스트 생성
        docs = []
        for i, row in sel_df.iterrows():
            row_dict = row.to_dict()

            # 각 행을 고유하게 구분하기 위한 key 생성 (ex: 행의 전체 해시)
            row_hash = hashlib.md5(str(row_dict).encode()).hexdigest()
            doc = Document(page_content=str(row_dict), metadata={"row_id": row_hash})
            docs.append(doc)

        # 👉 캐시 + 임베딩 설정
        embedding = OpenAIEmbeddings()
        store = LocalFileStore("./mycache/embedding")  # 중복 캐시 저장 방지
        cached_embedder = CacheBackedEmbeddings.from_bytes_store(
            underlying_embeddings=embedding, document_embedding_cache=store
        )

        # 👉 FAISS 벡터 DB 생성
        vectorstore = FAISS.from_documents(docs, cached_embedder)

        # 👉 FAISS 저장
        vectorstore.save_local(faiss_path)

    # 👉 retriever 반환
    retriever = vectorstore.as_retriever(
        search_type="mmr", search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.7}
    )
    return retriever


join_docs = RunnableLambda(lambda docs: "\n".join(doc.page_content for doc in docs))


def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message["role"]).write(chat_message["content"])


def create_chain(retriever):
    prompt_text = {
        "_type": "prompt",
        "template": (
            "당신은 한국어로 법률 상담을 제공하는 전문 AI 어시스턴트입니다.\n"
            "⛔ 반드시 유의하세요: context에 없는 내용은 절대 생성하지 마세요.\n"
            "모든 답변은 반드시 제공된 context에 기반해야 하며, 부족한 경우에는 '정확히 알 수 없습니다'라고 명확히 밝혀야 합니다.\n\n"
            "🧭 다음의 절차와 형식을 따르세요:\n"
            "1️⃣ 질문의 유형을 분류하세요: 정보 탐색 / 사건 상담 / 법 해석 / 예방 조언 / 기타\n"
            "2️⃣ 질문자의 상황 및 맥락(chat_history)을 반영해 맞춤형으로 답변하세요.\n"
            "3️⃣ context에서 관련 정보를 최대한 활용해 실무적인 답변을 구성하세요.\n"
            "5️⃣ 판례나 사건번호가 있다면, 반드시 아래 형식을 따르세요:\n"
            "### ✅ 판례 예시\n"
            "- 사건번호: [사건번호]\n"
            "- 판결 내용 요약: [간략한 설명]\n"
            "👉 더 많은 판례 확인: https://portal.scourt.go.kr/pgp/index.on?m=PGP1011M01&l=N&c=900\n"
            "📌 판례나 사건번호가 포함된 정보가 context에 없다면, 해당 섹션은 생략하세요."
            "6️⃣ 마크다운 형식으로 이모지와 헤더를 활용해 시각적으로 구성하세요.특히 헤더에는 이모티콘으로 구분자를 넣어주세요\n"
            "7️⃣ 민감하거나 감정적인 사안은 공감과 배려를 포함한 언어로 표현하세요.\n"
            "8️⃣ 복잡한 질문에는 답변 마지막에 간결한 요약을 포함하세요.\n"
            "9️⃣ 마지막에 사용자가 후속 질문을 할 수 있도록 유도하세요.\n"
            "🔟 출처가 context 기반인지, 모델의 사전 지식인지, 둘다 인지 반드시 명시하세요:\n"
            "<information>\n{context}\n</information>\n\n"
            "#질문:\n{question}\n\n"
            "#이전 대화 내용:\n{chat_history}\n\n"
            "#답변:\n"
        ),
        "input_variables": ["question", "context", "chat_history"],
    }

    prompt = loading.load_prompt_from_config(prompt_text)
    # llm = ChatOllama(model='gemma:7b', temperature=0)
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


retriever = processing(sel_df)
chain = create_chain(retriever)
st.session_state["chain"] = chain


user_input = st.chat_input("무엇이든 물어보세요")
print_messages()

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    history_str = "\n".join(
        f"{m['role']} : {m['content']}" for m in st.session_state.messages
    )
    # print("---------------")
    # print(history_str)
    # print("---------------")
    chain = st.session_state["chain"]

    # print(payload)
    if chain is not None:
        # 히스토리 (과거 질문과 대답 목록 )

        st.chat_message("user").write(user_input)
        # print("===>")
        # print(user_input)
        payload = {"question": user_input, "chat_history": history_str}
        response = chain.stream(payload)
        # add_message('user' , user_input)
        # print(response)

        with st.chat_message("assistant"):
            container = st.empty()

            ai_answer = ""
            for token in response:
                ai_answer += token
                container.markdown(ai_answer)
            add_message("assistant", ai_answer)
