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

if os.path.isdir("./mycache") == False:
    os.mkdir("./mycache")

if os.path.isdir("./mycache/files") == False:
    os.mkdir("./mycache/files")
    
if os.path.isdir("./mycache/embedding") == False:
    os.mkdir("./mycache/embedding")
    
store = LocalFileStore("./mycache/embedding")

# 세개 다 풀어서 합치고 df를 저장해서 바로 불러오는 걸로 변경
# with zipfile.ZipFile("result_1_merged.zip") as z:
#     for filename in z.namelist():           # ZIP 안 모든 파일 이름
#         if filename.endswith(".csv"):       # CSV만 선택
#             with z.open(filename) as f:
#                 df = pd.read_csv(io.StringIO(f.read().decode("utf-8")))
                
# df.drop('Unnamed: 0', axis = 1, inplace = True)   
# df.columns     


path_df = "./result1_df.csv"
df = pd.read_csv(path_df, encoding="utf-8")
sel_df = df[['참조조문','배상책임', '주제','재판부_판단', '결과','요약']]
# 일단 테스트용 5개만
# sel_df = sel_df.head()

client = OpenAI()


if "chain" not in st.session_state:
    st.session_state["chain"] = None

if "messages" not in st.session_state:
    st.session_state["messages"] = []


def add_message(role, message):
    st.session_state["messages"].append({"role": role, "content": message})

st.title("RAG 기반 챗봇")

import hashlib
import os
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings.cache import CacheBackedEmbeddings
from langchain.docstore.document import Document
from langchain.storage import LocalFileStore

@st.cache_resource(show_spinner="업로드 파일 처리중 기다리세요")
def processing(sel_df):
    '''업로드된 csv 기반 retriever 생성 (임베딩 캐싱 + FAISS 재사용)'''
    
    # 👉 DataFrame 해시로 유일한 캐시 키 생성
    df_hash = hashlib.md5(pd.util.hash_pandas_object(sel_df, index=True).values).hexdigest()
    faiss_path = f"./mycache/vectorstore/{df_hash}"

    # 👉 FAISS가 이미 있다면 로드
    if os.path.exists(faiss_path):
        vectorstore = FAISS.load_local(faiss_path, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    else:
        # Document 리스트 만들기
        docs = [Document(page_content=str(row.to_dict())) for _, row in sel_df.iterrows()]

        # 임베딩 + 캐시 설정
        embedding = OpenAIEmbeddings()
        
        store = LocalFileStore("./mycache/embedding")  # 기존 경로
        cached_embedder = CacheBackedEmbeddings.from_bytes_store(
            underlying_embeddings=embedding,
            document_embedding_cache=store
        )

        # 벡터 DB 생성
        vectorstore = FAISS.from_documents(docs, embedding=cached_embedder)

        # 👉 로컬 저장소에 저장 (해시 기반)
        vectorstore.save_local(faiss_path)

    # 검색기 반환
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.7}
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
        "- (여기에 관련 법률 조문을 bullet point로 작성)\n\n"
        "답변후에 혹시 이 주제에 대해 더 궁금한 것이 있으신가요?라고 물어봐줘"
        "답변후에 혹시 이 주제에 대해 더 궁금한 것이 있으신가요?라고 물어봐줘"
        
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


user_input = st.chat_input("질문을 하세요")
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



