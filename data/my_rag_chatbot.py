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
with zipfile.ZipFile("result_1_merged.zip") as z:
    for filename in z.namelist():           # ZIP 안 모든 파일 이름
        if filename.endswith(".csv"):       # CSV만 선택
            with z.open(filename) as f:
                df = pd.read_csv(io.StringIO(f.read().decode("utf-8")))
                
df.drop('Unnamed: 0', axis = 1, inplace = True)   
# df.columns     

sel_df = df[['배상책임', '주제', '쟁점', '당사자1_역할', '당사자1_주장', '당사자2_역할', '당사자2_주장','재판부_판단', '결과', '요약']]
# 일단 테스트용 5개만
sel_df = sel_df.head()

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
        vectorstore = FAISS.load_local(faiss_path, OpenAIEmbeddings())
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
    retriever = vectorstore.as_retriever()
    return retriever

join_docs = RunnableLambda(lambda docs: "\n".join(doc.page_content for doc in docs))

def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message["role"]).write(chat_message["content"])


def create_chain(retriever,vectorstore):

    prompt_text = {
        "_type": "prompt",
        "template": (
            "You are an assistant for question-answering tasks.\n"
            "Use the following retrieved information to answer the question.\n"
            "If you don't know the answer based on the information, say you don't know.\n"
            "Answer in Korean.\n\n"
            "<information>\n{context}\n</information>\n\n"
            "#Question:\n{question}\n\n"
            "#Answer:\n#chat_history:\n{chat_history}"
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


retriever,vectorstore = processing(sel_df)
chain = create_chain(retriever,vectorstore)
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



