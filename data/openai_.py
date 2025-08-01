import pandas as pd
import zipfile

dfs = {}  # 파일 이름 → DataFrame 저장 딕셔너리

with zipfile.ZipFile("result.zip") as z:
    for filename in z.namelist():  # ZIP 안 모든 파일 이름
        if filename.endswith(".csv"):  # CSV만 선택
            with z.open(filename) as f:
                dfs[filename] = pd.read_csv(f)

# 결과
for name, df in dfs.items():
    print(name, df.shape)


# pd.set_option('display.max_colwidth', None)  ----------------------------데이터 프레임에서 판례문 모든 글자 보여주는 설정

df = dfs["result_1.csv"]  # ------------------------각자 맡으신걸로 변경해주시면됩니다.

import re
import pandas as pd
import asyncio
from concurrent.futures import ThreadPoolExecutor
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from tqdm import tqdm


from langsmith import traceable
from dotenv import load_dotenv
import os

# ---------- LANGSMITH_API_KEY 확인 ----------

load_dotenv()
print(os.environ.get("LANGSMITH_API_KEY"))

# ---------- 1️⃣ 모델 설정 ----------
llm = ChatOpenAI(model="gpt-4o-mini-2024-07-18", temperature=0)


# ---------- 2️⃣ 결과 스키마 ----------
class LegalRoleSummary(BaseModel):
    배상책임: str = Field(description="배상 책임이 누구에게 있는지: 원고 / 피고 / 없음")
    주제: str = Field(description="주제 키워드 예: 부동산, 초상권 등")
    쟁점: str = Field(description="쟁점 키워드 예: 임대차 계약 해지, 명예훼손 등")
    당사자1_역할: str = Field(
        description="첫 번째 당사자의 역할명: 원고 / 검사 / 신청인 등"
    )
    당사자1_주장: str = Field(description="첫 번째 당사자의 주장 요약")
    당사자2_역할: str = Field(
        description="두 번째 당사자의 역할명: 피고 / 피고인 / 상대방 등"
    )
    당사자2_주장: str = Field(description="두 번째 당사자의 주장 요약")
    재판부_판단: str = Field(description="법원의 판단 요약")
    결과: str = Field(description="결과 키워드 예: 원고 승소, 피고 일부 패소 등")
    요약: str = Field(description="사건 전체 요약 한 문장")


parser = PydanticOutputParser(pydantic_object=LegalRoleSummary)


# ---------- 3️⃣ 전처리 함수 ----------
def preprocess_text(text):
    text = re.sub(r"【원고[^】]*】", "", text)
    text = re.sub(r"【피고[^】]*】", "", text)
    text = re.sub(r"【원심판결[^】]*】", "", text)
    reason_match = re.search(r"【이\s*유】(.+?)(판결한다|판결함|대법관)", text, re.S)
    if reason_match:
        text = reason_match.group(1).strip()
    return text


# ---------- 4️⃣ 프롬프트 ----------
system_prompt = """
당신은 판례 요약 전문가입니다. 아래 사건 판례내용을 읽고 JSON으로 요약하세요.

{row_info}

- 배상 책임: 원고 / 피고 / 없음
- 주제 키워드: 한 단어 (예: 초상권, 손해배상)
- 쟁점 키워드: 한 단어 또는 짧은 구 (예: 임대차 해지)
- 당사자1 역할: (예: 원고, 검사)
- 당사자1 주장: 짧게 요약
- 당사자2 역할: (예: 피고, 피고인)
- 당사자2 주장: 짧게 요약
- 재판부 판단: 간결하게 요약
- 결과: 간단히 (예: 원고 승소)
- 요약: 한 문장
"""

prompt = PromptTemplate(
    input_variables=["row_info", "format_instructions"],
    template=system_prompt + "\n{format_instructions}",
)

chain = LLMChain(llm=llm, prompt=prompt, output_parser=parser)
partial_vars = {"format_instructions": parser.get_format_instructions()}

# ---------- 5️⃣ 비동기 처리 함수 ----------

# 에러났을 경우 사건번호를 리스트에 저장
error_list = []


def process_row(idx, case_num, case_text):
    clean_text = preprocess_text(case_text)
    try:
        result = chain.invoke({"row_info": clean_text, **partial_vars})
        summary = result["text"].model_dump()
        summary["case_num"] = case_num
        # return result['text'].model_dump() # Pydantic 모델 인스턴스를 딕셔너리로 변환
        return summary
    except Exception as e:
        error_list.append(case_num)
        return {"case_num": case_num, "error": str(e)}


async def process_batch(df, start_index=0, batch_size=1000, max_workers=5):
    end_index = min(start_index + batch_size, len(df))
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(
                executor,
                process_row,
                idx,
                df.loc[idx, "사건번호"],
                df.loc[idx, "판례내용"],
            )
            for idx in range(start_index, end_index)
        ]
        for f in tqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc="🔄 병렬 요약 진행중",
            unit="건",
        ):
            result = await f
            results.append(result)

    output_file = f"legal_summary_{start_index}_{end_index - 1}.csv"
    pd.DataFrame(results).to_csv(output_file, index=False)
    print(f"✅ {output_file} 저장 완료 ({start_index} ~ {end_index - 1}건)")
    print("-------------------------")
    print(f"❌ 에러 사건번호 목록 : {error_list}")


# asyncio.run(process_batch(df, start_index=0, batch_size=5, max_workers=5)) #---------실행 코드
# start_index값이랑 batch_size만 변경해서 원하는 건수만큼 돌려주시면됩니다.
# 에러나면 밑에 코드 사용하세요.


import nest_asyncio  # --------------- 주피터 실행코드입니다.

nest_asyncio.apply()
await process_batch(df, start_index=600, batch_size=400, max_workers=5)

# 확인용
split_df = pd.read_csv("./legal_summary_600_.csv")
split_df.shape
split_df.head()
