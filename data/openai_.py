
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import pandas as pd

llm = ChatOpenAI(model="gpt-3.5-turbo")

class LegalRoleSummary(BaseModel):
    배상여부: str = Field(description="배상 여부: 있음 / 없음")
    주제: str = Field(description="주제 키워드 예: 부동산, 초상권 등")
    쟁점: str = Field(description="쟁점 키워드 예: 임대차 계약 해지, 명예훼손 등")
    당사자1_역할: str = Field(description="첫 번째 당사자의 역할명: 원고 / 검사 / 신청인 등")
    당사자1_주장: str = Field(description="첫 번째 당사자의 주장 요약")
    당사자2_역할: str = Field(description="두 번째 당사자의 역할명: 피고 / 피고인 / 상대방 등")
    당사자2_주장: str = Field(description="두 번째 당사자의 주장 요약")
    재판부_판단: str = Field(description="법원의 판단 요약")
    결과: str = Field(description="결과 키워드 예: 원고 승소, 피고 일부 패소 등")
    요약: str = Field(description="사건 전체 요약 한 문장")

parser = PydanticOutputParser(pydantic_object=LegalRoleSummary)

system_prompt = """
당신은 판례 요약 전문가입니다. 아래 사건 정보를 읽고 JSON으로 요약하세요.

{row_info}

- 배상 여부: 있음 / 없음
- 주제 키워드: 한 단어 (예: 초상권, 손해배상)
- 쟁점 키워드: 한 단어 또는 짧은 구 (예: 임대차 해지)
- 당사자1 역할: (예: 원고, 검사)
- 당사자1 주장: 짧게 요약
- 당사자2 역할: (예: 피고, 피고인)
- 당사자2 주장: 짧게 요약
- 재판부 판단: 간결하게 요약
- 결과: 간단히 (예: 원고 승소)
- 요약: 한 문장

💡 역할명은 문맥에 따라 정확히 판단하세요.
"""

prompt = PromptTemplate(
    input_variables=["row_info", "format_instructions"],
    template=system_prompt + "\n{format_instructions}",
)

chain = LLMChain(llm=llm, prompt=prompt, output_parser=parser)

partial_vars = {"format_instructions": parser.get_format_instructions()}

# 예시로 한 건만 처리
sam_df = drop_df.head(1)

def invoke_and_parse(text):
    return chain.invoke({"row_info": str(text), **partial_vars})

result = sam_df["판시사항"].apply(invoke_and_parse)
print(result[0])  # dict 형태 출력

result.item()

result[0]['text'].model_dump()



