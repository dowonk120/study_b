import re
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import time
from tqdm import tqdm  # ✅ 진행률 표시

# ---------- 1️⃣ 모델 설정 ----------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ---------- 2️⃣ 결과 스키마 ----------
class LegalRoleSummary(BaseModel):
    배상책임: str = Field(description="배상 책임이 누구에게 있는지: 원고 / 피고 / 없음")
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

# ---------- 3️⃣ 전처리 함수 ----------
def preprocess_text(text):
    text = re.sub(r'【원고[^】]*】', '', text)
    text = re.sub(r'【피고[^】]*】', '', text)
    text = re.sub(r'【원심판결[^】]*】', '', text)
    reason_match = re.search(r'【이\s*유】(.+?)(판결한다|판결함|대법관)', text, re.S)
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

# ---------- 5️⃣ 사용자 설정 ----------
start_index = 0      # ✅ 시작 위치 (다음 실행 시 1000, 2000으로 변경)
batch_size = 1000    # ✅ 처리할 건수
end_index = min(start_index + batch_size, len(df))

# ---------- 6️⃣ 실행 ----------
results = []
for idx in tqdm(range(start_index, end_index), desc="🔄 요약 진행중", unit="건"):
    case_text = df.loc[idx, "판례내용"]
    clean_text = preprocess_text(case_text)

    try:
        result = chain.invoke({"row_info": clean_text, **partial_vars})
        results.append(result['text'].model_dump())
    except Exception as e:
        results.append({"error": str(e)})
    
    time.sleep(0.3)

# ---------- 7️⃣ CSV 저장 ----------
output_file = f"legal_summary_{start_index}_{end_index}.csv"
pd.DataFrame(results).to_csv(output_file, index=False)
print(f"✅ {output_file} 저장 완료 ({start_index} ~ {end_index-1}건)")
