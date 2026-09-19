import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

GROQ_MODEL = os.getenv("GROQ_MODEL","qwen/qwen3.8-27b")


if not GROQ_API_KEY:
    raise RuntimeError("找不到 GROQ_API_KEY，請確認 .env 是否設定。")

client = Groq(api_key=GROQ_API_KEY)

# System Prompt
SYSTEM_PROMPT = """
你是一個「指定餐廳」的 AI 餐廳助理。

你的工作是根據提供的餐廳資料回答使用者問題。

請遵守以下規則：

1. 只能根據提供的餐廳資料回答。
2. 不可以自行捏造餐點、價格、營業時間或其他餐廳資訊。
3. 如果提供的資料不足以回答問題，請直接說：
   「目前提供的餐廳資料不足，無法確認這項資訊。」
4. 回答使用繁體中文。
5. 回答自然、簡潔、容易理解。
6. 不要把資料庫內容原封不動全部列出來。
7. 請將相關資訊整理成自然語言。
8. 如果資訊來自顧客評論分析，請明確說明是「根據顧客評論整理」。
9. 不要假裝自己去查詢 Google Maps 或其他外部網站。
"""


# 產生 LLM 回答
def generate_answer(user_question,context_documents):
    """
    使用 Groq LLM 根據 RAG 搜尋結果回答問題。
    """

    if not user_question.strip():
        raise ValueError("使用者問題不能是空的。")

    if not context_documents:
        return ("目前沒有找到足夠的餐廳相關資料，""無法回答這個問題。")

    # 整理 RAG 文件
    context_text = ""

    for index, document in enumerate(context_documents,start=1):

        context_text += (
            f"【資料 {index}】\n"
            f"{document['content']}\n\n"
        )

    
    # 建立 User Prompt
    user_prompt = f"""
以下是從餐廳知識庫搜尋到的相關資料：
{context_text}

使用者問題：
{user_question}

請根據以上資料回答使用者問題。
如果資料不足，請不要自行推測。
"""

    # 呼叫 Groq
    response = client.chat.completions.create(
        model=GROQ_MODEL,

        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],

        temperature=0.2,
        max_completion_tokens=500
    )


    # 取得回答
    answer = (
        response.choices[0]
        .message.content
        .strip()
    )

    return answer