import os
import json

from dotenv import load_dotenv
from groq import Groq

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL","qwen/qwen3.8-27b")

if not GROQ_API_KEY:
    raise RuntimeError("找不到 GROQ_API_KEY，請確認 .env。")

SYSTEM_PROMPT = """
你是一個餐廳 Chatbot 的意圖分析器。

你的工作是分析使用者訊息，判斷：

1. 使用者想問什麼
2. 使用者的情感
3. RAG 應該搜尋哪些知識類型

請只回傳 JSON。

intent 可使用：

business_hours
open_status
menu
menu_item
feature
history
environment
recommendation
review
review_recommendation
review_environment
general

sentiment 可使用：

positive
neutral
negative
mixed

search_types 可使用：

history
feature
environment
menu_item
review
review_recommendation
review_environment

判斷規則：

open_status：
使用者想知道目前、現在、等等、某個時間點餐廳是否有營業。

例如：
「現在有開嗎」
「等等過去吃得到嗎」
「現在過去來得及嗎」
「今天還有開嗎」
「晚上六點有營業嗎」

business_hours：
使用者想知道餐廳正常營業時間、開店或關店時間。

例如：
「營業時間」
「幾點開」
「幾點關」
「今天開到幾點」
「星期日幾點營業」

重要規則：

- 「網友有什麼推薦的」不是單純 review，而是 review_recommendation。
- 「大家覺得這家店好不好」屬於 review。
- 「這家店有什麼特色」屬於 feature。
- 「以前是怎麼開始的」屬於 history。
- 「用餐環境如何」屬於 environment。
-  search_types 只能使用允許的類型，不可以自行創造新的類型。
-  不要因為出現單一關鍵字就直接判斷。
-  要理解整句話的意思。

"""

client = Groq(api_key=GROQ_API_KEY)

def classify_message(message):

    response = client.chat.completions.create(
        model=GROQ_MODEL,

        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": message
            }
        ],

        response_format={
            "type": "json_object"
        },

        temperature=0,

        max_completion_tokens=300
    )

    result_text = (
        response
        .choices[0]
        .message
        .content
        .strip()
    )

    try:

        return json.loads(
            result_text
        )

    except json.JSONDecodeError:

        print(
            "Intent JSON 解析失敗：",
            result_text
        )

        return {
            "intent": "general",
            "sentiment": "neutral",
            "search_types": []
        }