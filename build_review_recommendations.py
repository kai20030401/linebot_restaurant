import os
import json

from collections import defaultdict
from dotenv import load_dotenv
from groq import Groq
from database import (
    get_review_analysis,
    save_recommendation_knowledge,
    delete_review_recommendation_knowledge
)

load_dotenv()
RESTAURANT_ID = int(os.getenv("RESTAURANT_ID","9527"))
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL","qwen/qwen3.8-27b")

if not GROQ_API_KEY:
    raise RuntimeError("找不到 GROQ_API_KEY，請確認 .env。")

client = Groq(api_key=GROQ_API_KEY)


# Step 1：聚合 Review Analysis
def aggregate_review_analysis(review_analysis):

    topics = defaultdict(
        lambda: {
            "mention_count": 0,
            "positive_count": 0,
            "neutral_count": 0,
            "negative_count": 0,
            "mixed_count": 0,
            "summaries": []
        }
    )

    for item in review_analysis:
        topic = item["topic"]

        if not topic:
            continue

        data = topics[topic]

        # mention count
        data["mention_count"] += (item["mention_count"] or 1)

        # sentiment count
        sentiment = item["sentiment"]

        if sentiment == "positive":
            data["positive_count"] += 1

        elif sentiment == "neutral":
            data["neutral_count"] += 1

        elif sentiment == "negative":
            data["negative_count"] += 1

        elif sentiment == "mixed":
            data["mixed_count"] += 1
    
        # summary
        summary = item["summary"]

        if summary:
            data["summaries"].append(summary)

    return dict(topics)


# Step 2：整理給 Groq 的資料
def build_llm_input(aggregated_data):
    result = []

    for topic, data in aggregated_data.items():

        result.append(
            {
                "topic": topic,
                "mention_count": data["mention_count"],
                "positive_count": data["positive_count"],
                "neutral_count": data["neutral_count"],
                "negative_count": data["negative_count"],
                "mixed_count": data["mixed_count"],
                "summaries": data["summaries"]
            }
        )

    return result


# Step 3：Groq 建立推薦知識
def generate_recommendation_knowledge(aggregated_data):
    llm_data = build_llm_input(aggregated_data)

    user_prompt = f"""
以下是從 Google 顧客評論中整理出的分析資料：

{json.dumps(
    llm_data,
    ensure_ascii=False,
    indent=2
)}

請根據這些資料建立「餐廳顧客評論推薦知識」。

重要規則：

1. 只能根據提供的資料。
2. 不可以自行增加評論沒有提到的餐點。
3. 不可以把相似但不同的餐點自行合併。
4. 「排骨」不能自行判斷成「鐵路排骨飯」或「排骨酥麵」。
5. 目前資料樣本只有少量評論，因此不要使用：
   「最受歡迎」、「所有顧客都推薦」等過度概括的說法。
6. 可以使用：
   「目前取得的評論中有顧客提到」
   「在目前評論樣本中」
   「部分顧客提到」
7. 請使用繁體中文。
8. 請回傳 JSON。

格式：

{{
    "recommendations": [
        {{
            "title": "推薦主題",
            "content": "根據目前評論資料整理出的推薦資訊",
            "confidence": 0.0
        }}
    ]
}}
"""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "你是一個餐廳評論分析與推薦知識整理系統。"
                    "只能根據提供的資料回答。"
                    "不要自行推測。"
                )
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
        max_completion_tokens=1000
    )

    result_text = (
        response
        .choices[0]
        .message
        .content
        .strip()
    )

    try:
        return json.loads(result_text)

    except json.JSONDecodeError:
        print("Groq 回傳不是有效 JSON:")
        print(result_text)

        return None


def main():    
    # 取得 Review Analysis
    review_analysis = get_review_analysis(RESTAURANT_ID)

    if not review_analysis:
        print("目前沒有 Review Analysis 資料。")
        return

    print(f"\n取得 {len(review_analysis)} 筆分析結果。")

    # Aggregation
    aggregated_data = aggregate_review_analysis(review_analysis)
    print("\n========================================")
    print("評論聚合結果")
    print("========================================")

    for topic, data in aggregated_data.items():
        print(f"\n主題：{topic}")
        print(f"提及次數：" f"{data['mention_count']}")
        print(f"正面：" f"{data['positive_count']}")
        print(f"中立：" f"{data['neutral_count']}")
        print(f"負面：" f"{data['negative_count']}")

    # Groq
    recommendation_result = (
        generate_recommendation_knowledge(
            aggregated_data
        )
    )

    if not recommendation_result:
        print("無法建立 Recommendation Knowledge。")
        return

    recommendations = (recommendation_result.get("recommendations", []))

    print("\n========================================")
    print("AI Recommendation Knowledge")
    print("==========================================")
    print(json.dumps(recommendation_result, ensure_ascii=False, indent=4))

    # 刪除舊的 AI Recommendation
    delete_review_recommendation_knowledge(RESTAURANT_ID)

    # 寫入 restaurant_knowledge
    saved_count = 0

    for recommendation in recommendations:
        title = recommendation.get("title")
        content = recommendation.get("content")
        confidence = recommendation.get("confidence",0.7)

        if not title or not content:
            continue

        knowledge_id = (
            save_recommendation_knowledge(
                restaurant_id=RESTAURANT_ID,
                title=title,
                content=content,
                confidence=confidence
            )
        )

        print(f"\n建立 Recommendation Knowledge：" f"{knowledge_id}")
        print(f"Title：{title}")
        print(f"Content：{content}")

        saved_count += 1

    print("Recommendation Knowledge 建立完成")
    print(f"成功建立：{saved_count} 筆")

if __name__ == "__main__":
    main()

