import os
import json

from dotenv import load_dotenv
from groq import Groq
from google_reviews import (get_google_place_reviews)
from database import (save_review_analysis)

load_dotenv()
RESTAURANT_ID = int(os.getenv("RESTAURANT_ID", "9527"))
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")

if not GROQ_API_KEY:
    raise RuntimeError("找不到 GROQ_API_KEY，請確認 .env。")

# Groq Client
client = Groq(api_key=GROQ_API_KEY)

# Review 分析 Prompt
SYSTEM_PROMPT = """
你是一個餐廳評論分析系統。

你的工作是分析 Google 餐廳評論，
並將評論轉換成結構化 JSON。

非常重要的規則：

1. 只能根據評論原文分析。
2. 不可以自行補充評論沒有提到的資訊。
3. 如果評論沒有明確提到餐點名稱，
   不可以猜測是哪一道餐點。
4. 「羹湯」不能自行推論成「土魠魚羹」。
5. 評論提到的主題必須忠實於原文。
6. sentiment 只能是：
   positive
   neutral
   negative
   mixed
7. summary 必須使用繁體中文。
8. 回傳內容必須是合法 JSON。
"""


# 分析單一評論
def analyze_review(review_text):

    if not review_text:
        return None

    user_prompt = f"""
請分析以下餐廳評論。

評論內容：

{review_text}

請回傳 JSON，格式必須完全符合：

{{
    "overall_sentiment": "positive",
    "topics": [
        {{
            "topic": "評論提到的主題",
            "sentiment": "positive",
            "summary": "根據評論整理出的簡短摘要"
        }}
    ]
}}

注意：

- topic 必須來自評論實際提到的內容。
- 不要自己補充評論沒有提到的餐點。
- 如果評論只說「羹湯」，topic 就只能是「羹湯」。
- 不要把「羹湯」自行判斷為「土魠魚羹」。
- 如果評論沒有明確提到任何餐點，可以讓 topics 聚焦在環境、服務、口味等主題。
"""

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
        response_format={"type": "json_object"},
        temperature=0.1,
        max_completion_tokens=800
    )

    result_text = (
        response
        .choices[0]
        .message
        .content
        .strip()
    )

    try:

        result = json.loads(result_text)
        return result

    except json.JSONDecodeError as e:

        print("Groq 回傳內容不是有效 JSON：")
        print(result_text)
        print("JSON Error：",e)

        return None


# 儲存單一評論分析
def save_analysis_result(analysis):

    if not analysis:
        return

    # Overall sentiment
    overall_sentiment = analysis.get("overall_sentiment", "neutral")

    save_review_analysis(
        restaurant_id=RESTAURANT_ID,
        analysis_type="review_summary",
        topic=None,
        sentiment=overall_sentiment,
        summary="Google 顧客評論整體情緒分析。",
        mention_count=1,
        confidence=0.8,
        analysis_model=GROQ_MODEL
    )

    # Topic analysis
    topics = analysis.get("topics", [])

    for topic_data in topics:
        topic = topic_data.get("topic")
        sentiment = topic_data.get("sentiment")
        summary = topic_data.get("summary")

        # 防止模型回傳空資料
        if not topic:
            continue

        if not summary:
            continue

        save_review_analysis(
            restaurant_id=RESTAURANT_ID,
            analysis_type="review_topic",
            topic=topic,
            sentiment=sentiment,
            summary=summary,
            mention_count=1,
            confidence=0.8,
            analysis_model=GROQ_MODEL
        )


# 主程式
def main():
    print("========================================")
    print("開始取得 Google Reviews")
    print("========================================")

    # 取得 Google Reviews
    place_data = get_google_place_reviews()
    reviews = place_data.get("reviews", [])

    if not reviews:
        print("目前沒有取得任何評論。")
        return

    print(f"\n取得 {len(reviews)} 則評論。")

    success_count = 0

    # 一則一則分析
    for index, review in enumerate(reviews, start=1):
        text_data = review.get("text", {})
        review_text = text_data.get("text", "")

        if not review_text:
            print(f"\n評論 {index} 沒有文字，跳過。")
            continue

        print("\n========================================")
        print(f"分析評論 {index}")
        print("========================================")
        print(f"評論內容：\n{review_text}")

        # Groq Analysis
        try:

            analysis = analyze_review(review_text)
            if not analysis:
                print("分析失敗，跳過。")
                continue

            # 顯示 AI 分析結果
            print("\nAI 分析結果：")
            print(json.dumps(analysis, ensure_ascii=False, indent=4))

            # 儲存資料庫
            save_analysis_result(analysis)
            print("分析結果已寫入 review_analysis。")

            success_count += 1

        except Exception as e:
            print(f"評論 {index} 分析失敗：" f"{e}")

    print("Google Review 分析完成")
    print(f"成功分析：" f"{success_count} / " f"{len(reviews)}")


if __name__ == "__main__":
    main()
