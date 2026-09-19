import os
import json

from datetime import datetime, time
from zoneinfo import ZoneInfo
from flask import Flask, request, abort
from dotenv import load_dotenv
from intent import classify_message

from linebot.v3 import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import (
    Configuration,
    ApiClient,
    MessagingApi,
    ReplyMessageRequest,
    TextMessage
)
from linebot.v3.webhooks import MessageEvent, TextMessageContent

from embedding import create_embedding
from llm import generate_answer

from database import (
    get_or_create_chat_session,
    save_chat_message,
    get_restaurant_info,
    get_restaurant_hours,
    get_menu_items,
    search_restaurant_documents
)

#dotenv_path = os.path.join(os.path.dirname(__file__), '.gitignore', '.env')
#load_dotenv(dotenv_path, override=True) #本機開發使用這個抓取env檔

load_dotenv()
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY")

# LINE 設定
CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
configuration = Configuration(
    access_token=CHANNEL_ACCESS_TOKEN
)
handler = WebhookHandler(CHANNEL_SECRET)

# 餐廳設定
RESTAURANT_ID = int(os.getenv("RESTAURANT_ID", "9527"))
GOOGLE_MAPS_URL = os.getenv("GOOGLE_MAPS_URL")
FACEBOOK_URL = os.getenv("FACEBOOK_URL")
OFFICIAL_WEBSITE_URL = os.getenv("OFFICIAL_WEBSITE_URL")


@app.route("/callback", methods=["POST"])
def callback():
    signature = request.headers.get("X-Line-Signature")
    body = request.get_data(as_text=True)
    app.logger.info(f"Request body: {body}")

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        print("Invalid signature")
        abort(400)

    return "OK"


# 接收 LINE 文字訊息
@handler.add(MessageEvent,message=TextMessageContent)
def handle_message(event):
    # 取得 LINE 使用者 ID
    user_id = event.source.user_id
    # 使用者訊息
    user_message = event.message.text.strip()
    print("LINE User:", user_id)
    print("Message:", user_message)

    # 建立 / 取得 Chat Session
    session_id = get_or_create_chat_session(
        restaurant_id=RESTAURANT_ID,
        user_identifier=user_id
    )

    # 儲存使用者訊息
    save_chat_message(
        session_id=session_id,
        role="user",
        content=user_message
    )

    # Bot 回覆邏輯
    reply_text, retrieved_document_ids = simple_chatbot(user_message)

    # 儲存 Bot 回覆
    save_chat_message(
        session_id=session_id,
        role="assistant",
        content=reply_text,
        model="groq_rag_v1",
        retrieved_document_ids=json.dumps(retrieved_document_ids)
    )

    # 回覆 LINE
    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)
        line_bot_api.reply_message(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text=reply_text)]
            )
        )


# 判斷使用者問題，再決定要用資料庫或是llm回覆
def simple_chatbot(message):
    message = message.lower()

    # 明確結構化問題
    if ("地址" in message
        or "google maps" in message):
        restaurant = get_restaurant_info(RESTAURANT_ID)

        if restaurant:
            return (
                f"📍 {restaurant['name']}\n"
                f"地址：{restaurant['address']}\n\n"
                f"🗺️ Google Maps\n"
                f"{GOOGLE_MAPS_URL}"
            ), []

        return "目前找不到餐廳資料。", []

    if ("聯絡電話" in message
        or "連絡電話" in message):
        restaurant = get_restaurant_info(RESTAURANT_ID)

        if restaurant:
            return (
                f"☎️ {restaurant['name']}\n"
                f"電話：{restaurant['phone']}"
            ), []

        return "目前找不到電話資訊。", []

    if ("facebook" in message
        or "fb" in message
        or "臉書" in message
        or "粉絲專頁" in message):

        if FACEBOOK_URL:
            return (
                f"📘 深海深美食 Facebook\n"
                f"{FACEBOOK_URL}"
            ), []

        return "目前沒有 Facebook 資訊。", []

    if ("官網" in message
        or "官方網站" in message):

        if OFFICIAL_WEBSITE_URL:
            return (
                f"🌐 深海深美食官方網站\n"
                f"{OFFICIAL_WEBSITE_URL}"
            ), []

        return "目前沒有官方網站資訊。", []

    if ("現在有開嗎" in message
        or "現在開嗎" in message
        or "現在營業嗎" in message
        or "現在有營業嗎" in message
        or "現在有沒有開" in message
        or "目前有開嗎" in message
        or "目前營業嗎" in message
        or "現在休息嗎" in message
        or "幾點開" in message
        or "幾點關" in message):
        return get_current_status_reply(), []

    if "營業時間" in message:
        return get_hours_reply(), []

    if "菜單" in message:
        return get_menu_reply(), []

    # 其他問題 → LLM Intent + RAG
    return rag_chatbot(message)


# RAG + Groq LLM
def rag_chatbot(user_message):

    # 1. LLM 判斷使用者意圖
    classification = classify_message(user_message)
    intent = classification.get("intent", "general")
    sentiment = classification.get("sentiment", "neutral")
    
    # 2. Search Type 白名單
    allowed_search_types = {
        "history",
        "feature",
        "environment",
        "menu_item",
        "review",
        "review_recommendation",
        "review_environment"
    }

    search_types = []

    for item in classification.get("search_types", []):

        if item in allowed_search_types:
            search_types.append(item)

    # 如果 LLM 沒有回傳有效的 search_types
    '''
    if not search_types:
        search_types = [
            "feature",
            "menu_item",
        ]'''

    print("Intent：", intent)
    print("Sentiment：", sentiment)
    print("Search Types：", search_types)
    

    # 3. 問題 Embedding
    query_embedding = create_embedding(user_message)
    
    # 4. pgvector
    documents = search_restaurant_documents(
        restaurant_id=RESTAURANT_ID,
        query_embedding=query_embedding,
        knowledge_types=search_types,
        top_k=3,
        similarity_threshold=0.2
    )
    
    # 5. 找不到資料
    if not documents:
        return ("目前餐廳知識庫中沒有足夠的資料可以回答這個問題。", [])

    # 6. Groq 產生答案
    answer = generate_answer(
        user_question=user_message,
        context_documents=documents
    )

    # 7. Document ID
    document_ids = []

    for document in documents:
        document_ids.append(document["id"])

    return answer, document_ids

# 目前營業狀態
def get_current_status_reply():

    # 台灣時間
    now = datetime.now(ZoneInfo("Asia/Taipei"))
    current_time = now.time()
    day_of_week = (now.weekday() + 1) % 7

    day_names = {
        0: "星期日",
        1: "星期一",
        2: "星期二",
        3: "星期三",
        4: "星期四",
        5: "星期五",
        6: "星期六"
    }

    today_name = day_names[day_of_week]

    # 取得所有營業時間
    all_hours = get_restaurant_hours(RESTAURANT_ID)

    if not all_hours:
        return "目前沒有營業時間資訊。"

    
    # 找今天的營業時段
    today_hours = [
        row
        for row in all_hours
        if row["day_of_week"] == day_of_week
    ]

    # 排序
    today_hours.sort(key=lambda x: x["period_order"])

    
    # 今天公休
    if (not today_hours
        or all(row["is_closed"] for row in today_hours)):

        next_period = find_next_open_period(all_hours,day_of_week)

        if next_period:
            next_day = day_names[next_period["day_of_week"]]
            open_time = next_period["open_time"].strftime("%H:%M")

            return (
                f"🔴 今天是{today_name}，本餐廳公休。\n\n"
                f"📅 下一個營業時間："
                f"{next_day} {open_time}"
            )

        return (f"🔴 今天是{today_name}，本餐廳公休。")

    
    # 判斷今天目前是否正在營業
    for period in today_hours:

        if period["is_closed"]:
            continue

        open_time = period["open_time"]
        close_time = period["close_time"]

        if (open_time is None
            or close_time is None):
            continue

        
        # 現在正在營業
        if open_time <= current_time < close_time:
            return (
                f"🟢 目前營業中\n\n"
                f"今天（{today_name}）\n"
                f"{open_time.strftime('%H:%M')}"
                f"～"
                f"{close_time.strftime('%H:%M')}\n\n"
                f"⏰ 本時段營業至 "
                f"{close_time.strftime('%H:%M')}"
            )

    
    # 現在不是營業時間，找今天下一個營業時段
    for period in today_hours:

        if period["is_closed"]:
            continue

        open_time = period["open_time"]
        close_time = period["close_time"]

        if (open_time is None
            or close_time is None):
            continue

        if current_time < open_time:
            return (
                f"🔴 目前休息中\n\n"
                f"📅 今天（{today_name}）"
                f"\n下一個營業時間："
                f"{open_time.strftime('%H:%M')}"
                f"～"
                f"{close_time.strftime('%H:%M')}"
            )

    
    # 今天已經打烊，找下一個營業日
    next_period = find_next_open_period(all_hours,day_of_week)

    if next_period:
        next_day = day_names[next_period["day_of_week"]]
        open_time = next_period["open_time"].strftime("%H:%M")
        close_time = next_period["close_time"].strftime("%H:%M")

        return (
            f"🔴 今天已經打烊\n\n"
            f"📅 下一個營業時間：\n"
            f"{next_day} "
            f"{open_time}～{close_time}"
        )

    return "目前沒有找到下一個營業時間。"

# 找下一個營業時段
def find_next_open_period(all_hours, current_day):

    # 一天最多往後找 7 天
    for offset in range(1, 8):
        next_day = (current_day + offset) % 7
        next_day_hours = [
            row
            for row in all_hours
            if row["day_of_week"] == next_day
            and not row["is_closed"]
            and row["open_time"] is not None
            and row["close_time"] is not None
        ]

        if next_day_hours:
            next_day_hours.sort(key=lambda x: x["period_order"])
            return next_day_hours[0]

    return None

# 營業時間回覆
def get_hours_reply():
    hours = get_restaurant_hours(RESTAURANT_ID)

    if not hours:
        return "目前沒有營業時間資訊。"

    day_names = {
        0: "星期日",
        1: "星期一",
        2: "星期二",
        3: "星期三",
        4: "星期四",
        5: "星期五",
        6: "星期六"
    }

    result = "🕐 你好以下是本店的營業時間\n\n"
    grouped = {}

    for hour in hours:
        day = hour["day_of_week"]

        if day not in grouped:
            grouped[day] = []

        grouped[day].append(hour)

    # 按星期排序
    for day in sorted(grouped.keys()):
        result += f"【{day_names[day]}】\n"
        day_hours = sorted(grouped[day],key=lambda x: x["period_order"])

        if all(hour["is_closed"] for hour in day_hours):
            result += "公休\n\n"
            continue

        periods = []
        for hour in day_hours:

            if hour["is_closed"]:
                continue

            if (hour["open_time"] is None
                or hour["close_time"] is None):
                continue

            open_time = hour["open_time"].strftime("%H:%M")
            close_time = hour["close_time"].strftime("%H:%M")
            periods.append(f"{open_time}～{close_time}")

        if periods:
            result += "、".join(periods)
        else:
            result += "公休"

        result += "\n\n"

    return result.strip()

# 菜單回覆
def get_menu_reply():
    menu_items = get_menu_items(RESTAURANT_ID)

    if not menu_items:
        return "目前尚未建立菜單資料。"

    result = "🍜 菜單\n\n"
    current_category = None

    for item in menu_items:
        category = (
            item["category_name"]
            or "其他"
        )

        if category != current_category:
            result += f"\n【{category}】\n"
            current_category = category

        price = item["price"]
        if price is not None:
            price_int = int(float(price))
            result += (
                f"• {item['name']} "
                f"${price_int}\n"
            )
        else:
            result += (
                f"• {item['name']}\n"
            )

    return result


if __name__ == "__main__":
    app.run(host="0.0.0.0",port=5000,debug=True)
