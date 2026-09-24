import os

from dotenv import load_dotenv
from database import (
    get_restaurant_knowledge,
    get_db_connection
)

load_dotenv()
RESTAURANT_ID = int(os.getenv("RESTAURANT_ID","9527"))

# 建立 RAG Document
def build_document_content(knowledge):
    """
    將 restaurant_knowledge 的資料
    整理成適合 RAG 使用的文字。
    """
    title = knowledge["title"]
    content = knowledge["content"]

    if title:
        document_content = (f"{title}\n"f"{content}")

    else:
        document_content = content

    return document_content.strip()


# 檢查 Document 是否已經存在
def document_exists(restaurant_id,knowledge_id):
    """
    確認這筆 knowledge 是否已經建立對應的 document。
    """
    conn = get_db_connection()

    try:

        with conn.cursor() as cursor:

            cursor.execute(
                """
                SELECT 1
                FROM restaurant_documents
                WHERE restaurant_id = %s
                AND knowledge_id = %s
                LIMIT 1;
                """,(restaurant_id,knowledge_id)
            )

            return cursor.fetchone() is not None

    finally:
        conn.close()



# 建立 Document
def insert_document(restaurant_id,knowledge_id,knowledge_type,content,source_type):
    """
    將資料寫入 restaurant_documents。
    """
    conn = get_db_connection()

    try:

        with conn.cursor() as cursor:

            cursor.execute(
                """
                INSERT INTO restaurant_documents (
                    restaurant_id,
                    knowledge_id,
                    knowledge_type,
                    content,
                    source_type,
                    is_active
                )
                VALUES (%s,%s,%s,%s,%s,TRUE);
                """,
                (restaurant_id,knowledge_id,knowledge_type,content,source_type)
            )
        conn.commit()

    except Exception as e:
        conn.rollback()
        print("建立 Document 失敗：",e)
        raise

    finally:
        conn.close()


# 主程式
def main():

    print("========================================")
    print("開始建立 Restaurant Documents")
    print("========================================")
    
    # 從 restaurant_knowledge 取得資料
    knowledge_list = get_restaurant_knowledge(RESTAURANT_ID)

    if not knowledge_list:
        print("\n找不到 restaurant_knowledge 資料。")
        print("請先確認 Version 2 已經建立餐廳知識。")
        return

    print(f"\n找到 {len(knowledge_list)} 筆餐廳知識。")
    created_count = 0
    skipped_count = 0

    # 一筆一筆建立 RAG Document
    for knowledge in knowledge_list:
        knowledge_id = knowledge["id"]
        knowledge_type = knowledge["knowledge_type"]
        title = knowledge["title"]
        source_type = (knowledge["source_type"] or "unknown")

        print("\n----------------------------------------")
        print(f"Knowledge ID：{knowledge_id}")
        print(f"Title：{title}")

        # 檢查是否已經建立
        if document_exists(RESTAURANT_ID,knowledge_id):
            print("結果：已存在，跳過。")
            skipped_count += 1
            continue

        # 建立 Document Content
        document_content = build_document_content(knowledge)

        print("建立內容：")
        print(document_content)

        insert_document(
            restaurant_id=RESTAURANT_ID,
            knowledge_id=knowledge_id,
            knowledge_type=knowledge_type,
            content=document_content,
            source_type=source_type
        )

        created_count += 1

    # 結果
    print("\n========================================")
    print("Document 處理完成")
    print("========================================")
    print(f"成功建立：{created_count} 筆")
    print(f"跳過既有：{skipped_count} 筆")
    print(f"總知識數量：{len(knowledge_list)} 筆")


if __name__ == "__main__":
    main()
