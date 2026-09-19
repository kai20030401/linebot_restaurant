import os
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()
#建立 PostgreSQL 連線
def get_db_connection():
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        database=os.getenv("DB_DATABASE"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD")
    )
    return conn


def get_or_create_chat_session(restaurant_id, user_identifier):
    """
    取得使用者目前的聊天 Session。
    如果沒有 Session，就建立新的 Session。
    """
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:

            # 找最近一個尚未結束的 Session
            cursor.execute(
                """
                SELECT id
                FROM chat_sessions
                WHERE restaurant_id = %s
                AND user_identifier = %s
                ORDER BY started_at DESC
                LIMIT 1;
                """,
                (restaurant_id, user_identifier)
            )

            session = cursor.fetchone()
            # 如果已經有 Session
            if session:
                return str(session["id"])

            # 建立新的 Session
            cursor.execute(
                """
                INSERT INTO chat_sessions (
                    restaurant_id,
                    user_identifier
                )
                VALUES (%s, %s)
                RETURNING id;
                """,
                (restaurant_id, user_identifier)
            )

            new_session = cursor.fetchone()
            conn.commit()
            return str(new_session["id"])

    except Exception as e:
        conn.rollback()
        print("建立 Chat Session 失敗：", e)
        raise
    finally:
        conn.close()


#儲存聊天訊息
def save_chat_message(
    session_id,
    role,
    content,
    model=None,
    retrieved_document_ids=None
):
    
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:

            cursor.execute(
                """
                INSERT INTO chat_messages (
                    session_id,
                    role,
                    content,
                    model,
                    retrieved_document_ids
                )
                VALUES (
                    %s,
                    %s,
                    %s,
                    %s,
                    %s::jsonb
                );
                """,
                (
                    session_id,
                    role,
                    content,
                    model,
                    retrieved_document_ids or "[]"
                )
            )

        conn.commit()

    except Exception as e:
        conn.rollback()
        print("儲存聊天訊息失敗：", e)
        raise
    finally:
        conn.close()

#取得餐廳基本資料
def get_restaurant_info(restaurant_id):
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:

            cursor.execute(
                """
                SELECT
                    name,
                    description,
                    phone,
                    city,
                    town,
                    zipcode,
                    address
                FROM restaurants
                WHERE restaurant_id = %s;
                """,
                (restaurant_id,)
            )

            restaurant = cursor.fetchone()
            return restaurant
    finally:
        conn.close()


#取得餐廳知識    
def get_restaurant_knowledge(restaurant_id,knowledge_type=None):
    conn = get_db_connection()

    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:

            if knowledge_type:
                cursor.execute(
                    """
                    SELECT
                        id,
                        knowledge_type,
                        title,
                        content,
                        source_type,
                        confidence
                    FROM restaurant_knowledge
                    WHERE restaurant_id = %s
                    AND knowledge_type = %s
                    AND is_active = TRUE
                    ORDER BY id;
                    """,(restaurant_id,knowledge_type)
                )

            else:

                cursor.execute(
                    """
                    SELECT
                        id,
                        knowledge_type,
                        title,
                        content,
                        source_type,
                        confidence
                    FROM restaurant_knowledge
                    WHERE restaurant_id = %s
                    AND is_active = TRUE
                    ORDER BY id;
                    """,(restaurant_id,)
                )

            return cursor.fetchall()
    finally:
        conn.close()


#取得餐廳營業時間
def get_restaurant_hours(restaurant_id, day_of_week=None):
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:

            if day_of_week is None:

                cursor.execute(
                    """
                    SELECT
                        day_of_week,
                        period_order,
                        open_time,
                        close_time,
                        is_closed
                    FROM restaurant_hours
                    WHERE restaurant_id = %s
                    ORDER BY day_of_week, period_order;
                    """,
                    (restaurant_id,)
                )

            else:
                cursor.execute(
                    """
                    SELECT
                        day_of_week,
                        period_order,
                        open_time,
                        close_time,
                        is_closed
                    FROM restaurant_hours
                    WHERE restaurant_id = %s
                    AND day_of_week = %s
                    ORDER BY period_order;
                    """,
                    (restaurant_id, day_of_week)
                )
            hours = cursor.fetchall()
            return hours
    finally:
        conn.close()


#取得目前供應中的餐點
def get_menu_items(restaurant_id):
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:

            cursor.execute(
                """
                SELECT
                    mi.name,
                    mi.price,
                    mc.name AS category_name
                FROM menu_items mi

                LEFT JOIN menu_categories mc
                ON mi.category_id = mc.id

                WHERE mi.restaurant_id = %s
                AND mi.is_available = TRUE

                ORDER BY
                    mc.display_order,
                    mi.name;
                """,
                (restaurant_id,)
            )

            menu_items = cursor.fetchall()
            return menu_items
    finally:
        conn.close()


#找出指定餐廳尚未產生 embedding 的文件。
def get_documents_without_embedding(restaurant_id):
    conn = get_db_connection()

    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:

            cursor.execute(
                """
                SELECT
                    id,
                    restaurant_id,
                    content,
                    source_type,
                    metadata
                FROM restaurant_documents
                WHERE restaurant_id = %s
                AND is_active = TRUE
                AND embedding IS NULL
                ORDER BY id;
                """,
                (restaurant_id,)
            )

            return cursor.fetchall()

    finally:
        conn.close()


#將 embedding 寫入 restaurant_documents
def update_document_embedding(document_id,embedding):
    conn = get_db_connection()

    try:
        vector_string = ("[" + ",".join(str(value) for value in embedding) + "]")

        with conn.cursor() as cursor:
            cursor.execute(
                """
                UPDATE restaurant_documents
                SET
                    embedding = %s::vector,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = %s;
                """,(vector_string,document_id)
            )
        conn.commit()

    except Exception as e:
        conn.rollback()
        print("儲存 embedding 失敗：",e)
        raise

    finally:
        conn.close()

# Rag搜尋
def search_restaurant_documents(
    restaurant_id,
    query_embedding,
    knowledge_types=None,
    top_k=3,
    similarity_threshold=0.2
):
    """
    使用 pgvector 搜尋餐廳相關文件。
    knowledge_types:允許搜尋的知識類型列表。
    """
    conn = get_db_connection()

    try:

        vector_string = ("[" + ",".join(str(value) for value in query_embedding) + "]")

        query = """
            SELECT
                id,
                knowledge_id,
                knowledge_type,
                content,
                source_type,
                metadata,

                1 - (
                    embedding <=> %s::vector
                ) AS similarity

            FROM restaurant_documents

            WHERE restaurant_id = %s
            AND is_active = TRUE
            AND embedding IS NOT NULL
        """

        params = [vector_string,restaurant_id]

        # 限制 Knowledge Type
        if knowledge_types:
            placeholders = ", ".join(["%s"] * len(knowledge_types))
            query += f"""AND knowledge_type IN ({placeholders})"""
            params.extend(knowledge_types)

        # 排序
        query += """
            ORDER BY
                embedding <=> %s::vector

            LIMIT %s;
        """

        params.extend([vector_string, top_k])

        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query,params)
            documents = cursor.fetchall()

            #test
            print("\n===== PGVECTOR 原始結果 =====")
            for document in documents:
                print(
                    f"ID={document['id']} "
                    f"Type={document['knowledge_type']} "
                    f"Similarity={document['similarity']}"
                    )


        # 相似度過濾
        filtered_documents = []
        for document in documents:
            similarity = float(document["similarity"])

            if similarity >= similarity_threshold:
                filtered_documents.append(document)

        return filtered_documents

    finally:
        conn.close()


# 儲存評論分析
def save_review_analysis(
    restaurant_id,
    analysis_type,
    topic,
    sentiment,
    summary,
    mention_count=1,
    confidence=None,
    analysis_model=None
):
    conn = get_db_connection()

    try:

        with conn.cursor() as cursor:

            cursor.execute(
                """
                INSERT INTO review_analysis (
                    restaurant_id,
                    analysis_type,
                    topic,
                    sentiment,
                    summary,
                    mention_count,
                    confidence,
                    analysis_model
                )
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s);
                """,
                (
                    restaurant_id,
                    analysis_type,
                    topic,
                    sentiment,
                    summary,
                    mention_count,
                    confidence,
                    analysis_model
                )
            )

        conn.commit()

    except Exception as e:
        conn.rollback()
        print("儲存 Review Analysis 失敗：", e)
        raise

    finally:
        conn.close()


# 取得評論分析
def get_review_analysis(restaurant_id):
    conn = get_db_connection()

    try:

        with conn.cursor(cursor_factory=RealDictCursor) as cursor:

            cursor.execute(
                """
                SELECT
                    id,
                    analysis_type,
                    topic,
                    sentiment,
                    summary,
                    mention_count,
                    confidence,
                    analysis_model
                FROM review_analysis
                WHERE restaurant_id = %s
                AND topic IS NOT NULL
                AND analysis_type = 'review_topic'
                ORDER BY id;
                """,
                (restaurant_id,)
            )

            return cursor.fetchall()

    finally:
        conn.close()


# Groq 整理好的推薦結果寫進 restaurant_knowledge
def save_recommendation_knowledge(
    restaurant_id,
    title,
    content,
    confidence,
    source_type="google_review_analysis"
):
    conn = get_db_connection()

    try:

        with conn.cursor(cursor_factory=RealDictCursor) as cursor:

            cursor.execute(
                """
                INSERT INTO restaurant_knowledge (
                    restaurant_id,
                    knowledge_type,
                    title,
                    content,
                    source_type,
                    confidence,
                    is_active
                )
                VALUES (%s,'recommendation',%s,%s,%s,%s,TRUE)
                RETURNING id;
                """,
                (
                    restaurant_id,
                    title,
                    content,
                    source_type,
                    confidence
                )
            )

            knowledge = cursor.fetchone()

        conn.commit()

        return knowledge["id"]

    except Exception as e:
        conn.rollback()
        print("儲存 Recommendation Knowledge 失敗：", e)
        raise

    finally:
        conn.close()


def delete_review_recommendation_knowledge(restaurant_id):
    """
    刪除之前由 Google Review 分析產生的 recommendation knowledge。
    """
    conn = get_db_connection()

    try:

        with conn.cursor() as cursor:

            cursor.execute(
                """
                DELETE FROM restaurant_knowledge
                WHERE restaurant_id = %s
                AND knowledge_type = 'recommendation'
                AND source_type = 'google_review_analysis';
                """,
                (restaurant_id,)
            )

        conn.commit()

    except Exception as e:
        conn.rollback()
        print("刪除舊 Recommendation Knowledge 失敗：",e)
        raise

    finally:
        conn.close()