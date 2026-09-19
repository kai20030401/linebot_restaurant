import os

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

model = SentenceTransformer(EMBEDDING_MODEL)


# 建立單筆 Embedding
def create_embedding(text):
    """
    將單筆文字轉換成 Embedding。
    回傳：
        list[float]
    """

    if not text or not text.strip():
        raise ValueError("Embedding 的文字不能是空的。")

    embedding = model.encode(
        text,
        normalize_embeddings=True
    )

    return embedding.tolist()


# 建立多筆 Embedding
def create_embeddings(texts):
    """
    一次將多筆文字轉換成 Embeddings。
    texts:
        list[str]
    回傳：
        list[list[float]]
    """

    if not texts:
        return []

    for text in texts:

        if not text or not text.strip():
            raise ValueError("Embedding 的文字不能包含空字串。")

    embeddings = model.encode(
        texts,
        normalize_embeddings=True
    )

    return embeddings.tolist()


# 取得 Embedding 維度
def get_embedding_dimension():
    """
    取得目前模型的 Embedding 維度。
    """
    return model.get_sentence_embedding_dimension()

