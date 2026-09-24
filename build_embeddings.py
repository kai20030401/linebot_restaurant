import os 
 
from dotenv import load_dotenv 
from database import ( 
    get_documents_without_embedding, 
    update_document_embedding 
) 
from embedding import ( 
    create_embeddings, 
    get_embedding_dimension 
) 
 
load_dotenv() 
RESTAURANT_ID = int(os.getenv("RESTAURANT_ID","9527")) 
 
def main(): 
    # 取得尚未建立 embedding 的文件 
    documents = get_documents_without_embedding(RESTAURANT_ID) 
 
    if not documents: 
        print("\n沒有需要建立 Embedding 的文件。") 
        return 
 
    print(f"\n找到 {len(documents)} 筆文件。") 
 
    # 確認 Embedding 維度 
    dimension = get_embedding_dimension() 
    print(f"目前 Embedding 維度：{dimension}") 
 
    # 取得所有文件文字 
    texts = [document["content"] for document in documents] 
 
    # 批次建立 Embeddings 
    embeddings = create_embeddings(texts) 
 
    # 寫入 PostgreSQL 
    success_count = 0 
 
    for document, embedding in zip(documents,embeddings): 
        document_id = document["id"] 
        print(f"\n處理 Document ID："f"{document_id}") 
 
        try: 
 
            update_document_embedding(document_id,embedding) 
            print(f"Document {document_id} "f"寫入成功。") 
            success_count += 1 
 
        except Exception as e: 
            print(f"Document {document_id} "f"寫入失敗：{e}") 
 
 
    print("Embedding 建立完成") 
    print(f"成功：{success_count} / "f"{len(documents)}") 
    print(f"Embedding 維度：{dimension}") 
 
if __name__ == "__main__": 
    main() 

 