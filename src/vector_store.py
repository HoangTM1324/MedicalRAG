import os
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document

def prepare_documents(dataset):
    """Chuyển đổi dataset thành list các Document của LangChain."""
    docs = []
    print("Đang xử lý dữ liệu thành Document...")
    for item in dataset:
        # Lấy context làm nội dung chính để search
        # PubMedQA có cấu trúc context: {'contexts': [...]}
        context_list = item.get('context', {}).get('contexts', [])
        full_context = "\n".join(context_list)
        
        # Metadata giúp truy xuất ngược lại câu hỏi hoặc label sau này
        meta = {
            "pubid": item.get('pubid', 'unknown'),
            "question": item.get('question', ''),
            "final_decision": item.get('final_decision', 'maybe')
        }
        
        if full_context.strip(): # Chỉ thêm nếu có nội dung
            docs.append(Document(page_content=full_context, metadata=meta))
            
    print(f"-> Đã chuẩn bị {len(docs)} documents.")
    return docs

def build_vector_db(documents, embedding_model, persist_path, batch_size=1000):
    """Tạo và lưu ChromaDB."""
    if os.path.exists(persist_path):
        print(f"⚠️ Cảnh báo: Thư mục {persist_path} đã tồn tại. Đang xóa để tạo mới...")
        import shutil
        shutil.rmtree(persist_path)
        
    print(f"Đang tạo Vector DB tại {persist_path}...")
    
    # Batch processing để tránh tràn RAM nếu data lớn
    total_docs = len(documents)
    for i in range(0, total_docs, batch_size):
        batch = documents[i : i + batch_size]
        print(f"  - Processing batch {i}/{total_docs}...")
        if i == 0:
            # Batch đầu tiên khởi tạo DB
            db = Chroma.from_documents(
                documents=batch,
                embedding=embedding_model,
                persist_directory=persist_path
            )
        else:
            # Các batch sau thêm vào DB đã có
            db.add_documents(batch)
            
    print(">>> Hoàn tất! Database đã được lưu.")
    return db
