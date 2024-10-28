import os
import numpy as np
import json
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

# Pinecone初期化
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
index = pc.Index("shibuya")

def upload_to_pinecone():
    # データ読み込み
    embeddings_dict = np.load("./embeddings/test/combined_embeddings.npy", allow_pickle=True).item()
    with open("./embeddings/test/content_metadata.json", 'r', encoding='utf-8') as f:
        metadata_dict = json.load(f)

    # 最初のデータの中身を確認
    first_key = list(embeddings_dict.keys())[0]
    print(f"\nFirst key: {first_key}")
    print(f"Embeddings structure: {embeddings_dict[first_key].keys()}")
    print(f"Text embedding shape: {embeddings_dict[first_key]['text'].shape}")

    vectors = []
    for content_id, embeddings in embeddings_dict.items():
        # text embeddingのみを使用
        text_embedding = embeddings['text']
        
        # ベクトルデータの形式を確認
        if not isinstance(text_embedding, np.ndarray):
            print(f"Warning: text_embedding is not numpy array for {content_id}")
            continue

        # IDをASCII文字列に変換
        safe_id = f"doc_{len(vectors)}"  # 単純な連番

        vector = {
            'id': safe_id,
            'values': text_embedding.tolist(),  # numpy arrayをリストに変換
            'metadata': metadata_dict.get(content_id, {})
        }
        
        print(f"\nVector {safe_id}:")
        print(f"- values length: {len(vector['values'])}")
        print(f"- metadata keys: {vector['metadata'].keys()}")
        
        vectors.append(vector)

    # アップロード
    if vectors:
        print(f"\nUploading {len(vectors)} vectors...")
        index.upsert(vectors=vectors)
        
        # 確認
        stats = index.describe_index_stats()
        print(f"Total vectors after upload: {stats.total_vector_count}")

if __name__ == "__main__":
    upload_to_pinecone()