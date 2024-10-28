import os
import numpy as np
import json
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
index = pc.Index("shibuya")

def upload_to_pinecone():
    # データ読み込み
    embeddings_dict = np.load("./embeddings/chapter01/combined_embeddings.npy", allow_pickle=True).item()
    with open("./embeddings/chapter01/content_metadata.json", 'r', encoding='utf-8') as f:
        metadata_dict = json.load(f)

    vectors = []
    for content_id, embeddings in embeddings_dict.items():
        # content_structureをスキップ
        if content_id == 'content_structure':
            continue

        # 全embeddingを結合
        combined_embedding = np.zeros(3072)  # 全体のサイズを確保
        
        if 'text' in embeddings:
            combined_embedding[:1536] = embeddings['text']
        if 'image' in embeddings:
            combined_embedding[1536:2048] = embeddings['image']
        if 'metadata' in embeddings:
            combined_embedding[2048:] = embeddings.get('metadata', np.zeros(1024))

        vector = {
            'id': f"doc_{content_id}",
            'values': combined_embedding.tolist(),
            'metadata': metadata_dict.get(content_id, {})
        }
        vectors.append(vector)
        print(f"Prepared vector for {content_id}: dimension={len(vector['values'])}")

    # アップロード
    if vectors:
        print(f"\nUploading {len(vectors)} vectors...")
        try:
            # バッチサイズを小さくして処理
            batch_size = 5
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                print(f"Uploading batch {i//batch_size + 1}/{(len(vectors)-1)//batch_size + 1}")
                index.upsert(vectors=batch)
            
            # 確認
            stats = index.describe_index_stats()
            print(f"Total vectors after upload: {stats.total_vector_count}")
        except Exception as e:
            print(f"Upload error: {str(e)}")

if __name__ == "__main__":
    upload_to_pinecone()