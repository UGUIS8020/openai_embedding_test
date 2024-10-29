import os
import numpy as np
import json
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

# Pinecone APIの初期化
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
index = pc.Index("shibuya")

def validate_metadata(metadata):
    """Pineconeに適した形式かをチェック"""
    valid_metadata = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)) or \
           (isinstance(value, list) and all(isinstance(item, str) for item in value)):
            valid_metadata[key] = value
        else:
            print(f"Warning: Unsupported metadata format for '{key}': {value}")
    return valid_metadata

def upload_to_pinecone(batch_size=5):
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
        combined_embedding = np.zeros(3072)
        if 'text' in embeddings:
            combined_embedding[:1536] = embeddings['text']
        if 'image' in embeddings:
            combined_embedding[1536:2048] = embeddings['image']
        if 'metadata' in embeddings:
            combined_embedding[2048:] = embeddings.get('metadata', np.zeros(1024))

        # メタデータのバリデーション
        metadata = validate_metadata(metadata_dict.get(content_id, {}))
        
        vector = {
            'id': f"doc_{content_id}",
            'values': combined_embedding.tolist(),
            'metadata': metadata
        }
        vectors.append(vector)
        print(f"Prepared vector for {content_id}: dimension={len(vector['values'])}")

    # アップロード処理
    if vectors:
        print(f"\nUploading {len(vectors)} vectors...")
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            print(f"Uploading batch {i // batch_size + 1}/{(len(vectors) - 1) // batch_size + 1}")
            try:
                index.upsert(vectors=batch)
            except Exception as e:
                print(f"Upload error in batch {i // batch_size + 1}: {str(e)}")
                continue  # エラー発生時に処理を続行

        # アップロード結果の確認
        try:
            stats = index.describe_index_stats()
            print(f"Total vectors after upload: {stats.total_vector_count}")
        except Exception as e:
            print(f"Error retrieving index stats: {str(e)}")

if __name__ == "__main__":
    upload_to_pinecone(batch_size=5)