read_folder_name = "渋谷歯科技工所"

import os
import numpy as np
import json
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()

# Pineconeの初期化
pinecone_api_key = os.getenv('PINECONE_API_KEY')
if not pinecone_api_key:
    raise ValueError("PINECONE_API_KEY not found in environment variables")

pc = Pinecone(api_key=pinecone_api_key)
index_name = "shibuya"

# インデックスの状態を確認
print("Checking existing indexes...")
print(f"Available indexes: {pc.list_indexes().names()}")

if index_name not in pc.list_indexes().names():
    print(f"Creating new index with dimension 3072: {index_name}")
    pc.create_index(
        name=index_name,
        dimension=3072,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

index = pc.Index(index_name)

# アップロード前のインデックス状態を確認
print("\nIndex stats before upload:")
stats = index.describe_index_stats()
print(f"Total vectors: {stats.total_vector_count}")

def load_embeddings(embeddings_folder):
    """embeddingsフォルダーからデータを読み込む"""
    embeddings_path = os.path.join(embeddings_folder, "combined_embeddings.npy")
    embeddings_dict = np.load(embeddings_path, allow_pickle=True).item()
    
    metadata_path = os.path.join(embeddings_folder, "content_metadata.json")
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata_dict = json.load(f)
    
    print(f"Loaded {len(embeddings_dict)} embeddings")
    return embeddings_dict, metadata_dict

def upload_to_pinecone(embeddings_dict, metadata_dict, index): 
    """PineconeにEmbeddingデータをアップロード"""
    batch_size = 100
    vectors = []
    
    # 既存のベクターを確認
    stats_before = index.describe_index_stats()
    print(f"\nBefore upload - Total vectors: {stats_before.total_vector_count}")
    
    for content_id, embeddings in embeddings_dict.items():
        print(f"\nProcessing content_id: {content_id}")
        
        try:
            combined_embedding = np.concatenate([
                embeddings.get('text', np.zeros(1536)),
                embeddings.get('image', np.zeros(512)),
                embeddings.get('metadata', np.zeros(1024))
            ])

            safe_id = f"doc_{content_id}"  # IDをより明確に
            
            vector = {
                'id': safe_id,
                'values': combined_embedding.tolist(),
                'metadata': {
                    'original_id': str(content_id),
                    'content_id': str(content_id),
                    'text': str(metadata_dict.get(content_id, {}).get('text', '')),
                    'image_path': str(metadata_dict.get(content_id, {}).get('image_path', '')),
                    'metadata': json.dumps(metadata_dict.get(content_id, {}).get('metadata', {}))
                }
            }
            vectors.append(vector)

        except Exception as e:
            print(f"Error processing {content_id}: {str(e)}")

    # 一括アップロード
    if vectors:
        try:
            print(f"\nUploading batch of {len(vectors)} vectors")
            index.upsert(vectors=vectors)
                
            # アップロード後の状態を確認
            stats_after = index.describe_index_stats()
            print(f"\nAfter upload - Total vectors: {stats_after.total_vector_count}")
                
        except Exception as e:
            print(f"Error uploading batch: {str(e)}")
            raise e

def main():
    # 現在の作業ディレクトリを表示
    current_dir = os.getcwd()
    embeddings_folder = os.path.join(current_dir, "embeddings", read_folder_name)
    
    print(f"\nPath Information:")
    print(f"Current directory: {current_dir}")
    print(f"Full embeddings path: {embeddings_folder}")
    
    # ファイルの存在確認
    embeddings_path = os.path.join(embeddings_folder, "combined_embeddings.npy")
    metadata_path = os.path.join(embeddings_folder, "content_metadata.json")
    
    print(f"\nFile checks:")
    print(f"Embeddings file exists: {os.path.exists(embeddings_path)}")
    print(f"Metadata file exists: {os.path.exists(metadata_path)}")
    
    if not os.path.exists(embeddings_folder):
        raise ValueError(f"Embeddings folder not found: {embeddings_folder}")
        
    print("\nTrying to load files...")
    try:
        embeddings_dict = np.load(embeddings_path, allow_pickle=True).item()
        print(f"Successfully loaded embeddings: {len(embeddings_dict)} items")
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata_dict = json.load(f)
            print(f"Successfully loaded metadata: {len(metadata_dict)} items")
    except Exception as e:
        print(f"Error loading files: {str(e)}")
        raise
    
    print("\nUploading to Pinecone...")
    upload_to_pinecone(embeddings_dict, metadata_dict, index)
    
    print("\nIndex stats after upload:")
    stats = index.describe_index_stats()
    print(f"Total vectors: {stats.total_vector_count}")    

if __name__ == "__main__":
    main()