# check_embeddings.py
import os
import numpy as np
import json

def check_embeddings(embeddings_folder: str):
    """生成されたembeddingの次元数を確認"""
    print("\n=== Checking Embeddings ===")
    embeddings_path = os.path.join(embeddings_folder, "combined_embeddings.npy")
    metadata_path = os.path.join(embeddings_folder, "content_metadata.json")
    
    # NPYファイルの読み込み
    print(f"\nLoading embeddings from: {embeddings_path}")
    embeddings_dict = np.load(embeddings_path, allow_pickle=True).item()
    
    # メタデータの読み込み
    print(f"Loading metadata from: {metadata_path}")
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata_dict = json.load(f)
    
    print("\n=== Embeddings Analysis ===")
    for content_id, embeddings in embeddings_dict.items():
        print(f"\nContent ID: {content_id}")
        
        # 各embeddingタイプの次元を確認
        text_emb = embeddings.get('text')
        if text_emb is not None:
            print(f"Text embedding: shape={text_emb.shape}, non-zero={np.count_nonzero(text_emb)}")
        else:
            print("Text embedding: None")
        
        image_emb = embeddings.get('image')
        if image_emb is not None:
            print(f"Image embedding: shape={image_emb.shape}, non-zero={np.count_nonzero(image_emb)}")
        else:
            print("Image embedding: None")
            
        # 結合後の次元数を確認
        total_dim = 0
        for emb in embeddings.values():
            if isinstance(emb, np.ndarray):
                total_dim += len(emb)
        print(f"Total dimension: {total_dim}")
        
        # メタデータの確認
        if content_id in metadata_dict:
            print("Metadata: Present")
            for key in metadata_dict[content_id].keys():
                print(f"- {key}")
        else:
            print("Metadata: Missing")

def summarize_issues(embeddings_folder: str):
    """問題のあるembeddingをまとめて表示"""
    embeddings_path = os.path.join(embeddings_folder, "combined_embeddings.npy")
    embeddings_dict = np.load(embeddings_path, allow_pickle=True).item()
    
    print("\n=== Issues Summary ===")
    issues_found = False
    
    for content_id, embeddings in embeddings_dict.items():
        issues = []
        
        # 次元数のチェック
        total_dim = 0
        for emb_type, emb in embeddings.items():
            if isinstance(emb, np.ndarray):
                dim = len(emb)
                total_dim += dim
                if dim == 0:
                    issues.append(f"Zero dimension in {emb_type} embedding")
                elif emb_type == 'text' and dim != 1536:  # OpenAI embedding
                    issues.append(f"Unexpected text embedding dimension: {dim}")
                elif emb_type == 'image' and dim != 512:  # CLIP embedding
                    issues.append(f"Unexpected image embedding dimension: {dim}")
        
        if total_dim == 0:
            issues.append("Total dimension is zero")
        
        if issues:
            issues_found = True
            print(f"\nContent ID: {content_id}")
            for issue in issues:
                print(f"- {issue}")
    
    if not issues_found:
        print("No issues found!")

def main():
    embeddings_folder = "./embeddings/test02"
    
    if not os.path.exists(embeddings_folder):
        raise ValueError(f"Embeddings folder not found: {embeddings_folder}")
    
    check_embeddings(embeddings_folder)
    summarize_issues(embeddings_folder)

if __name__ == "__main__":
    main()