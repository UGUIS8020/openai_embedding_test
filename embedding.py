
input_folder_name = "渋谷歯科技工所"
output_folder_name = "渋谷歯科技工所"

import os
from dataclasses import dataclass
from typing import Dict, Optional
import json
from PIL import Image
import numpy as np
from openai import OpenAI
from transformers import CLIPProcessor, CLIPModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
import torch
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv('OPENAI_API_KEY')
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

@dataclass
class ContentGroup:
    base_name: str  # 例：'content1'のような共通の基本名
    text_path: Optional[str]  # 例：'content1.txt'
    image_path: Optional[str]  # 例：'content1.jpg'
    json_path: Optional[str]  # 例：'content1.json'

class ContentProcessor:
    def __init__(self, folder_path: str, openai_api_key: str):
        self.folder_path = folder_path
        self.client = OpenAI(api_key=openai_api_key)
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def find_content_groups(self) -> list[ContentGroup]:
        """フォルダー内のファイルをグループ化"""
        files = os.listdir(self.folder_path)
        
        # 拡張子を除いたベース名でファイルをグループ化
        file_groups = {}
        for file in files:
            base_name = os.path.splitext(file)[0]
            ext = os.path.splitext(file)[1].lower()
            
            if base_name not in file_groups:
                file_groups[base_name] = ContentGroup(
                    base_name=base_name,
                    text_path=None,
                    image_path=None,
                    json_path=None
                )
            
            full_path = os.path.join(self.folder_path, file)
            if ext == '.txt':
                file_groups[base_name].text_path = full_path
            elif ext in ['.jpg', '.jpeg', '.png']:
                file_groups[base_name].image_path = full_path
            elif ext == '.json':
                file_groups[base_name].json_path = full_path

        return list(file_groups.values())

    def process_content_group(self, group: ContentGroup) -> Dict:
        """コンテンツグループの処理とembedding生成"""
        embeddings = {}
        content = {}

        # テキストの処理
        if group.text_path and os.path.exists(group.text_path):
            with open(group.text_path, 'r', encoding='utf-8') as f:
                text_content = f.read()
                content['text'] = text_content
                
                # テキストを分割
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=2000,  # 一度に処理するテキストの長さ
                    chunk_overlap=200,  # オーバーラップの長さ
                    length_function=len,
                    separators=["\n\n", "\n", "。", "、", " ", ""]
                )
                
                # テキストを分割してembeddingを生成
                chunks = text_splitter.split_text(text_content)
                print(f"Split text into {len(chunks)} chunks")
                
                # 各チャンクのembeddingを生成
                chunk_embeddings = []
                for chunk in chunks:
                    try:
                        embedding = self.get_text_embedding(chunk)[:1536]
                        chunk_embeddings.append(embedding)
                    except Exception as e:
                        print(f"Error processing chunk: {str(e)}")
                
                # 全チャンクの平均embeddingを計算
                if chunk_embeddings:
                    embeddings['text'] = np.mean(chunk_embeddings, axis=0)
                    print(f"Created average embedding from {len(chunk_embeddings)} chunks")
                else:
                    print("Warning: No embeddings were generated")
                    embeddings['text'] = np.zeros(1536)

        # 画像の処理 (512次元)
        if group.image_path and os.path.exists(group.image_path):
            image = Image.open(group.image_path)
            content['image_path'] = group.image_path
            embeddings['image'] = self.get_image_embedding(image)
        else:  # 画像がない場合
            embeddings['image'] = np.zeros(512)

        # JSONの処理 (1024次元に制限)
        if group.json_path and os.path.exists(group.json_path):
            with open(group.json_path, 'r', encoding='utf-8') as f:
                json_content = json.load(f)
                content['metadata'] = json_content
                json_str = json.dumps(json_content, ensure_ascii=False)
                metadata_embedding = self.get_text_embedding(json_str)[:1024]
                embeddings['metadata'] = metadata_embedding
        else:  # メタデータがない場合
            embeddings['metadata'] = np.zeros(1024)

        # 次元数の確認と出力
        for key, emb in embeddings.items():
            print(f"{key} embedding dimension: {len(emb)}")

        return {
            'content': content,
            'embeddings': embeddings
        }

    def get_text_embedding(self, text: str) -> np.ndarray:
        """テキストのembedding生成"""
        response = self.client.embeddings.create(
            model="text-embedding-3-large",
            input=text
        )
        # 1536次元に制限する必要がある
        embedding = np.array(response.data[0].embedding)[:1536]  # ここを修正
        print(f"Text embedding dimension: {len(embedding)}")
        return embedding

    def get_image_embedding(self, image: Image.Image) -> np.ndarray:
        """画像のembedding生成"""
        inputs = self.clip_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**inputs)
        return image_features.squeeze().numpy()

def main():
    input_folder = f"./data/{input_folder_name}"
    print(f"\nChecking input folder: {input_folder}")
    if os.path.exists(input_folder):
        files = os.listdir(input_folder)
        print(f"Files found in input folder: {files}")
    else:
        print("Input folder not found!")

    # 入力フォルダーと出力フォルダーの指定    
    input_folder = f"./data/{input_folder_name}"  # 入力フォルダー    
    output_folder = f"./embeddings/{output_folder_name}"     # 出力フォルダー
    
    # 入力フォルダーの存在確認
    if not os.path.exists(input_folder):
        raise ValueError(f"Input folder not found: {input_folder}")
    
    # embeddingsフォルダーが存在しない場合は作成
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created embeddings folder at: {output_folder}")

    processor = ContentProcessor(input_folder, openai_api_key)
    
    # コンテンツグループの検出
    content_groups = processor.find_content_groups()
    
    # 結果を格納する辞書
    processed_contents = {}
    
    # 各グループの処理
    for group in content_groups:
        print(f"Processing group: {group.base_name}")
        try:
            result = processor.process_content_group(group)
            processed_contents[group.base_name] = result
        except Exception as e:
            print(f"Error processing group {group.base_name}: {str(e)}")

                # 各グループの処理部分の後に追加
    print(f"\nProcessed contents summary:")
    print(f"Number of processed items: {len(processed_contents)}")
    for name, data in processed_contents.items():
        print(f"\nContent name: {name}")
        print(f"Available embeddings: {list(data['embeddings'].keys())}")
        print(f"Content keys: {list(data['content'].keys())}")

    # embedding保存前に次元数を確認
    embeddings_dict = {}
    for name, data in processed_contents.items():
        embeddings = data['embeddings']
        # 次元数の確認と調整
        if 'text' in embeddings:
            embeddings['text'] = embeddings['text'][:1536]
        if 'metadata' in embeddings:
            embeddings['metadata'] = embeddings['metadata'][:1024]
        # 画像embeddingは512次元なのでそのまま
        
        print(f"\nContent: {name}")
        for key, emb in embeddings.items():
            print(f"- {key} dimension: {len(emb)}")
        
        embeddings_dict[name] = embeddings

    print(f"\nPreparing to save:")
    print(f"Embeddings dict contains {len(embeddings_dict)} items")
    for name, emb in embeddings_dict.items():
        print(f"Item '{name}' has embeddings: {list(emb.keys())}")
        for key, value in emb.items():
            print(f"- {key}: shape {value.shape}")

    # 保存
    np.save(
        os.path.join(output_folder, "combined_embeddings.npy"), 
        embeddings_dict
    )
    
    # コンテンツとメタデータの保存
    content_dict = {
        name: data['content']
        for name, data in processed_contents.items()
    }
    with open(os.path.join(output_folder, "content_metadata.json"), 'w', encoding='utf-8') as f:
        json.dump(content_dict, f, ensure_ascii=False, indent=2)

    print(f"\n処理が完了しました。")
    print(f"結果は {output_folder} に保存されました：")
    print(f"- {os.path.join(output_folder, 'combined_embeddings.npy')}")
    print(f"- {os.path.join(output_folder, 'content_metadata.json')}")
    
    return processed_contents

if __name__ == "__main__":
    main()