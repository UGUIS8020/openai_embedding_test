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
import datetime
import uuid

load_dotenv()

input_folder_name = "chapter01"
output_folder_name = "chapter01"

openai_api_key = os.getenv('OPENAI_API_KEY')
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

@dataclass
class ContentGroup:
    base_name: str
    text_path: Optional[str]
    image_path: Optional[str]
    json_path: Optional[str]

    def generate_default_metadata(self) -> dict:
        """デフォルトのメタデータを生成"""
        current_time = datetime.datetime.now().isoformat()
        return {
            "title": self.base_name,
            "sequence_number": None,  # 後で設定
            "content_id": str(uuid.uuid4()),
            "created_at": current_time,
            "updated_at": current_time,
            # "content_type_has_text": self.text_path is not None,  # 辞書形式ではなく単独のブール型フィールドに変更
            # "content_type_has_image": self.image_path is not None,  # 辞書形式ではなく単独のブール型フィールドに変更
            "description": "",
            "tags": [],
            "version": "1.0"
        }

class ContentProcessor:
    def __init__(self, folder_path: str, openai_api_key: str):
        self.folder_path = folder_path
        self.client = OpenAI(api_key=openai_api_key)
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.content_structure = self.load_content_structure()

    def find_content_groups(self) -> list[ContentGroup]:
        """フォルダー内のファイルをグループ化"""
        files = os.listdir(self.folder_path)
        
        file_groups = {}
        for file in files:
            # content_structure.jsonや他のJSONファイルをスキップ
            if file.endswith('.json'):
                continue
                
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

        return list(file_groups.values())

    def load_content_structure(self) -> dict:
        """content_structure.jsonからコンテンツ構造を読み込む"""
        structure_path = os.path.join(self.folder_path, "content_structure.json")
        if os.path.exists(structure_path):
            with open(structure_path, 'r', encoding='utf-8') as f:
                structure = json.load(f)
                result = {}
                
                # structureが配列の場合の処理
                if isinstance(structure, list):
                    for item in structure:
                        for content in item.get('contents', []):
                            if 'text_file' in content:
                                file_name = content['text_file'].split('.')[0]
                                result[file_name] = {
                                    'id': item.get('id', f"page_{item.get('page', 0):02d}"),
                                    'page': item.get('page', 0),
                                    'title': item.get('page_title', {
                                        'en': f"Page {item.get('page', 0)}",
                                        'ja': f"ページ{item.get('page', 0)}"
                                    })
                                }
                # structureがオブジェクトの場合の処理
                else:
                    # 共通の処理: contentsの処理
                    for section in structure.get('contents', []):
                        result.update(self._process_section(section))
                    
                    # 追加の処理: figuresの処理（存在する場合）
                    if 'figures' in structure:
                        result.update(self._process_figures(structure['figures']))
                
                return result
        return {}

    def _process_section(self, section: dict) -> dict:
        """セクション（chapter/section）の処理"""
        result = {}
        for page in section.get('pages', []):
            # ファイル情報の取得
            contents = page.get('contents', [])
            for content in contents:
                if content.get('type') == 'text':
                    file_name = content.get('file_name', '').split('.')[0]
                    if file_name:
                        result[file_name] = {
                            'id': page['id'],
                            'page': page.get('page_number'),
                            'title': page.get('page_title'),
                            'section_title': section.get('section_title'),
                            'type': section.get('type'),
                            'description': page.get('description', ''),
                            'metadata': page.get('metadata', {})
                        }
        return result

    def _process_figures(self, figures: list) -> dict:
        """図の処理"""
        result = {}
        for fig_group in figures:
            for fig in fig_group.get('figures', []):
                file_name = fig['text_file'].split('.')[0]
                result[file_name] = {
                    'id': fig['id'],
                    'figure_id': fig['figure_id'],
                    'description': fig['description'],
                    'group_title': fig_group['title'],
                    'type': 'figure'
                }
        return result

    def process_content_group(self, group: ContentGroup, sequence_number: int) -> Dict:
        """コンテンツグループの処理とembedding生成"""
        embeddings = {}
        content = {}
        
        # メタデータの基本情報生成
        metadata = group.generate_default_metadata()
        metadata["sequence_number"] = sequence_number
        
        # content_structureからの情報があれば追加
        base_name = os.path.splitext(os.path.basename(group.text_path))[0] if group.text_path else group.base_name
        if base_name in self.content_structure:
            structure_info = self.content_structure[base_name]
            metadata.update({
                "content_id": structure_info['id'],
                "title": structure_info['title']['ja'],
                "title_en": structure_info['title']['en'],
                "page_number": structure_info['page']
            })

        # テキストの処理
        if group.text_path and os.path.exists(group.text_path):
            with open(group.text_path, 'r', encoding='utf-8') as f:
                text_content = f.read()
                content['text'] = text_content
                content['metadata'] = metadata
                
                # 説明文が空の場合のみ生成を試みる
                if not metadata.get("description"):
                    try:
                        first_lines = '\n'.join(text_content.split('\n')[:3])
                        metadata["description"] = first_lines[:100] + ("..." if len(first_lines) > 100 else "")
                    except Exception as e:
                        print(f"Error generating description: {str(e)}")

                # ここからembedding生成処理を追加
                try:
                    # テキストを分割
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=2000,
                        chunk_overlap=200,
                        length_function=len,
                        separators=["\n\n", "\n", "。", "、", " ", ""]
                    )
                    
                    chunks = text_splitter.split_text(text_content)
                    print(f"Split text into {len(chunks)} chunks")
                    
                    # テキストembedding生成
                    chunk_embeddings = []
                    for chunk in chunks:
                        embedding = self.get_text_embedding(chunk)
                        chunk_embeddings.append(embedding)
                    
                    if chunk_embeddings:
                        embeddings['text'] = np.mean(chunk_embeddings, axis=0)
                        print(f"Created text embedding with dimension: {len(embeddings['text'])}")

                    # メタデータembedding生成
                    metadata_str = json.dumps(metadata, ensure_ascii=False)
                    embeddings['metadata'] = self.get_text_embedding(metadata_str)[:1024]
                    
                    # 画像embedding（画像がない場合は0ベクトル）
                    embeddings['image'] = np.zeros(512)
                    
                except Exception as e:
                    print(f"Error generating embeddings: {str(e)}")
                    embeddings['text'] = np.zeros(1536)
                    embeddings['metadata'] = np.zeros(1024)
                    embeddings['image'] = np.zeros(512)

        if group.image_path and os.path.exists(group.image_path):
            try:
                image = Image.open(group.image_path)
                content['image_path'] = group.image_path
                embeddings['image'] = self.get_image_embedding(image)
                print(f"Processed image: {group.image_path}")
            except Exception as e:
                print(f"Error processing image {group.image_path}: {str(e)}")
                embeddings['image'] = np.zeros(512)
        else:
            embeddings['image'] = np.zeros(512)

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
        embedding = np.array(response.data[0].embedding)[:1536]
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

    output_folder = f"./embeddings/{output_folder_name}"
    
    if not os.path.exists(input_folder):
        raise ValueError(f"Input folder not found: {input_folder}")
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created embeddings folder at: {output_folder}")

    processor = ContentProcessor(input_folder, openai_api_key)
    content_groups = processor.find_content_groups()
    
    processed_contents = {}
    
    for idx, group in enumerate(content_groups, 1):
        print(f"Processing group: {group.base_name} (#{idx})")
        try:
            result = processor.process_content_group(group, sequence_number=idx)
            processed_contents[group.base_name] = result
        except Exception as e:
            print(f"Error processing group {group.base_name}: {str(e)}")

    print(f"\nProcessed contents summary:")
    print(f"Number of processed items: {len(processed_contents)}")
    for name, data in processed_contents.items():
        print(f"\nContent name: {name}")
        print(f"Available embeddings: {list(data['embeddings'].keys())}")
        print(f"Content keys: {list(data['content'].keys())}")

    embeddings_dict = {}
    for name, data in processed_contents.items():
        embeddings = data['embeddings']
        if 'text' in embeddings:
            embeddings['text'] = embeddings['text'][:1536]
        if 'metadata' in embeddings:
            embeddings['metadata'] = embeddings['metadata'][:1024]
        
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

    np.save(
        os.path.join(output_folder, "combined_embeddings.npy"), 
        embeddings_dict
    )
    
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