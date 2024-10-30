import os
from dataclasses import dataclass
from typing import Dict, Optional, List
import json
from PIL import Image
import numpy as np
from openai import OpenAI
from transformers import CLIPProcessor, CLIPModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
import torch
from dotenv import load_dotenv
from pathlib import Path
import datetime
import uuid
import logging
from typing import TypedDict

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
            "sequence_number": None,
            "content_id": str(uuid.uuid4()),
            "created_at": current_time,
            "updated_at": current_time,
            "description": "",
            "tags": [],
            "version": "1.0"
        }

class ContentProcessor:
    def __init__(self, folder_path: str, openai_api_key: str):
        self.folder_path = folder_path
        self.client = OpenAI(api_key=openai_api_key)  # 新しいOpenAIクライアントの初期化
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.content_structure = self.load_content_structure()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=300,
            length_function=len,
            separators=["。", "、", "\n\n", "\n", " ", ""]
        )

    def find_content_groups(self) -> List[ContentGroup]:
        """フォルダー内のファイルをグループ化"""
        files = os.listdir(self.folder_path)
        file_groups = {}
        
        for file in files:
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

    def get_text_embeddings(self, texts: list) -> list[np.ndarray]:
        """テキストのembeddingを生成（新しいOpenAI API形式）"""
        try:
            response = self.client.embeddings.create(
                model="text-embedding-3-large",
                input=texts
            )
            embeddings = [np.array(item.embedding)[:1536] for item in response.data]
            logger.info(f"Generated {len(embeddings)} embeddings")
            return embeddings
        except Exception as e:
            logger.error(f"Error generating text embeddings: {e}")
            return [np.zeros(1536) for _ in texts]
        
    def generate_summary_with_gpt(self, text: str) -> str:
        """GPT-4を使用してテキスト全体から20文字程度の要約を生成"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4-1106-preview",
                messages=[
                    {
                        "role": "system",
                        "content": "与えられたテキストの内容を20文字程度の簡潔な日本語で要約してください。"
                    },
                    {
                        "role": "user",
                        "content": text
                    }
                ],
                max_tokens=50,
                temperature=0.3
            )
            summary = response.choices[0].message.content.strip()
            
            # 20文字を超える場合は切る
            if len(summary) > 20:
                cut_positions = [pos for pos, char in enumerate(summary[:20]) 
                            if char in ['は', 'が', 'を', 'に', 'で', 'と', '、']]
                if cut_positions:
                    last_pos = max(cut_positions)
                    summary = summary[:last_pos + 1]
                else:
                    summary = summary[:17] + "..."
                    
            return summary
            
        except Exception as e:
            logger.error(f"Error generating GPT summary: {e}")
            return "テキストの要約"


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
                "content_id": structure_info.get('id'),
                "title": structure_info.get('title', {}).get('ja'),
                "title_en": structure_info.get('title', {}).get('en'),
                "page_number": structure_info.get('page')
            })

        # テキストの処理
        if group.text_path and os.path.exists(group.text_path):
            try:
                with open(group.text_path, 'r', encoding='utf-8') as f:
                    text_content = f.read()
                    
                    # GPT-4を使用して全体の要約を生成
                    summary = self.generate_summary_with_gpt(text_content)
                    content['text'] = summary
                    content['metadata'] = metadata

                    # embedding生成用には完全なテキストを使用
                    chunks = self.text_splitter.split_text(text_content)
                    logger.info(f"Split text into {len(chunks)} chunks")
                    
                    chunk_embeddings = self.get_text_embeddings(chunks)
                    if chunk_embeddings:
                        embeddings['text'] = np.mean(chunk_embeddings, axis=0)
                        embeddings['metadata'] = self.get_text_embeddings([json.dumps(metadata, ensure_ascii=False)])[0][:1024]
                    
                    embeddings['image'] = np.zeros(512)
                    
            except Exception as e:
                logger.error(f"Error in text processing: {e}")
                embeddings['text'] = np.zeros(1536)
                embeddings['metadata'] = np.zeros(1024)
                embeddings['image'] = np.zeros(512)

        # 画像の処理
        if group.image_path and os.path.exists(group.image_path):
            try:
                image = Image.open(group.image_path)
                content['image_path'] = os.path.basename(group.image_path)
                embeddings['image'] = self.get_image_embedding(image)
                logger.info(f"Processed image: {group.image_path}")
            except Exception as e:
                logger.error(f"Error processing image {group.image_path}: {e}")
                embeddings['image'] = np.zeros(512)
        else:
            embeddings['image'] = np.zeros(512)

        return {
            'content': content,
            'embeddings': embeddings
        }

    # 他のメソッドは変更なし
    def load_content_structure(self):
        """content_structure.jsonからコンテンツ構造を読み込む"""
        structure_path = os.path.join(self.folder_path, "content_structure.json")
        if os.path.exists(structure_path):
            with open(structure_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    @torch.no_grad()
    def get_image_embedding(self, image: Image.Image) -> np.ndarray:
        """画像のembedding生成"""
        try:
            inputs = self.clip_processor(images=image, return_tensors="pt")
            image_features = self.clip_model.get_image_features(**inputs)
            return image_features.squeeze().numpy()
        except Exception as e:
            logger.error(f"Error generating image embedding: {e}")
            return np.zeros(512)

def main():
    input_folder = f"./data/{input_folder_name}"
    logger.info(f"\nChecking input folder: {input_folder}")
    if os.path.exists(input_folder):
        files = os.listdir(input_folder)
        logger.info(f"Files found in input folder: {files}")
    else:
        logger.error("Input folder not found!")
        return

    output_folder = f"./embeddings/{output_folder_name}"
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        logger.info(f"Created embeddings folder at: {output_folder}")

    processor = ContentProcessor(input_folder, openai_api_key)
    content_groups = processor.find_content_groups()
    
    processed_contents = {}
    
    for idx, group in enumerate(content_groups, 1):
        logger.info(f"Processing group: {group.base_name} (#{idx})")
        try:
            result = processor.process_content_group(group, sequence_number=idx)
            processed_contents[group.base_name] = result
        except Exception as e:
            logger.error(f"Error processing group {group.base_name}: {e}")

    # 結果の保存
    np.save(
        os.path.join(output_folder, "combined_embeddings.npy"), 
        {name: data['embeddings'] for name, data in processed_contents.items()}
    )
    
    with open(os.path.join(output_folder, "content_metadata.json"), 'w', encoding='utf-8') as f:
        json.dump(
            {name: data['content'] for name, data in processed_contents.items()},
            f, 
            ensure_ascii=False, 
            indent=2
        )

    logger.info(f"\n処理が完了しました。")
    logger.info(f"結果は {output_folder} に保存されました：")
    logger.info(f"- {os.path.join(output_folder, 'combined_embeddings.npy')}")
    logger.info(f"- {os.path.join(output_folder, 'content_metadata.json')}")

if __name__ == "__main__":
    main()