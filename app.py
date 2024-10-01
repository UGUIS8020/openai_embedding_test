import json
import os
from PIL import Image
import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModel
from openai import OpenAI
from dotenv import load_dotenv

# OpenAI API キーの設定
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

# 画像処理用の事前学習モデル
image_model_name = "microsoft/resnet-50"
image_extractor = AutoFeatureExtractor.from_pretrained(image_model_name)
image_model = AutoModel.from_pretrained(image_model_name)

# OpenAI クライアントの初期化
client = OpenAI(api_key=openai_api_key)

# ... (他のコードは変更なし)

def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=text
    )
    return response.data[0].embedding

def process_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    # JSONデータをテキストに変換
    text_representation = json.dumps(data)
    # OpenAI API を使用してJSONをベクトル化
    embedding = get_embedding(text_representation)
    return np.array(embedding)

def process_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    # OpenAI API を使用してテキストをベクトル化
    embedding = get_embedding(text)
    return np.array(embedding)

def process_image(file_path):
    image = Image.open(file_path)
    inputs = image_extractor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = image_model(**inputs)
    # 画像特徴の抽出（最後の隠れ層の平均を使用）
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embedding

data_directory = "./data/jpg"

def main():
    vectors = []
    for filename in os.listdir(data_directory):
        file_path = os.path.join(data_directory, filename)
        if filename.endswith('.json'):
            vectors.append(process_json(file_path))
        elif filename.endswith('.txt'):
            vectors.append(process_text(file_path))
        elif filename.endswith('.png'):
            vectors.append(process_image(file_path))
    
    # ベクトルの結合と保存
    combined_vector = np.vstack(vectors)
    np.save('combined_vector.npy', combined_vector)

if __name__ == "__main__":
    main()