import numpy as np

# ファイルからデータを読み込む
loaded_vectors = np.load('combined_vector.npy')

# データの形状を確認（行数 x 列数）
print(loaded_vectors.shape)

# 最初の数行を表示
print(loaded_vectors[:5])