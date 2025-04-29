# 安装依赖
#!pip install sentence-transformers pandas scikit-learn matplotlib

import pandas as pd
from sentence_transformers import SentenceTransformer

# 示例数据集（新闻标题）
articles = [
    "Apple unveils new iPhone with advanced AI camera",
    "Google announces breakthrough in quantum computing",
    "Microsoft acquires robotics startup for $5 billion",
    "Apple stock surges after record quarterly earnings",
    "Google launches new privacy-focused search algorithm",
    "Amazon invests $10B in renewable energy projects"
]
df = pd.DataFrame({"text": articles})


# 加载嵌入模型（网页1、网页7）
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = model.encode(df['text'].tolist(), convert_to_tensor=False)

# 输出嵌入维度示例（384维）
print(f"Embedding shape: {embeddings.shape}")  # (6, 384)


from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# K-means聚类（网页1、网页3）
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(embeddings)

# PCA降维可视化
pca = PCA(n_components=2)
vis_embeddings = pca.fit_transform(embeddings)

plt.scatter(vis_embeddings[:,0], vis_embeddings[:,1], c=clusters)
plt.title("Article Clusters Visualization (PCA)")
plt.show()



# 添加聚类标签到数据框
df['cluster'] = clusters
print(df)

# 评估聚类质量（网页2）
from sklearn.metrics import silhouette_score
score = silhouette_score(embeddings, clusters)
print(f"Silhouette Score: {score:.2f}")  # 示例输出0.65


