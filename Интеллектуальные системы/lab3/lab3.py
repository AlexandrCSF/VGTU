import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

iris = datasets.load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y_true = iris.target
target_names = iris.target_names

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
y_pred = kmeans.fit_predict(X)

fig, axes = plt.subplots(1, 2, figsize=(12,5))
axes[0].scatter(X.iloc[:, 0], X.iloc[:, 1], c=y_pred, cmap='viridis', s=50)
axes[0].scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], c='red', marker='x', s=100)
axes[0].set_xlabel('sepal length (cm)')
axes[0].set_ylabel('sepal width (cm)')
axes[0].set_title('Кластеры по sepal length/width')

axes[1].scatter(X.iloc[:, 2], X.iloc[:, 3], c=y_pred, cmap='viridis', s=50)
axes[1].scatter(kmeans.cluster_centers_[:,2], kmeans.cluster_centers_[:,3], c='red', marker='x', s=100)
axes[1].set_xlabel('petal length (cm)')
axes[1].set_ylabel('petal width (cm)')
axes[1].set_title('Кластеры по petal length/width')
plt.show()

mismatch = y_pred != y_true
plt.figure(figsize=(8,6))
plt.scatter(X.iloc[:, 2][~mismatch], X.iloc[:, 3][~mismatch], c='blue', label='Правильно', alpha=0.6)
plt.scatter(X.iloc[:, 2][mismatch], X.iloc[:, 3][mismatch], c='red', label='Ошибки', alpha=0.8)
plt.xlabel('petal length (cm)')
plt.ylabel('petal width (cm)')
plt.title('Сравнение с истинными метками')
plt.legend()
plt.show()
