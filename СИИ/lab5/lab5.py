import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import warnings

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datasets import load_dataset

np.random.seed(42)

# Загрузка датасета
dataset = load_dataset("MonoHime/ru_sentiment_dataset")
df = pd.DataFrame(dataset['train'])

# Определение столбцов
text_column = 'text'
label_column = 'sentiment'

# Фильтрация классов (1 - positive, 2 - negative)
df_binary = df[df[label_column].isin([1, 2])].copy()
df_binary['target'] = df_binary[label_column].map({1: 1, 2: 0})

print(f"Размер выборки: {len(df_binary)}")
print(df_binary['target'].value_counts())

X = df_binary[text_column].values
y = df_binary['target'].values

# Предобработка текста
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

X_processed = [preprocess_text(t) for t in X]

# Разделение выборки
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42, stratify=y
)

# TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2, max_df=0.8)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Обучение классификатора
clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X_train_tfidf, y_train)
y_pred = clf.predict(X_test_tfidf)

# Метрики
metrics = {
    'accuracy': accuracy_score(y_test, y_pred),
    'precision': precision_score(y_test, y_pred, zero_division=0),
    'recall': recall_score(y_test, y_pred, zero_division=0),
    'f1': f1_score(y_test, y_pred, zero_division=0)
}

print("\nTF-IDF Results:")
print(f"Accuracy:  {metrics['accuracy']:.4f}")
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall:    {metrics['recall']:.4f}")
print(f"F1-score:  {metrics['f1']:.4f}")

# График
fig, ax = plt.subplots(figsize=(8, 5))

metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-score']
values = [metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1']]

bars = ax.bar(metrics_names, values, color='steelblue', edgecolor='black')
ax.set_ylim(0, 1)
ax.set_ylabel('Score')
ax.set_title('TF-IDF + Logistic Regression')

for bar, val in zip(bars, values):
    ax.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10)

plt.tight_layout()
plt.show()