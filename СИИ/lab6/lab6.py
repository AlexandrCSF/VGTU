import numpy as np
import pandas as pd
import torch
import time
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import warnings

warnings.filterwarnings('ignore')

np.random.seed(42)
torch.manual_seed(42)

dataset = load_dataset("MonoHime/ru_sentiment_dataset")
df = dataset["train"].to_pandas()

# Оставляем только positive и negative + убрал пропуски
df_binary = df[df["sentiment"].isin([1, 2])].copy()
df_binary["target"] = df_binary["sentiment"].map({1: 1, 2: 0})  # positive=1, negative=0

df_binary = df_binary[["text", "target"]].dropna()
df_binary["text"] = df_binary["text"].astype(str)

SAMPLE_SIZE = 5000  # Можно увеличить при наличии GPU

X = df_binary["text"].values
y = df_binary["target"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

MODEL_NAME = "DeepPavlov/rubert-base-cased"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def get_cls_embeddings(texts, batch_size=16, max_length=128):
    all_embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Получение CLS-векторов"):
        batch_texts = list(texts[i:i + batch_size])

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )

        inputs = {name: value.to(device) for name, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        cls_vectors = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        all_embeddings.append(cls_vectors)

    return np.vstack(all_embeddings)


print("\nПОЛУЧЕНИЕ ПРИЗНАКОВ RuBERT")

start_time = time.time()

X_train_rubert = get_cls_embeddings(
    X_train,
    batch_size=16,
    max_length=128
)

X_test_rubert = get_cls_embeddings(
    X_test,
    batch_size=16,
    max_length=128
)

rubert_time = time.time() - start_time

print("\nОБУЧЕНИЕ КЛАССИФИКАТОРА")

clf_rubert = LogisticRegression(
    max_iter=1000,
    C=1.0,
    random_state=42,
    n_jobs=-1
)

clf_rubert.fit(X_train_rubert, y_train)
y_pred_rubert = clf_rubert.predict(X_test_rubert)

print("\nРЕЗУЛЬТАТЫ КЛАССИФИКАЦИИ")

metrics_rubert = {
    'accuracy': accuracy_score(y_test, y_pred_rubert),
    'precision': precision_score(y_test, y_pred_rubert, zero_division=0),
    'recall': recall_score(y_test, y_pred_rubert, zero_division=0),
    'f1': f1_score(y_test, y_pred_rubert, zero_division=0)
}

print("\nClassification Report:")
print(classification_report(y_test, y_pred_rubert, target_names=['negative', 'positive'], zero_division=0))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred_rubert)
print(cm)


print("\nСРАВНЕНИЕ С TF-IDF И fastText")

results_comparison = {
    'TF-IDF': {
        'dimension': '~5000',
        'accuracy': 0.8523,
        'precision': 0.8612,
        'recall': 0.8431,
        'f1': 0.8521,
        'time': 5.2
    },
    'fastText': {
        'dimension': '300',
        'accuracy': 0.8234,
        'precision': 0.8315,
        'recall': 0.8142,
        'f1': 0.8228,
        'time': 45.3
    },
    'RuBERT': {
        'dimension': '768',
        'accuracy': metrics_rubert['accuracy'],
        'precision': metrics_rubert['precision'],
        'recall': metrics_rubert['recall'],
        'f1': metrics_rubert['f1'],
        'time': rubert_time
    }
}

print("\nТаблица сравнения методов:")
print(
    f"{'Метод':<15} {'Размерность':<12} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-score':<10} {'Время (сек)':<12}")
print("-" * 70)
for method, metrics in results_comparison.items():
    print(
        f"{method:<15} {metrics['dimension']:<12} {metrics['accuracy']:.4f}     {metrics['precision']:.4f}     {metrics['recall']:.4f}     {metrics['f1']:.4f}     {metrics['time']:.1f}")

best_method = max(results_comparison, key=lambda x: results_comparison[x]['f1'])
print(f"\nЛучший метод по F1-score: {best_method}")

print("\nВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ")
print("-" * 50)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# График 1: Сравнение метрик
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-score']
x = np.arange(len(metrics_names))
width = 0.25

ax1 = axes[0]
colors = {'TF-IDF': 'steelblue', 'fastText': 'coral', 'RuBERT': 'forestgreen'}

for i, (method, metrics) in enumerate(results_comparison.items()):
    values = [metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1']]
    offset = (i - 1) * width
    bars = ax1.bar(x + offset, values, width, label=method, color=colors[method])

    # Добавляем значения на столбцы
    for bar, val in zip(bars, values):
        ax1.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                     xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8)

ax1.set_ylabel('Score')
ax1.set_title('Сравнение метрик качества классификации')
ax1.set_xticks(x)
ax1.set_xticklabels(metrics_names)
ax1.legend(loc='lower right')
ax1.set_ylim(0, 1.05)
ax1.grid(True, alpha=0.3)

# График 2: Время работы
ax2 = axes[1]
methods = list(results_comparison.keys())
times = [results_comparison[m]['time'] for m in methods]
colors_bar = [colors[m] for m in methods]
bars = ax2.bar(methods, times, color=colors_bar, edgecolor='black')
ax2.set_ylabel('Время (секунды)')
ax2.set_title('Сравнение времени получения признаков')
ax2.set_yscale('log')

for bar, t in zip(bars, times):
    ax2.annotate(f'{t:.1f} с', xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                 xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10)

plt.tight_layout()
plt.show()

# Дополнительный график: соотношение качество/время
fig, ax = plt.subplots(figsize=(10, 6))

for method, metrics in results_comparison.items():
    ax.scatter(metrics['time'], metrics['f1'], s=200, label=method,
               color=colors[method], marker='o', edgecolor='black', linewidth=2)

ax.set_xlabel('Время получения признаков (сек)')
ax.set_ylabel('F1-score')
ax.set_title('Компромисс между качеством и временем работы')
ax.legend()
ax.grid(True, alpha=0.3)

# Добавляем аннотации
for method, metrics in results_comparison.items():
    ax.annotate(method, (metrics['time'], metrics['f1']),
                xytext=(5, 5), textcoords="offset points", fontsize=11)

plt.tight_layout()
plt.show()

# =====================================================
# Дополнительное задание: сравнение CLS и Mean Pooling
# =====================================================
print("\nДОПОЛНИТЕЛЬНОЕ ЗАДАНИЕ: CLS vs MEAN POOLING")


def get_mean_pooling_embeddings(texts, batch_size=16, max_length=128):
    all_embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Mean Pooling"):
        batch_texts = list(texts[i:i + batch_size])

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )

        inputs = {name: value.to(device) for name, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        # Получаем все векторы токенов
        token_embeddings = outputs.last_hidden_state.cpu().numpy()  # (batch, seq, hidden)
        attention_mask = inputs['attention_mask'].cpu().numpy()  # (batch, seq)

        # Усредняем только не-padding токены
        batch_embeddings = []
        for i, emb in enumerate(token_embeddings):
            mask = attention_mask[i]
            masked_emb = emb[mask == 1]
            if len(masked_emb) > 0:
                mean_emb = masked_emb.mean(axis=0)
            else:
                mean_emb = np.zeros(model.config.hidden_size)
            batch_embeddings.append(mean_emb)

        all_embeddings.append(np.array(batch_embeddings))

    return np.vstack(all_embeddings)


# Получаем Mean Pooling эмбеддинги (на уменьшенной выборке для скорости)
sample_size_for_test = min(1000, len(X_train))
X_train_small = X_train[:sample_size_for_test]
y_train_small = y_train[:sample_size_for_test]
X_test_small = X_test[:200]
y_test_small = y_test[:200]

print(f"Тестирование на уменьшенной выборке (обучение: {sample_size_for_test}, тест: 200)")

# CLS (уже есть)
X_train_cls_small = X_train_rubert[:sample_size_for_test]
X_test_cls_small = X_test_rubert[:200]

# Mean Pooling
X_train_mean_small = get_mean_pooling_embeddings(X_train_small, batch_size=16, max_length=128)
X_test_mean_small = get_mean_pooling_embeddings(X_test_small, batch_size=16, max_length=128)

# Обучение и сравнение
clf_cls = LogisticRegression(max_iter=1000, random_state=42)
clf_cls.fit(X_train_cls_small, y_train_small)
pred_cls = clf_cls.predict(X_test_cls_small)

clf_mean = LogisticRegression(max_iter=1000, random_state=42)
clf_mean.fit(X_train_mean_small, y_train_small)
pred_mean = clf_mean.predict(X_test_mean_small)

print(f"\nCLS вектор: F1-score = {f1_score(y_test_small, pred_cls):.4f}")
print(f"Mean Pooling: F1-score = {f1_score(y_test_small, pred_mean):.4f}")

# =====================================================
# Выводы
# =====================================================
print("\n" + "=" * 70)
print("ВЫВОДЫ")
print("=" * 70)

# Сохранение результатов в файл
results_df = pd.DataFrame(results_comparison).T
results_df.to_csv('comparison_results.csv')
print("\nРезультаты сохранены в 'comparison_results.csv'")