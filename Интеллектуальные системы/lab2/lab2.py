import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import io

iris = datasets.load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y
df['species'] = df['target'].map(lambda v: target_names[v])

head = df.head()
shape = df.shape
buf = io.StringIO()
df.info(buf=buf)
info_text = buf.getvalue()
missing = df.isnull().sum()
describe = df.describe().T


print("Shape:", shape)
print("Info:")
print(info_text)
print("\nMissing values per column:")
print(missing)
print("\nDescriptive statistics:")

scatter_matrix(df[feature_names], diagonal='hist', alpha=0.7)
plt.suptitle("Scatter matrix of Iris features", y=0.95)
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(2,2, figsize=(10,8))
axes = axes.flatten()
for i, col in enumerate(feature_names):
    axes[i].hist(df[col], bins=15)
    axes[i].set_title(col)
plt.suptitle("Feature histograms")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,6))
df.boxplot(column=feature_names)
plt.title("Boxplots of features")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

knn = KNeighborsClassifier()
param_grid_knn = {'n_neighbors': list(range(1,16)), 'weights': ['uniform','distance']}
grid_knn = GridSearchCV(knn, param_grid_knn, cv=5, scoring='accuracy')
grid_knn.fit(X_train, y_train)
best_knn = grid_knn.best_estimator_

svc = SVC()
param_grid_svc = {'kernel': ['linear','rbf'], 'C': [0.1,1,10,50], 'gamma': ['scale','auto']}
grid_svc = GridSearchCV(svc, param_grid_svc, cv=5, scoring='accuracy')
grid_svc.fit(X_train, y_train)
best_svc = grid_svc.best_estimator_

y_pred_knn = best_knn.predict(X_test)
y_pred_svc = best_svc.predict(X_test)

metrics = []
for name, y_pred in [("KNN", y_pred_knn), ("SVC", y_pred_svc)]:
    acc = accuracy_score(y_test, y_pred)
    prec_macro = precision_score(y_test, y_pred, average='macro')
    rec_macro = recall_score(y_test, y_pred, average='macro')
    f1_macro = f1_score(y_test, y_pred, average='macro')
    report = classification_report(y_test, y_pred, target_names=target_names, digits=4)
    cm = confusion_matrix(y_test, y_pred)
    metrics.append({
        'model': name,
        'accuracy': acc,
        'precision_macro': prec_macro,
        'recall_macro': rec_macro,
        'f1_macro': f1_macro,
        'classification_report': report,
        'confusion_matrix': cm
    })

metrics_df = pd.DataFrame([{
    'model': m['model'],
    'accuracy': m['accuracy'],
    'precision_macro': m['precision_macro'],
    'recall_macro': m['recall_macro'],
    'f1_macro': m['f1_macro']
} for m in metrics]).set_index('model')


for m in metrics:
    cm = m['confusion_matrix']
    plt.figure(figsize=(5,4))
    plt.imshow(cm, interpolation='nearest')
    plt.title(f"Confusion matrix: {m['model']}")
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     verticalalignment="center",
                     fontsize=12)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()
    print(f"--- {m['model']} classification report ---")
    print(m['classification_report'])

report_text = []
report_text.append("Лабораторная работа 2 — Классификация набора Iris")
report_text.append("")
report_text.append("Постановка задачи:")
report_text.append("Классификация видов ирисов (setosa, versicolor, virginica) по четырём измеренным признакам.")
report_text.append("")
report_text.append("Результаты предварительного анализа:")
report_text.append(f"Форма данных: {shape}")
report_text.append("Пропусков нет.")
report_text.append("")
report_text.append("Выбранные модели и подбор гиперпараметров:")
report_text.append(f"KNN best params: {grid_knn.best_params_}")
report_text.append(f"SVC best params: {grid_svc.best_params_}")
report_text.append("")
report_text.append("Результаты на тестовой выборке (20%):")
for m in metrics:
    report_text.append(f"Модель: {m['model']}")
    report_text.append(f"Accuracy: {m['accuracy']:.4f}")
    report_text.append(f"Precision (macro): {m['precision_macro']:.4f}")
    report_text.append(f"Recall (macro): {m['recall_macro']:.4f}")
    report_text.append(f"F1 (macro): {m['f1_macro']:.4f}")
    report_text.append("")
report_text.append("Краткий вывод:")
if metrics[0]['f1_macro'] > metrics[1]['f1_macro']:
    report_text.append("KNN показал чуть лучшую F1-меру на тестовой выборке.")
elif metrics[0]['f1_macro'] < metrics[1]['f1_macro']:
    report_text.append("SVC показал чуть лучшую F1-меру на тестовой выборке.")
else:
    report_text.append("Обе модели показали одинаковую F1-меру на тестовой выборке.")