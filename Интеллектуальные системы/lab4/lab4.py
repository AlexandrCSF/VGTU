import time
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt

data = datasets.load_wine()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

dt = DecisionTreeClassifier(random_state=42)
start = time.time()
dt.fit(X_train, y_train)
dt_time = time.time() - start
y_pred_dt = dt.predict(X_test)
dt_acc = accuracy_score(y_test, y_pred_dt)
dt_f1 = f1_score(y_test, y_pred_dt, average='weighted')

best_rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
start = time.time()
best_rf.fit(X_train, y_train)
rf_time = time.time() - start
y_pred_rf = best_rf.predict(X_test)
rf_acc = accuracy_score(y_test, y_pred_rf)
rf_f1 = f1_score(y_test, y_pred_rf, average='weighted')

best_gb = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, max_depth=3, random_state=42)
start = time.time()
best_gb.fit(X_train, y_train)
gb_time = time.time() - start
y_pred_gb = best_gb.predict(X_test)
gb_acc = accuracy_score(y_test, y_pred_gb)
gb_f1 = f1_score(y_test, y_pred_gb, average='weighted')

results = pd.DataFrame([
    ['Decision Tree', dt_acc, dt_f1, dt_time],
    ['Random Forest', rf_acc, rf_f1, rf_time],
    ['Gradient Boosting', gb_acc, gb_f1, gb_time]
], columns=['model', 'accuracy', 'f1_weighted', 'train_time']).set_index('model')

print("\nResults:")
print(results)

plt.figure(figsize=(8,5))
plt.bar(results.index, results['accuracy'])
plt.title('Accuracy comparison')
plt.ylim(0,1)
plt.show()

plt.figure(figsize=(8,5))
plt.bar(results.index, results['f1_weighted'])
plt.title('F1-weighted comparison')
plt.ylim(0,1)
plt.show()

plt.figure(figsize=(8,5))
plt.bar(results.index, results['train_time'])
plt.title('Training time (seconds)')
plt.ylabel('Time (s)')
plt.show()
