"""
Лабораторная работа №3: CNN для классификации изображений (CIFAR-10).
TensorFlow/Keras: загрузка данных, архитектура, обучение, графики, анализ ошибок,
дополнительное сравнение (базовая модель vs модель без Dropout).
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

# --- 4.1. Загрузка и подготовка данных ---
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
y_train = y_train.flatten()
y_test = y_test.flatten()

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

CLASS_NAMES_RU = (
    "самолёт",
    "автомобиль",
    "птица",
    "кошка",
    "олень",
    "собака",
    "лягушка",
    "лошадь",
    "корабль",
    "грузовик",
)

# --- 4.2. Визуализация примеров из обучающей выборки ---
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
indices = np.random.RandomState(42).choice(len(x_train), size=10, replace=False)
for ax, idx in zip(axes.flat, indices):
    ax.imshow(x_train[idx])
    ax.set_title(CLASS_NAMES_RU[y_train[idx]])
    ax.axis("off")
plt.suptitle("Примеры изображений CIFAR-10 (обучающая выборка)")
plt.tight_layout()


def build_cnn(input_shape=(32, 32, 3), num_classes=10, use_dropout=True, filters=(32, 64)):
    """CNN: ≥2 Conv2D, Pooling, Flatten, Dense, выход на num_classes."""
    f1, f2 = filters
    model = keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(f1, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(f2, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
        ]
    )
    if use_dropout:
        model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation="softmax"))
    return model


EPOCHS = 15
BATCH_SIZE = 128

# --- 4.3–4.4. Базовая модель: компиляция и обучение ---
model_base = build_cnn(use_dropout=True)
model_base.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

print(model_base.summary())

history_base = model_base.fit(
    x_train,
    y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=0.1,
    verbose=1,
)

# --- 4.5. Графики accuracy и loss ---
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
epochs_range = range(1, len(history_base.history["accuracy"]) + 1)
ax1.plot(epochs_range, history_base.history["accuracy"], label="train")
ax1.plot(epochs_range, history_base.history["val_accuracy"], label="val")
ax1.set_xlabel("Эпоха")
ax1.set_ylabel("Accuracy")
ax1.set_title("Точность (базовая модель)")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(epochs_range, history_base.history["loss"], label="train")
ax2.plot(epochs_range, history_base.history["val_loss"], label="val")
ax2.set_xlabel("Эпоха")
ax2.set_ylabel("Loss")
ax2.set_title("Функция потерь (базовая модель)")
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.tight_layout()

test_loss, test_acc = model_base.evaluate(x_test, y_test, verbose=0)
print(f"\nТестовая выборка (базовая модель): loss={test_loss:.4f}, accuracy={test_acc:.4f}")

# --- 4.6. Примеры верных и ошибочных предсказаний ---
probs = model_base.predict(x_test, batch_size=BATCH_SIZE, verbose=0)
y_pred = np.argmax(probs, axis=1)
correct = np.where(y_pred == y_test)[0]
wrong = np.where(y_pred != y_test)[0]

rng = np.random.RandomState(0)
correct_sample = rng.choice(correct, size=min(6, len(correct)), replace=False)
wrong_sample = rng.choice(wrong, size=min(6, len(wrong)), replace=False)


def plot_predictions(indices, title):
    fig_p, axes_p = plt.subplots(2, 3, figsize=(10, 7))
    for ax, idx in zip(axes_p.flat, indices):
        ax.imshow(x_test[idx])
        true_name = CLASS_NAMES_RU[y_test[idx]]
        pred_name = CLASS_NAMES_RU[y_pred[idx]]
        conf = probs[idx, y_pred[idx]]
        ax.set_title(f"Истина: {true_name}\nПредсказание: {pred_name} ({conf:.2f})")
        ax.axis("off")
    fig_p.suptitle(title)
    plt.tight_layout()


plot_predictions(correct_sample, "Примеры верной классификации")
plot_predictions(wrong_sample, "Примеры ошибочных предсказаний")

print(
    "\nАнализ ошибок: на CIFAR-10 объекты мелкие (32×32), классы визуально близки "
    "(например, кошка/собака, самолёт/птица). Простая CNN с малым полем зрения "
    "часто путает текстуры и фон; без аугментации данных переобучение на train "
    "также ухудшает обобщение."
)

# --- 4.7. Дополнительное исследование: та же архитектура без Dropout ---
model_nodrop = build_cnn(use_dropout=False)
model_nodrop.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
history_nodrop = model_nodrop.fit(
    x_train,
    y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=0.1,
    verbose=1,
)
test_loss_nd, test_acc_nd = model_nodrop.evaluate(x_test, y_test, verbose=0)

fig3, (axa, axb) = plt.subplots(1, 2, figsize=(12, 4))
axa.plot(history_base.history["val_accuracy"], label="с Dropout (0.5)")
axa.plot(history_nodrop.history["val_accuracy"], label="без Dropout")
axa.set_xlabel("Эпоха")
axa.set_ylabel("Val accuracy")
axa.set_title("Сравнение: валидационная точность")
axa.legend()
axa.grid(True, alpha=0.3)

axb.plot(history_base.history["val_loss"], label="с Dropout (0.5)")
axb.plot(history_nodrop.history["val_loss"], label="без Dropout")
axb.set_xlabel("Эпоха")
axb.set_ylabel("Val loss")
axb.set_title("Сравнение: валидационный loss")
axb.legend()
axb.grid(True, alpha=0.3)
plt.tight_layout()

print("\n=== Сравнение (доп. исследование: Dropout) ===")
print(f"Базовая (Dropout 0.5) — тест accuracy: {test_acc:.4f}")
print(f"Без Dropout          — тест accuracy: {test_acc_nd:.4f}")

plt.show()
