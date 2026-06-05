import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import Sequential
from keras.layers import SimpleRNN, GRU, LSTM, Dense, Dropout
from keras.optimizers import Adam


def f(x):
    return np.sin(x) + 0.3 * np.cos(x)


x_train = np.linspace(-3, 3, 300)
x_test = np.linspace(3, 4, 100)
x_all = np.linspace(-3, 4, 400)

y_train_true = f(x_train)
y_test_true = f(x_test)
y_all_true = f(x_all)

window = 200


def create_sequences(data, window):
    X, Y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i + window])
        Y.append(data[i + window])
    return np.array(X), np.array(Y)


X_train, Y_train = create_sequences(y_train_true, window)

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))


def train_and_predict(model_type, layers, dropout_rate=0.0):
    model = Sequential()

    if model_type == 'SimpleRNN':
        if layers > 1:
            model.add(SimpleRNN(32, return_sequences=True, input_shape=(window, 1)))
        else:
            model.add(SimpleRNN(32, input_shape=(window, 1)))
    elif model_type == 'GRU':
        if layers > 1:
            model.add(GRU(32, return_sequences=True, input_shape=(window, 1)))
        else:
            model.add(GRU(32, input_shape=(window, 1)))
    elif model_type == 'LSTM':
        if layers > 1:
            model.add(LSTM(32, return_sequences=True, input_shape=(window, 1)))
        else:
            model.add(LSTM(32, input_shape=(window, 1)))

    for _ in range(1, layers):
        if model_type == 'SimpleRNN':
            model.add(SimpleRNN(32, return_sequences=(_ < layers - 1)))
        elif model_type == 'GRU':
            model.add(GRU(32, return_sequences=(_ < layers - 1)))
        elif model_type == 'LSTM':
            model.add(LSTM(32, return_sequences=(_ < layers - 1)))

    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))

    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

    history = model.fit(X_train, Y_train, epochs=50, batch_size=16, verbose=0, validation_split=0.1)

    last_sequence = y_train_true[-window:].reshape(1, window, 1)
    predictions = []

    for _ in range(len(x_test)):
        pred = model.predict(last_sequence, verbose=0)[0, 0]
        predictions.append(pred)
        new_sequence = np.append(last_sequence[0, 1:, 0], pred).reshape(1, window, 1)
        last_sequence = new_sequence

    return model, history, np.array(predictions)


configs = [
    ('SimpleRNN', 1, 0.0),
    ('SimpleRNN', 2, 0.2),
    ('GRU', 1, 0.0),
    ('GRU', 2, 0.2),
    ('LSTM', 1, 0.0),
    ('LSTM', 2, 0.2),
]

results = {}
for model_type, layers, dropout in configs:
    print(f"Training {model_type} with layers={layers}, dropout={dropout}")
    model, history, preds = train_and_predict(model_type, layers, dropout)
    results[(model_type, layers, dropout)] = {
        'model': model,
        'history': history,
        'predictions': preds,
        'mse': mean_squared_error(y_test_true, preds),
        'mae': mean_absolute_error(y_test_true, preds)
    }

print("\n" + "=" * 60)
print("Сравнение качества прогноза на интервале [3,4]:")
print("=" * 60)
for key, val in results.items():
    model_type, layers, dropout = key
    print(f"{model_type} (layers={layers}, dropout={dropout}): MSE = {val['mse']:.6f}, MAE = {val['mae']:.6f}")

plt.figure(figsize=(14, 8))

plt.plot(x_all, y_all_true, 'k-', linewidth=2, label='Истинная функция f(x) = sin(x) + 0.3cos(x)')

colors = {
    'SimpleRNN': 'blue',
    'GRU': 'green',
    'LSTM': 'red'
}
markers = {
    1: 'o',
    2: 's'
}

for (model_type, layers, dropout), val in results.items():
    label = f"{model_type} (layers={layers}, dropout={dropout})"
    plt.plot(x_test, val['predictions'], '--', color=colors[model_type],
             marker=markers[layers], markevery=10, linewidth=1.5, alpha=0.8, label=label)

plt.axvline(x=3, color='gray', linestyle=':', linewidth=1.5, label='Граница обучения/прогноза')

plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Прогнозирование функции f(x) = sin(x) + 0.3cos(x) на интервале [3,4] (вне обучающей выборки)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 6))
for (model_type, layers, dropout), val in results.items():
    errors = np.abs(val['predictions'] - y_test_true)
    label = f"{model_type} (layers={layers}, dropout={dropout})"
    plt.plot(x_test, errors, '-', color=colors[model_type],
             marker=markers[layers], markevery=15, linewidth=1.5, alpha=0.7, label=label)

plt.xlabel('x')
plt.ylabel('Абсолютная ошибка прогноза')
plt.title('Абсолютные ошибки прогноза на интервале [3,4]')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\n" + "=" * 60)
print("Анализ устойчивости прогноза (MAE на подынтервалах [3,3.5] и [3.5,4]):")
print("=" * 60)

idx_split = len(x_test) // 2
for (model_type, layers, dropout), val in results.items():
    mae_first = mean_absolute_error(y_test_true[:idx_split], val['predictions'][:idx_split])
    mae_second = mean_absolute_error(y_test_true[idx_split:], val['predictions'][idx_split:])
    degradation = mae_second - mae_first
    print(f"{model_type} (layers={layers}, dropout={dropout}):")
    print(f"  MAE [3, 3.5] = {mae_first:.6f}, MAE [3.5, 4] = {mae_second:.6f}, ухудшение = {degradation:.6f}")