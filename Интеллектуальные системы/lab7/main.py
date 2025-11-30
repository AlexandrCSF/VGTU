import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.layers import Dense

x = np.linspace(-3, 3, 1000)
y = x**3 - 3*x**2 + 2

def create_and_train_model(neurons, activation, x, y):
    model = keras.Sequential()
    model.add(Dense(neurons, input_dim=1, activation=activation))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(0.01))
    history = model.fit(x, y, epochs=500, verbose=0)
    return model

activations = ['linear', 'tanh', 'relu', 'sigmoid']

models = {}
for activation in activations:
    models[activation] = create_and_train_model(10, activation, x, y)

plt.figure(figsize=(15, 10))
plt.suptitle('Аппроксимация функции \(x^3 - 3x^2 + 2\) с различными функциями активации')

for i, activation in enumerate(activations, 1):
    plt.subplot(2, 2, i)
    plt.scatter(x, y, color='black', label='Исходная функция')
    y_pred = models[activation].predict(x)
    plt.plot(x, y_pred, color='magenta', linewidth=2, label='Аппроксимация')
    plt.title(f'Функция активации: {activation}')
    plt.legend()

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

neurons_list = [5, 10, 20]
plt.figure(figsize=(15, 10))
plt.suptitle('Аппроксимация функции \(x^3 - 3x^2 + 2\) с разным количеством нейронов')

for i, neurons in enumerate(neurons_list, 1):
    plt.subplot(2, 2, i)
    model = create_and_train_model(neurons, 'tanh', x, y)
    plt.scatter(x, y, color='black', label='Исходная функция')
    y_pred = model.predict(x)
    plt.plot(x, y_pred, color='magenta', linewidth=2, label=f'Нейронов: {neurons}')
    plt.legend()

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
