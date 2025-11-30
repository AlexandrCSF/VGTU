import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Dense

operands = []
results = []

for op1 in range(100):
    for op2 in range(100):
        operands.append([op1, op2])
        results.append(2 * op1 - 4 * op2)

operands = np.array(operands)
results = np.array(results)

model = keras.Sequential()
model.add(Dense(units=1, input_shape=(2,), activation='linear'))

model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(0.01))

history = model.fit(operands, results, epochs=50, verbose=1)

test_data = np.array([100, 50])
test_data_with_dims = np.expand_dims(test_data, axis=0)
predicted_result = model.predict(test_data_with_dims)

print(f"Результат предсказания для входных данных {test_data}: {predicted_result[0][0]}")

weights = model.get_weights()
print(f"Весовые коэффициенты: {weights}")
