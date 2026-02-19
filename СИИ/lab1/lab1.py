import numpy as np
import matplotlib.pyplot as plt

class Bandit:
    def __init__(self, arms, means):
        self.arms = arms
        self.means = means

    def pull(self, arm_index):
        return self.arms[arm_index]()

def greedy_strategy(Q, k, t):
    return np.argmax(Q)

def epsilon_greedy_strategy(Q, k, t, epsilon):
    if np.random.rand() < epsilon:
        return np.random.choice(len(Q))
    else:
        return np.argmax(Q)

def softmax_strategy(Q, k, t, tau):
    probabilities = np.exp(Q / tau) / np.sum(np.exp(Q / tau))
    return np.random.choice(len(Q), p=probabilities)

def pursuit_strategy(Q, k, t, beta):
    if t == 0:
        return np.random.choice(len(Q))
    best_action = np.argmax(Q)
    probabilities = np.ones(len(Q)) / len(Q)
    probabilities[best_action] += beta * (1 - probabilities[best_action])
    probabilities /= np.sum(probabilities)
    return np.random.choice(len(Q), p=probabilities)

def run_experiment(bandit, strategy, params, T=1000, N=100):
    total_rewards = np.zeros(T)
    optimal_action_counts = np.zeros(T)
    optimal_action = np.argmax(bandit.means)

    for _ in range(N):
        np.random.seed()
        Q = np.zeros(len(bandit.arms))
        k = np.zeros(len(bandit.arms))

        for t in range(T):
            action = strategy(Q, k, t, **params)
            reward = bandit.pull(action)
            total_rewards[t] += reward

            if action == optimal_action:
                optimal_action_counts[t] += 1

            k[action] += 1
            Q[action] += (reward - Q[action]) / k[action]

    avg_rewards = total_rewards / N
    optimal_action_percent = optimal_action_counts / N * 100

    return avg_rewards, optimal_action_percent

def plot_results(avg_rewards_list, optimal_action_percent_list, strategies_names):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    for avg_rewards in avg_rewards_list:
        plt.plot(avg_rewards)
    plt.title("Накопленное среднее вознаграждение")
    plt.xlabel("Шаги")
    plt.ylabel("Среднее вознаграждение")
    plt.legend(strategies_names)

    plt.subplot(1, 2, 2)
    for optimal_action_percent in optimal_action_percent_list:
        plt.plot(optimal_action_percent)
    plt.title("Доля выбора оптимального действия (%)")
    plt.xlabel("Шаги")
    plt.ylabel("Процент выбора")
    plt.legend(strategies_names)

    plt.tight_layout()
    plt.show()

# Пример распределений для рук бандита
arms = [
    lambda: np.random.normal(loc=0, scale=1),
    lambda: np.random.normal(loc=1, scale=1),
    lambda: np.random.normal(loc=2, scale=1)
]

means = [0, 1, 2]  # Математические ожидания для каждой руки
bandit = Bandit(arms, means)

# Параметры стратегий
params_list = [
    {"epsilon": 0.1},
    {"tau": 0.1},
    {"beta": 0.1}
]

# Запуск экспериментов
avg_rewards_list = []
optimal_action_percent_list = []
strategies_names = ["ε-жадная (ε=0.1)", "Softmax (τ=0.1)", "Метод преследования (β=0.1)"]

for params, strategy in zip(params_list, [epsilon_greedy_strategy, softmax_strategy, pursuit_strategy]):
    avg_rewards, optimal_action_percent = run_experiment(bandit, strategy, params)
    avg_rewards_list.append(avg_rewards)
    optimal_action_percent_list.append(optimal_action_percent)

# Построение графиков
plot_results(avg_rewards_list, optimal_action_percent_list, strategies_names)
